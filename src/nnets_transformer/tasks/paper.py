"""Paper task preparation; scoring labels always belong to the clean prompt."""
from dataclasses import asdict, replace
import copy
import csv
import hashlib
import json
import random
from pathlib import Path
from .sources import source, module, SHA256
from nnets_transformer.records import Example, Scoring

CANDIDATE_POLICY = {'ioi': 'single-token-continuation-of-clean-prompt',
    'gendered_pronoun': 'single-token-continuation-of-clean-prompt',
    'greater_than': 'standalone-two-digit-vocabulary-token-after-century',
    'sva': 'exact-word-boundary-single-token-verb-forms'}
VERSIONS = {name: f'tigges-2024-{name}-v1' for name in ('ioi', 'gendered_pronoun', 'greater_than', 'sva')}


def encode(tokenizer, text):
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def continuation(tokenizer, prompt, answer):
    prefix = encode(tokenizer, prompt)
    full = encode(tokenizer, prompt+answer)
    if full[:len(prefix)] != prefix or len(full) != len(prefix)+1:
        raise ValueError(f'Protocol requires an exact single-token continuation: {answer!r}')
    return full[-1]


def pair(adapter, task, identity, clean, corrupt, targets, contrasts, roles):
    good, bad = encode(adapter.tokenizer, clean), encode(adapter.tokenizer, corrupt)
    if len(good) != len(bad):
        raise ValueError('All-position donor alignment requires equal-length paired tokenization')
    metric = 'logit_difference' if task in {'ioi', 'gendered_pronoun'} else 'probability_group_difference'
    example = Example(identity, good, (True,)*len(good), Scoring(metric, len(good)-1, tuple(targets), tuple(contrasts)),
                      {**roles, 'clean_text': clean, 'corrupt_text': corrupt, 'alignment': 'identity-all-positions'},
                      corrupted_tokens=bad, corrupted_mask=(True,)*len(bad))
    example.validate(adapter.vocab_size, adapter.context_length)
    example.corrupted().validate(adapter.vocab_size, adapter.context_length)
    return example


class _GeneratorTokenizer:
    """Source generators assume no automatic special tokens; adapters may prepend BOS by default."""
    def __init__(self, tokenizer):
        self._tokenizer = copy.deepcopy(tokenizer)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, *args, **kwargs):
        kwargs['add_special_tokens'] = False
        return self._tokenizer(*args, **kwargs)

    def encode(self, *args, **kwargs):
        kwargs['add_special_tokens'] = False
        return self._tokenizer.encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


class PaperTask:
    def __init__(self, name):
        self.name = name

    def prepare(self, adapter, config):
        root = source(config.source_directory)
        rng = random.Random(config.seed)
        # Over-generate so calibration can exclude any prompt shared with evaluation.
        count = config.examples+4*config.calibration_examples+32
        tokenizer = adapter.tokenizer
        if self.name == 'ioi':
            # The source generator mutates global RNGs. Restore caller states.
            import numpy as np
            import torch
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            try:
                original = module(root/'path_patching_cm/ioi_dataset.py')
                # The generator pads its own tensors; padding is never used by our retokenized prompts.
                generator_tokenizer = _GeneratorTokenizer(tokenizer)
                dataset = original.IOIDataset(prompt_type='mixed', N=count, tokenizer=generator_tokenizer,
                                               prepend_bos=False, seed=config.seed, device='cpu')
                flipped = original.gen_flipped_prompts(dataset.ioi_prompts, dataset.templates_by_prompt,
                    'ABB->ABA, BAB->BAA', original.NAMES, config.seed+sum(map(ord, 'ABB->ABA, BAB->BAA')))
                examples = []
                for index, (good, bad) in enumerate(zip(dataset.ioi_prompts, flipped)):
                    answer = ' '+good['IO']
                    # Released texts include the answer; never include it in the prompt.
                    if not good['text'].endswith(answer):
                        raise ValueError('IOI source answer boundary changed')
                    clean = good['text'][:-len(answer)]
                    # The source flip rewrites only the first three names; the final word stays the clean IO.
                    if not bad['text'].endswith(answer):
                        raise ValueError('IOI corrupt answer boundary changed')
                    corrupt = bad['text'][:-len(answer)]
                    targets = [continuation(tokenizer, clean, answer)]
                    contrasts = [continuation(tokenizer, clean, ' '+good['S'])]
                    roles = {key: int(value[index]) for key, value in dataset.word_idx.items() if key in {'IO', 'S1', 'S2'}}
                    roles['template'] = dataset.templates_by_prompt[index]
                    examples.append(pair(adapter, self.name, f'ioi-{index:06d}', clean, corrupt, targets, contrasts, roles))
            finally:
                random.setstate(python_state); np.random.set_state(numpy_state); torch.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
        elif self.name == 'gendered_pronoun':
            rows = list(csv.DictReader((root/'data/gender-pronoun.csv').read_text().splitlines()))
            indices = list(range(len(rows))); rng.shuffle(indices)
            if config.examples > len(rows):
                raise ValueError('Requested pronoun evaluation exceeds released 100 rows')
            # IDs are the released Pythia labels, translated to strings before retokenization.
            labels = {'344': ' he', '703': ' she'}
            examples = []
            for index in indices:
                row = rows[index]
                target = continuation(tokenizer, row['clean'], labels[row['clean_answer_idx']])
                contrast = continuation(tokenizer, row['clean'], labels[row['corrupted_answer_idx']])
                examples.append(pair(adapter, self.name, f'pronoun-row-{index}', row['clean'], row['corrupted'],
                                     [target], [contrast], {'target_string': labels[row['clean_answer_idx']]}))
        elif self.name == 'greater_than':
            original = module(root/'data/greater_than_dataset.py')
            valid = original.get_valid_years(tokenizer, 1100, 1800).tolist()
            by_suffix = {suffix: [y for y in valid if y % 100 == suffix] for suffix in range(2, 99)}
            if any(not years for years in by_suffix.values()):
                raise ValueError('Tokenizer cannot represent the balanced paper year protocol')
            # Source convention: each suffix is the standalone two-digit vocabulary token
            # predicted after the century token. Some centuries' canonical BPE would merge
            # century+suffix; count those non-canonical candidates instead of hiding them.
            candidates = []
            for suffix in range(100):
                tokens = encode(tokenizer, f'{suffix:02d}')
                if len(tokens) != 1 or tokenizer.decode(tokens) != f'{suffix:02d}':
                    raise ValueError('Year suffixes must be exact single vocabulary tokens')
                candidates.append(tokens[0])
            if len(set(candidates)) != 100:
                raise ValueError('Year candidates must be distinct single tokens')
            examples = []
            for index in range(count):
                boundary = index % 97+2
                year = rng.choice(by_suffix[boundary]); noun = rng.choice(original.default_nouns)
                clean = original.generate_real_sentence(noun, year)
                corrupt = original.generate_bad_sentence(noun, year)
                prompt = encode(tokenizer, clean)
                if tokenizer.decode(prompt[-1:]) != f' {year//100}':
                    raise ValueError('Greater-than prompt must end with the century token')
                canonical = sum(encode(tokenizer, clean+f'{s:02d}') == prompt+(candidates[s],) for s in range(100))
                examples.append(pair(adapter, self.name, f'year-{index:06d}', clean, corrupt,
                    candidates[boundary+1:], candidates[:boundary+1], {'clean_year': year, 'boundary': boundary,
                    'noun': noun, 'canonical_candidate_continuations': canonical}))
        else:
            verbs = list(csv.DictReader((root/'data/sva/combined_verb_list.csv').read_text().splitlines()))
            def verb_tokens(column):
                result = set()
                for row in verbs:
                    tokens = encode(tokenizer, ' '+row[column])
                    if len(tokens) == 1 and tokenizer.decode(tokens) == ' '+row[column]:
                        result.add(tokens[0])
                return tuple(sorted(result))
            singular, plural = verb_tokens('sing'), verb_tokens('plur')
            if not singular or not plural or set(singular)&set(plural):
                raise ValueError('SVA verb groups must be nonempty and disjoint')
            candidates = []
            for path in sorted((root/'data/sva/sva_raw_data').glob('ML-*.csv')):
                rows = list(csv.DictReader(path.read_text().splitlines()))
                singular_rows = [row['sentence'] for row in rows if row['label'] == '0']
                plural_rows = [row['sentence'] for row in rows if row['label'] == '1']
                if len(singular_rows) != len(plural_rows):
                    raise ValueError('Released SVA construction has unbalanced pairs')
                for index, (sing, plur) in enumerate(zip(singular_rows, plural_rows)):
                    if len(encode(tokenizer, sing)) != len(encode(tokenizer, plur)):
                        continue
                    for number, clean, corrupt, targets, contrasts in [('singular', sing, plur, singular, plural), ('plural', plur, sing, plural, singular)]:
                        candidates.append((f'{path.stem}-{index}-{number}', clean, corrupt, targets, contrasts,
                                           {'number': number, 'construction': path.stem, 'pair_id': f'{path.stem}-{index}'}))
            rng.shuffle(candidates)
            # Keep both directions of each source pair in one split to prevent leakage.
            examples = []
            used_pairs = set()
            for identity, clean, corrupt, targets, contrasts, roles in candidates:
                if roles['pair_id'] in used_pairs:
                    continue
                used_pairs.add(roles['pair_id'])
                examples.append(pair(adapter, self.name, identity, clean, corrupt, targets, contrasts, roles))
                if len(examples) == count:
                    break
        if len(examples) < config.examples:
            raise ValueError(f'Insufficient eligible {self.name} examples for the requested evaluation size')
        evaluation = examples[:config.examples]
        # Calibration prompts (forwarded clean prompts) never equal any evaluation clean/corrupt prompt.
        evaluation_tokens = {e.tokens for e in evaluation} | {e.corrupted_tokens for e in evaluation}
        pool, seen, excluded = [], set(), 0
        for example in examples[config.examples:]:
            if example.tokens in evaluation_tokens or example.tokens in seen:
                excluded += 1
                continue
            seen.add(example.tokens)
            pool.append(example)
        # Position-aligned means need every evaluation position covered: take covering prompts
        # first (at most a quarter of the budget), then fill in seeded order.
        longest = max(len(e.tokens) for e in evaluation)
        covering = [e for e in pool if len(e.tokens) >= longest][:max(1, config.calibration_examples//4)]
        chosen = {e.id for e in covering}
        selected = covering+[e for e in pool if e.id not in chosen]
        calibration = [replace(e, split='calibration') for e in selected[:config.calibration_examples]]
        if len(calibration) < config.calibration_examples:
            raise ValueError(f'Only {len(calibration)} leakage-free {self.name} calibration examples; '
                             'reduce task.calibration_examples or task.examples')
        manifest = {'dataset_version': VERSIONS[self.name], 'task': self.name, 'scoring_version': VERSIONS[self.name],
            'units': 'logits' if self.name in {'ioi', 'gendered_pronoun'} else 'probability',
            'higher_is_better': True, 'seed': config.seed, 'sample_count': len(evaluation),
            'source': {'url': 'NeurIPS-2024-official-supplement', 'sha256': SHA256},
            'tokenizer_revision': adapter.provenance['tokenizer_revision'],
            'sample_manifest': [asdict(e) for e in evaluation], 'calibration_manifest': [asdict(e) for e in calibration],
            'alignment': 'identity-all-positions', 'clean_labels_preserved': True,
            'calibration_policy': 'clean-prompts-disjoint-from-evaluation-clean-and-corrupt-v1',
            'calibration_excluded_overlaps': excluded,
            'calibration_position_coverage': [sum(len(e.tokens) > i for e in calibration) for i in range(longest)],
            'candidate_policy': CANDIDATE_POLICY[self.name],
            'candidate_group_sizes': sorted({(len(e.scoring.targets), len(e.scoring.contrasts)) for e in evaluation}),
            'subset': 'paper-sized-seeded-subset' if config.examples == (70 if self.name in {'ioi','gendered_pronoun'} else 200) else 'development'}
        manifest['dataset_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        return evaluation, manifest

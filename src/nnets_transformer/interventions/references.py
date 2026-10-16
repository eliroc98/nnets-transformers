"""Checkpoint-specific streaming means and deterministic aligned donor preparation."""
import random
import torch
from nnets_transformer.records import Example, Scoring, Selection
from .replacements import Zero, TensorReplacement
from nnets_transformer.metrics.behavioral import score_batch

MEAN_VERSIONS = {"position": "position-aligned-example-mean-v1", "global": "global-valid-position-mean-v1"}


def restore(record):
    data = dict(record)
    data['scoring'] = Scoring(**{k: tuple(v) if isinstance(v, list) else v for k, v in data['scoring'].items()})
    for name in ('tokens', 'attention_mask', 'corrupted_tokens', 'corrupted_mask'):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    return Example(**data)


def convention(config):
    """Replacement provenance stored in the dataset manifest and run identity."""
    result = {"replacement": config.name, "direction": config.direction}
    if config.name == "mean":
        result["mean"] = MEAN_VERSIONS[config.mean_reduction]
    if config.name == "donor":
        result.update(donor_policy=config.donor_policy, donor_seed=config.donor_seed if config.donor_policy == "resample" else None)
    return result


def donors(examples, config):
    if config.name != 'donor':
        return {}
    if config.donor_policy == 'matched':
        if any(e.corrupted_tokens is None for e in examples):
            raise ValueError('Matched donors require task-defined clean/corrupt pairs')
        return {e.id: {'example_id': e.id, 'condition': 'corrupt' if config.direction == 'noising' else 'clean',
                       'alignment': 'identity-all-positions'} for e in examples}
    rng = random.Random(config.donor_seed)
    result = {}
    for example in examples:
        pool = [other for other in examples if other.id != example.id and other.tokens != example.tokens
                and len(other.tokens) == len(example.tokens) and other.attention_mask == example.attention_mask]
        if not pool:
            raise ValueError(f'No nonself, length/mask-aligned donor for {example.id}')
        donor = rng.choice(pool)
        result[example.id] = {'example_id': donor.id, 'condition': 'clean', 'alignment': 'identity-all-positions'}
    return result


class PositionMean:
    """Per-position calibration mean, padded/truncated to the recipient's sequence axis."""
    def __init__(self, means):
        self.means = means

    def apply(self, value, selection, component):
        source = self.means[selection.hook]
        length = value.shape[component.sequence_axis]
        if source.shape[0] < length:
            # Positions beyond calibration coverage are right padding (validated before use).
            source = torch.cat((source, source.new_zeros((length-source.shape[0], *source.shape[1:]))))
        TensorReplacement({selection.hook: source[:length][None]}).apply(value, selection, component)


class References:
    def __init__(self, adapter, examples, dataset, sources, config, job):
        self.adapter, self.config, self.job = adapter, config, job
        self.examples = {e.id: e for e in examples}
        self.plan = dataset.get('donor_manifest', {})
        # Replacement values are needed at every intervened hook, observed or not.
        self.hooks = list(dict.fromkeys(s.hook for s in sources))
        self.means, self.counts = {}, {}
        if config.name == 'mean':
            calibration = [restore(e) for e in dataset.get('calibration_manifest', [])]
            if not calibration or any(e.split != 'calibration' for e in calibration):
                raise ValueError('Mean requires declared disjoint calibration samples')
            evaluation = {e.tokens for e in examples} | {e.corrupted_tokens for e in examples if e.corrupted_tokens}
            if {e.tokens for e in calibration} & evaluation:
                raise ValueError('Calibration/evaluation overlap')
            hooks = [Selection(h) for h in self.hooks]
            job.stage('mean_calibration', reduction=MEAN_VERSIONS[config.mean_reduction], examples=len(calibration))
            for index, example in enumerate(calibration):
                _, cache = adapter.forward(example, hooks)
                valid = torch.tensor(example.attention_mask, dtype=torch.bool)
                for hook, value in cache.items():
                    value = value[0].float()
                    mask = valid.to(value.device)
                    if config.mean_reduction == 'global':
                        total, count = value[mask].sum(0), mask.sum()
                    else:
                        total = value*mask.view(-1, *[1]*(value.ndim-1))
                        count = mask.float()
                    if hook in self.means and config.mean_reduction == 'position':
                        old, old_count = self.means[hook], self.counts[hook]
                        size = max(len(old), len(total))
                        pad = lambda t: torch.cat((t, t.new_zeros((size-len(t), *t.shape[1:])))) if len(t) < size else t
                        total, count = pad(old)+pad(total), pad(old_count)+pad(count)
                    elif hook in self.means:
                        total, count = self.means[hook]+total, self.counts[hook]+count
                    self.means[hook], self.counts[hook] = total, count
                job.progress('calibration_examples', index+1, len(calibration))
            if config.mean_reduction == 'global':
                self.means = {hook: value/self.counts[hook] for hook, value in self.means.items()}
            else:
                longest = max(len(e.tokens) for e in examples)
                for hook, total in self.means.items():
                    count = self.counts[hook]
                    covered = int((count > 0).sum())
                    if covered < longest or bool((count[:longest] == 0).any()):
                        raise ValueError(f'Position-aligned mean covers {covered} positions; evaluation needs {longest}')
                    self.means[hook] = total/count.clamp_min(1).view(-1, *[1]*(total.ndim-1))
                self.coverage = {hook: int(count.min()) for hook, count in self.counts.items()}
                job.event('mean_coverage', minimum_examples_per_position=min(self.coverage.values()))

    def batch(self, examples, observations, length):
        hooks = list(dict.fromkeys([*(s.hook for s in observations), *self.hooks]))
        cached = [Selection(h) for h in hooks]
        clean_logits, clean_cache = self.adapter.forward_batch(examples, cached, pad_length=length)
        clean_scores = score_batch(clean_logits, [e.scoring for e in examples]).cpu().tolist()
        del clean_logits
        paired = all(e.corrupted_tokens is not None for e in examples)
        if self.config.direction == 'denoising' and not paired:
            raise ValueError('Denoising requires a corrupted recipient')
        corrupt_cache, corrupt_scores = None, [None]*len(examples)
        if paired:
            corrupt_logits, corrupt_cache = self.adapter.forward_batch([e.corrupted() for e in examples], cached, pad_length=length)
            # Corrupt runs keep the clean answer definitions.
            corrupt_scores = score_batch(corrupt_logits, [e.scoring for e in examples]).cpu().tolist()
            del corrupt_logits
        if self.config.name == 'zero':
            replacements = [Zero()]*len(examples)
        elif self.config.name == 'mean':
            value = PositionMean(self.means) if self.config.mean_reduction == 'position' else \
                TensorReplacement({k: v[None, None] for k, v in self.means.items()}, broadcast=True)
            replacements = [value]*len(examples)
        elif self.config.donor_policy == 'matched':
            cache = corrupt_cache if self.config.direction == 'noising' else clean_cache
            replacements = [TensorReplacement({k: v[i:i+1] for k, v in cache.items()}) for i in range(len(examples))]
        else:
            # Resampled donors are forwarded together; each recipient gets its own donor slice.
            donor_examples = [self.examples[self.plan[e.id]['example_id']] for e in examples]
            _, cache = self.adapter.forward_batch(donor_examples, [Selection(h) for h in self.hooks], pad_length=length)
            replacements = [TensorReplacement({k: v[i:i+1] for k, v in cache.items()}) for i in range(len(examples))]
        recipients = examples if self.config.direction == 'noising' else [e.corrupted() for e in examples]
        baseline_cache = clean_cache if self.config.direction == 'noising' else corrupt_cache
        return recipients, baseline_cache, clean_scores, corrupt_scores, replacements


def behavior(example, dataset, clean, corrupt, patched, config):
    gap = None if corrupt is None else clean-corrupt
    valid = gap is not None and abs(gap) > config.recovery_epsilon
    return {'metric': example.scoring.metric, 'units': dataset['units'], 'clean_score': clean,
            'corrupt_score': corrupt, 'patched_score': patched, 'signed_change': patched-clean,
            'recovery_per_example': (patched-corrupt)/gap if valid else None,
            'recovery_status': 'valid' if valid else 'undefined_small_gap' if gap is not None else 'unpaired',
            'normalization': 'per-example diagnostic; primary recovery uses dataset means',
            'direction': config.direction}

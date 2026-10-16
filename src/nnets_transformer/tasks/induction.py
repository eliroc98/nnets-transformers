"""Seeded random-token repeated-prefix induction, independent of legacy inputs.

Prompt: BOS A[0:L] A[0:L-1]; predict A[L-1]. Tokens within A are distinct,
non-special tokenizer vocabulary tokens. This is a declared development probe,
not a natural-language or Tigges paper-task dataset.
"""
from dataclasses import asdict, replace
import hashlib
import json
import random
from nnets_transformer.records import Example, Scoring

SCORING_VERSION = "induction-repeat-target-logprob-v1"


class Induction:
    def prepare(self, adapter, config):
        tokenizer = adapter.tokenizer
        special = set(tokenizer.all_special_ids)
        pool = sorted(set(tokenizer.get_vocab().values()) - special)
        pool = [token for token in pool if 0 <= token < adapter.vocab_size]
        bos = tokenizer.bos_token_id
        if bos is None or not 0 <= bos < adapter.vocab_size:
            raise ValueError("Induction requires a valid tokenizer BOS token")
        if config.pattern_length < 2 or config.pattern_length > len(pool):
            raise ValueError("Pattern length exceeds the distinct non-special token pool")
        if 2 * config.pattern_length > adapter.context_length:
            raise ValueError("Induction prompt exceeds context length")
        rng = random.Random(config.seed)
        examples = []
        for index in range(config.examples+config.calibration_examples):
            pattern = rng.sample(pool, config.pattern_length)
            tokens = (bos, *pattern, *pattern[:-1])
            example = Example(f"induction-{index:06d}", tokens, (True,) * len(tokens),
                Scoring("target_log_probability", len(tokens)-1, (pattern[-1],)),
                {"first_copy": list(range(1, len(pattern)+1)),
                 "second_copy": list(range(len(pattern)+1, len(tokens))),
                 "target_first_position": len(pattern)})
            example.validate(adapter.vocab_size, adapter.context_length)
            if index >= config.examples:
                example = replace(example, split="calibration")
            examples.append(example)
        manifest = {"dataset_version": "random-token-induction-v1", "task": "induction",
                    "scoring_version": SCORING_VERSION, "units": "nats", "higher_is_better": True,
                    "split": "evaluation", "seed": config.seed, "pattern_length": config.pattern_length,
                    "sample_count": config.examples, "bos_token_id": bos,
                    "tokenizer_revision": adapter.provenance["tokenizer_revision"],
                    "sample_manifest": [asdict(e) for e in examples[:config.examples]],
                    "calibration_manifest": [asdict(e) for e in examples[config.examples:]]}
        manifest["dataset_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        return examples[:config.examples], manifest

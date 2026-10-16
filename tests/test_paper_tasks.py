"""T04 contracts: paper tasks, calibrated means, donors, directions and resume."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers.utils import import_utils
import_utils._torchvision_available = False
from transformer_lens import HookedTransformer, HookedTransformerConfig

from test_fresh_runner import retained_directory, yaml
from nnets_transformer.config import Experiment, Task, Scope, ComponentSelection, Replacement
from nnets_transformer.models.transformer_lens import TransformerLensAdapter
from nnets_transformer.tasks import get_task
from nnets_transformer.interventions.references import References, donors, behavior, convention
from nnets_transformer.interventions.replacements import TensorReplacement
from nnets_transformer.metrics.behavioral import score, score_batch
from nnets_transformer.records import Scoring, Selection
from nnets_transformer.execution.runner import run, serial_records
from nnets_transformer.execution.batching import records
from nnets_transformer.observability import Job

TOKENIZER = Path("models/historical-cache/models--EleutherAI--pythia-14m/snapshots/5b020995bfc7aee2931b0f35bd70cf7ee8b1db62")
SOURCE = Path("artifacts/tigges-2024-source/supplement.zip")
AVAILABLE = (TOKENIZER / "tokenizer.json").exists() and SOURCE.exists()
TASKS = ("ioi", "gendered_pronoun", "greater_than", "sva")
_ADAPTER = None


def adapter():
    """Tiny random model with the real Pythia tokenizer; shared because tokenizer loading dominates."""
    global _ADAPTER
    if _ADAPTER is None:
        from transformers import AutoTokenizer
        torch.set_num_threads(2)
        model = HookedTransformer(HookedTransformerConfig(n_layers=1, n_heads=2, d_model=16, d_head=8, d_mlp=32,
            n_ctx=64, d_vocab=50304, act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary",
            rotary_dim=8, parallel_attn_mlp=True, seed=5, device="cpu"))
        # Use TransformerLens's own tokenizer processing (it prepends BOS by default).
        model.set_tokenizer(AutoTokenizer.from_pretrained(TOKENIZER))
        assert model.tokenizer("x").input_ids[0] == model.tokenizer.bos_token_id
        _ADAPTER = TransformerLensAdapter(model, {"model_revision": "fixture", "tokenizer_revision": "fixture",
                                                  "inventory_version": "tl-existing-sites-v2"})
    _ADAPTER.forward_count = 0
    return _ADAPTER


def task(name, examples=6, calibration=4, seed=42):
    return Task(name=name, scoring_version=f"tigges-2024-{name}-v1", examples=examples,
                calibration_examples=calibration, seed=seed, source_directory=str(SOURCE.parent))


def decode(token):
    return adapter().tokenizer.decode([token])


class Job0:
    """Minimal job sink for direct reference construction."""
    def stage(self, *a, **k): pass
    def progress(self, *a, **k): pass
    def event(self, *a, **k): pass


@unittest.skipUnless(AVAILABLE, "requires the cached Pythia tokenizer and checksummed Tigges supplement")
class PaperTaskContracts(unittest.TestCase):
    def test_preparation_is_deterministic_aligned_and_split_disjoint(self):
        model = adapter()
        for name in TASKS:
            with self.subTest(task=name):
                examples, manifest = get_task(name).prepare(model, task(name))
                again, same = get_task(name).prepare(model, task(name))
                self.assertEqual(examples, again)
                self.assertEqual(manifest, same)
                self.assertEqual(len(examples), 6)
                self.assertEqual(manifest["subset"], "development")
                calibration = manifest["calibration_manifest"]
                self.assertEqual(len(calibration), 4)
                evaluation = {e.tokens for e in examples} | {e.corrupted_tokens for e in examples}
                self.assertFalse({tuple(c["tokens"]) for c in calibration} & evaluation)
                for e in examples:
                    self.assertEqual(len(e.tokens), len(e.corrupted_tokens))
                    self.assertNotEqual(e.tokens, e.corrupted_tokens)
                    # Strict next-token prediction at the final prompt token.
                    self.assertEqual(e.scoring.position, len(e.tokens)-1)
                    self.assertFalse(set(e.scoring.targets) & set(e.scoring.contrasts))
                changed, _ = get_task(name).prepare(model, task(name, seed=7))
                self.assertNotEqual([e.tokens for e in changed], [e.tokens for e in examples])

    def test_task_specific_candidates_and_boundaries(self):
        model = adapter()
        ioi, _ = get_task("ioi").prepare(model, task("ioi"))
        for e in ioi:
            clean = model.tokenizer.decode(e.tokens)
            target, contrast = decode(e.scoring.targets[0]), decode(e.scoring.contrasts[0])
            self.assertTrue(target.startswith(" ") and contrast.startswith(" "))
            # Clean: subject repeated; corrupt flips S2 to the IO name (ABB->ABA).
            self.assertEqual(clean.count(contrast.strip()), 2)
            self.assertEqual(model.tokenizer.decode(e.corrupted_tokens).count(target.strip()), 2)
        pronouns, _ = get_task("gendered_pronoun").prepare(model, task("gendered_pronoun"))
        self.assertEqual({decode(e.scoring.targets[0]) for e in pronouns} | {decode(e.scoring.contrasts[0]) for e in pronouns}, {" he", " she"})
        for e in pronouns:
            self.assertEqual(decode(e.scoring.targets[0]), e.roles["target_string"])
        years, manifest = get_task("greater_than").prepare(model, task("greater_than"))
        self.assertEqual(manifest["candidate_policy"], "standalone-two-digit-vocabulary-token-after-century")
        for e in years:
            boundary = e.roles["boundary"]
            self.assertEqual(e.roles["clean_year"] % 100, boundary)
            # Equality is invalid: the boundary suffix belongs to the contrast group.
            self.assertEqual([decode(t) for t in e.scoring.contrasts], [f"{s:02d}" for s in range(boundary+1)])
            self.assertEqual([decode(t) for t in e.scoring.targets], [f"{s:02d}" for s in range(boundary+1, 100)])
            century = e.roles["clean_year"]//100
            self.assertTrue(model.tokenizer.decode(e.corrupted_tokens).endswith(f"{century}01 to the year {century}"))
        sva, manifest = get_task("sva").prepare(model, task("sva", examples=20))
        numbers = {e.roles["number"] for e in sva}
        self.assertEqual(numbers, {"singular", "plural"})
        groups = {e.roles["number"]: set(e.scoring.targets) for e in sva}
        self.assertFalse(groups["singular"] & groups["plural"])
        self.assertIn(model.tokenizer.encode(" is", add_special_tokens=False)[0], groups["singular"])
        self.assertIn(model.tokenizer.encode(" are", add_special_tokens=False)[0], groups["plural"])
        self.assertEqual(len({e.roles["pair_id"] for e in sva}), len(sva))

    def test_group_sum_signs_equality_boundary_and_label_anchoring(self):
        vocabulary = 8
        logits = torch.zeros((1, 1, vocabulary))
        logits[0, 0, 3] = 4.  # Mass at the boundary token counts against the score.
        probability = logits[0, 0].softmax(-1)
        spec = Scoring("probability_group_difference", 0, (4, 5), (0, 1, 2, 3))
        expected = float(probability[[4, 5]].sum()-probability[[0, 1, 2, 3]].sum())
        self.assertAlmostEqual(float(score(logits, spec)), expected, places=6)
        self.assertLess(expected, 0)
        # Full-vocabulary denominator: non-candidate tokens 6, 7 dilute both groups.
        self.assertLess(float(probability[[0, 1, 2, 3, 4, 5]].sum()), 1)
        batch = torch.cat([logits, logits.flip(-1)])
        torch.testing.assert_close(score_batch(batch, [spec, spec]), torch.stack([score(batch[:1], spec), score(batch[1:], spec)]))
        model = adapter()
        examples, dataset = get_task("greater_than").prepare(model, task("greater_than"))
        refs = References(model, examples, dataset, [Selection("hook_embed")], Replacement(name="zero"), Job0())
        _, _, clean, corrupt, _ = refs.batch(examples[:2], [Selection("hook_embed")], len(examples[0].tokens))
        corrupt_logits, _ = model.forward(examples[0].corrupted(), [])
        # Corrupt baselines are scored with the clean example's boundary.
        self.assertAlmostEqual(corrupt[0], float(score(corrupt_logits, examples[0].scoring)), places=5)

    def test_replacement_tensors_identity_and_directions(self):
        model = adapter()
        examples, dataset = get_task("ioi").prepare(model, task("ioi"))
        example = examples[0]
        embed = Selection("hook_embed")
        heads = Selection("blocks.0.attn.hook_z", 1)
        observed = [embed, Selection("blocks.0.attn.hook_z")]
        clean_logits, clean = model.forward(example, observed)
        corrupt_logits, corrupt = model.forward(example.corrupted(), observed)
        # No-op identity: patching a clean activation with itself leaves logits unchanged.
        same, _ = model.forward(example, observed, (embed, TensorReplacement(clean)))
        torch.testing.assert_close(same, clean_logits, atol=0, rtol=0)
        clean_score, corrupt_score = float(score(clean_logits, example.scoring)), float(score(corrupt_logits, example.scoring))
        for direction, expected in (("noising", corrupt_score), ("denoising", clean_score)):
            config = Replacement(name="donor", donor_policy="matched", direction=direction)
            refs = References(model, [example], dataset, [embed], config, Job0())
            recipients, baseline, clean_s, corrupt_s, values = refs.batch([example], observed, len(example.tokens))
            logits, _ = model.forward(recipients[0], observed, (embed, values[0]))
            # The only input-dependent site is the embedding (rotary positions): a full patch transfers behavior.
            self.assertAlmostEqual(float(score(logits, example.scoring)), expected, places=5)
            record = behavior(example, dataset, clean_s[0], corrupt_s[0], float(score(logits, example.scoring)), config)
            self.assertAlmostEqual(record["recovery_per_example"], 0. if direction == "noising" else 1., places=4)
        config = Replacement(name="donor", donor_policy="matched")
        refs = References(model, [example], dataset, [heads], config, Job0())
        recipients, _, _, _, values = refs.batch([example], observed, len(example.tokens))
        _, patched = model.forward(recipients[0], observed, (heads, values[0]))
        torch.testing.assert_close(patched[heads.hook][:, :, 1], corrupt[heads.hook][:, :, 1])
        torch.testing.assert_close(patched[heads.hook][:, :, 0], clean[heads.hook][:, :, 0])
        self.assertTrue(all(not h.fwd_hooks for h in model.model.hook_dict.values()))

    def test_calibrated_means_are_exact_and_cover_positions(self):
        model = adapter()
        examples, dataset = get_task("greater_than").prepare(model, task("greater_than"))
        hook = Selection("blocks.0.attn.hook_z")
        from nnets_transformer.interventions.references import restore
        calibration = [restore(json.loads(json.dumps(record))) for record in dataset["calibration_manifest"]]
        caches = [model.forward(e, [hook])[1][hook.hook][0] for e in calibration]
        length = min(len(e.tokens) for e in calibration)
        for reduction in ("position", "global"):
            refs = References(model, examples, dataset, [hook], Replacement(name="mean", mean_reduction=reduction), Job0())
            if reduction == "global":
                expected = torch.cat([c.reshape(-1, *c.shape[1:]) for c in caches]).mean(0)
                torch.testing.assert_close(refs.means[hook.hook], expected, atol=1e-6, rtol=1e-5)
            else:
                expected = torch.stack([c[:length] for c in caches]).mean(0)
                torch.testing.assert_close(refs.means[hook.hook][:length], expected, atol=1e-6, rtol=1e-5)
            _, _, _, _, values = refs.batch(examples[:1], [hook], len(examples[0].tokens))
            head = Selection(hook.hook, 0)
            _, patched = model.forward(examples[0], [hook], (head, values[0]))
            mean = refs.means[hook.hook]
            target = mean[:len(examples[0].tokens), :, :] if reduction == "position" else mean.expand(len(examples[0].tokens), *mean.shape)
            torch.testing.assert_close(patched[hook.hook][0, :, 0], target[:, 0].to(patched[hook.hook].dtype))
        # Evaluation prompts longer than every calibration prompt cannot be position-aligned.
        short = dict(dataset, calibration_manifest=[dict(r, tokens=r["tokens"][:3], attention_mask=r["attention_mask"][:3],
            scoring=dict(r["scoring"], position=2)) for r in dataset["calibration_manifest"]])
        with self.assertRaisesRegex(ValueError, "Position-aligned mean covers"):
            References(model, examples, short, [hook], Replacement(name="mean"), Job0())
        with self.assertRaisesRegex(ValueError, "overlap"):
            leaked = dict(dataset, calibration_manifest=[dict(r) for r in dataset["calibration_manifest"]])
            leaked["calibration_manifest"][0]["tokens"] = list(examples[0].corrupted_tokens)
            References(model, examples, leaked, [hook], Replacement(name="mean"), Job0())

    def test_resampled_donors_are_seeded_nonself_and_aligned(self):
        model = adapter()
        examples, _ = get_task("sva").prepare(model, task("sva", examples=24))
        config = Replacement(name="donor", donor_policy="resample", donor_seed=3)
        plan = donors(examples, config)
        self.assertEqual(plan, donors(examples, config))
        self.assertNotEqual(plan, donors(examples, replace(config, donor_seed=4)))
        by_id = {e.id: e for e in examples}
        for recipient, record in plan.items():
            self.assertNotEqual(record["example_id"], recipient)
            self.assertEqual(len(by_id[record["example_id"]].tokens), len(by_id[recipient].tokens))
        self.assertEqual(convention(config)["donor_seed"], 3)
        lonely = [examples[0], replace(examples[1], tokens=examples[1].tokens[:-1], attention_mask=examples[1].attention_mask[:-1])]
        with self.assertRaisesRegex(ValueError, "No nonself"):
            donors(lonely, config)

    def test_unstable_recovery_is_undefined_not_divided(self):
        example = get_task("ioi").prepare(adapter(), task("ioi"))[0][0]
        record = behavior(example, {"units": "logits"}, 1.0, 1.0+1e-9, 0.5, Replacement(name="zero"))
        self.assertIsNone(record["recovery_per_example"])
        self.assertEqual(record["recovery_status"], "undefined_small_gap")
        record = behavior(example, {"units": "logits"}, 2.0, -1.0, 0.5, Replacement(name="zero"))
        self.assertAlmostEqual(record["recovery_per_example"], .5)

    def test_serial_and_optimized_agree_for_every_replacement(self):
        model = adapter()
        examples, dataset = get_task("ioi").prepare(model, task("ioi", examples=5))
        self.assertGreater(len({len(e.tokens) for e in examples}), 1, "fixture must exercise right padding")
        sources = model.selections("head")[:5]
        observations = model.selections("component")
        work = [(f"{e}-{i}", example, s) for e, example in enumerate(examples) for i, s in enumerate(sources)]
        execution = Experiment().execution
        for config in (Replacement(name="zero"), Replacement(name="mean"), Replacement(name="mean", mean_reduction="global"),
                       Replacement(name="donor"), Replacement(name="donor", direction="denoising")):
            with self.subTest(replacement=convention(config)):
                refs = References(model, examples, dict(dataset, donor_manifest=donors(examples, config)), sources, config, Job0())
                serial = {k: r for k, _, r in serial_records(work, {w[0] for w in work}, model, observations, None, dataset, references=refs)}
                optimized = {k: r for k, _, r in records(work, {w[0] for w in work}, model, observations, None, dataset, execution, references=refs)}
                for key, expected in serial.items():
                    actual = optimized[key]
                    for name in ("clean_score", "corrupt_score", "patched_score"):
                        self.assertAlmostEqual(actual["behavior"][name], expected["behavior"][name], places=4)
                    for got, want in zip(actual["activations"], expected["activations"]):
                        self.assertAlmostEqual(got["cosine_similarity"], want["cosine_similarity"], places=4)
        # Resampled donors are not defined for unequal-length IOI prompts in this tiny subset, so use SVA.
        sva, sva_dataset = get_task("sva").prepare(model, task("sva", examples=16))
        config = Replacement(name="donor", donor_policy="resample")
        refs = References(model, sva, dict(sva_dataset, donor_manifest=donors(sva, config)), sources, config, Job0())
        work = [(f"{e}-{i}", example, s) for e, example in enumerate(sva) for i, s in enumerate(sources[:2])]
        serial = {k: r for k, _, r in serial_records(work, {w[0] for w in work}, model, observations, None, sva_dataset, references=refs)}
        optimized = {k: r for k, _, r in records(work, {w[0] for w in work}, model, observations, None, sva_dataset, execution, references=refs)}
        for key, expected in serial.items():
            self.assertAlmostEqual(optimized[key]["behavior"]["patched_score"], expected["behavior"]["patched_score"], places=5)

    def test_runner_resume_equivalence_with_mean_and_donor(self):
        from nnets_transformer.execution.artifacts import Artifacts
        model = adapter()
        for name, replacement in (("gendered_pronoun", Replacement(name="mean")),
                                  ("greater_than", Replacement(name="donor", direction="denoising"))):
            with self.subTest(task=name), retained_directory() as tmp:
                root = Path(tmp)
                def configure(path):
                    cfg = Experiment(command="run", task=task(name, examples=3, calibration=3), replacement=replacement,
                                     scope=Scope(resolution="component", components=[ComponentSelection("hook_embed"),
                                     ComponentSelection("blocks.0.attn.hook_z", 1)]), output_directory=str(path))
                    cfg.execution.backend = "optimized"
                    cfg.execution.examples_per_shard = 2
                    cfg.execution.example_batch_size = 2
                    return cfg
                cfg = configure(root / "results")
                commit = Artifacts.commit
                def stop(store, key, record):
                    commit(store, key, record)
                    if len(store.manifest["shards"]) == 1:
                        raise KeyboardInterrupt("simulated interruption")
                with Job("interrupt", root=root / "logs") as job, patch.object(Artifacts, "commit", stop), self.assertRaises(KeyboardInterrupt):
                    run(cfg, yaml(cfg), job, adapter=model)
                cfg.execution.resume = True
                with Job("resume", root=root / "logs") as job:
                    run(cfg, yaml(cfg), job, adapter=model)
                fresh = configure(root / "independent")
                with Job("independent", root=root / "logs") as job:
                    run(fresh, yaml(fresh), job, adapter=model)
                for artifact in ("similarities.csv", "dataset.json"):
                    self.assertEqual((Path(cfg.output_directory) / artifact).read_bytes(), (Path(fresh.output_directory) / artifact).read_bytes())
                resumed = json.loads((Path(cfg.output_directory) / "behavior.json").read_text())
                independent = json.loads((Path(fresh.output_directory) / "behavior.json").read_text())
                self.assertEqual(resumed, independent)
                self.assertEqual(set(resumed["summaries"]), {"hook_embed", "blocks.0.attn.hook_z.1"})
                dataset = json.loads((Path(cfg.output_directory) / "dataset.json").read_text())
                self.assertEqual(dataset["replacement_convention"], convention(replacement))
                if name == "greater_than":
                    # Denoising the full embedding restores clean behavior on every example.
                    self.assertAlmostEqual(resumed["summaries"]["hook_embed"]["recovery"], 1., places=4)


if __name__ == "__main__":
    unittest.main()

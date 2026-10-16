import builtins
from dataclasses import replace
import json
from pathlib import Path
from contextlib import contextmanager
import uuid
import io
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from transformers.utils import import_utils
import_utils._torchvision_available = False
from transformer_lens import HookedTransformer, HookedTransformerConfig
from omegaconf import OmegaConf
from nnets_transformer.config import Experiment, Task, Scope, ComponentSelection, load
from nnets_transformer.models.transformer_lens import TransformerLensAdapter
from nnets_transformer.tasks.induction import Induction
from nnets_transformer.interventions.replacements import Identity, Zero
from nnets_transformer.metrics.behavioral import score
from nnets_transformer.metrics.activation import compare
from nnets_transformer.records import Scoring
from nnets_transformer.execution.runner import run
from nnets_transformer.observability import Job


@contextmanager
def retained_directory():
    root = Path("artifacts/fresh-validation") / uuid.uuid4().hex
    root.mkdir(parents=True)
    print(f"Retained fresh-validation artifacts: {root.resolve()}")
    yield str(root)


class Tokenizer:
    all_special_ids = [0]
    bos_token_id = 0
    padding_side = "right"
    def get_vocab(self):
        return {str(i): i for i in range(32)}


def tiny():
    torch.set_num_threads(2)
    model = HookedTransformer(HookedTransformerConfig(n_layers=1, n_heads=2, d_model=16,
        d_head=8, d_mlp=32, n_ctx=32, d_vocab=32, act_fn="gelu", normalization_type="LN",
        positional_embedding_type="rotary", rotary_dim=8, parallel_attn_mlp=True, seed=11, device="cpu"))
    model.tokenizer = Tokenizer()
    return TransformerLensAdapter(model, {"model_revision": "fixture", "tokenizer_revision": "fixture",
                                         "inventory_version": "tl-existing-sites-v2"})


def config(root):
    cfg = Experiment()
    cfg.command = "run"
    cfg.task = Task(name="induction", scoring_version="induction-repeat-target-logprob-v1", examples=2, pattern_length=4)
    cfg.scope = Scope(resolution="component", components=[ComponentSelection("hook_embed"),
                      ComponentSelection("blocks.0.attn.hook_q", 0)])
    cfg.output_directory = str(root / "results")
    return cfg


def yaml(cfg):
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))


class FreshContracts(unittest.TestCase):
    def test_task_reproducibility_positions_and_candidate_anchoring(self):
        adapter = tiny()
        cfg = config(Path("unused"))
        examples, manifest = Induction().prepare(adapter, cfg.task)
        again, same = Induction().prepare(adapter, cfg.task)
        self.assertEqual(examples, again)
        self.assertEqual(manifest, same)
        example = examples[0]
        self.assertEqual(example.tokens[1:4], example.tokens[5:])
        self.assertEqual(example.scoring.targets[0], example.tokens[4])
        self.assertEqual(example.scoring.position, 7)
        self.assertEqual(example.tokens[0], 0)
        self.assertNotIn(0, example.tokens[1:])
        changed, _ = Induction().prepare(adapter, replace(cfg.task, seed=43))
        self.assertNotEqual(changed, examples)
        with self.assertRaises(ValueError):
            Induction().prepare(adapter, replace(cfg.task, pattern_length=20))

    def test_scores_use_full_vocabulary_and_group_sums(self):
        logits = torch.tensor([[[1., 2., 3., 8.]]])
        probability = logits[0, 0].softmax(-1)
        specification = Scoring("probability_group_difference", 0, (0, 1), (2,))
        self.assertAlmostEqual(float(score(logits, specification)), float(probability[0]+probability[1]-probability[2]))
        self.assertAlmostEqual(float(score(logits, Scoring("logit_difference", 0, (1,), (2,)))), -1)
        self.assertAlmostEqual(float(score(logits, Scoring("target_log_probability", 0, (1,)))),
                               float(logits[0, 0].log_softmax(-1)[1]))

    def test_identity_head_zero_all_positions_and_hook_cleanup(self):
        adapter = tiny()
        example = Induction().prepare(adapter, config(Path("unused")).task)[0][0]
        selections = adapter.selections("head")
        selected = next(s for s in selections if s.hook.endswith("hook_q") and s.head == 1)
        logits, clean = adapter.forward(example, selections)
        same, _ = adapter.forward(example, selections, (selected, Identity()))
        self.assertTrue(torch.equal(logits, same))
        _, changed = adapter.forward(example, selections, (selected, Zero()))
        self.assertEqual(torch.count_nonzero(changed[selected.hook][:, :, 1, :]), 0)
        self.assertTrue(torch.equal(clean[selected.hook][:, :, 0, :], changed[selected.hook][:, :, 0, :]))
        self.assertTrue(all(not h.fwd_hooks for h in adapter.model.hook_dict.values()))
        def fail(value, hook):
            raise RuntimeError("deliberate forward failure")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with adapter.model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", fail)]):
                adapter.forward(example, selections, (selected, Zero()))
        self.assertTrue(all(not h.fwd_hooks for h in adapter.model.hook_dict.values()))

    def test_masks_and_zero_norm_status(self):
        adapter = tiny()
        selection = adapter.selections("component")[0]
        component = adapter.components[selection.hook]
        clean = torch.tensor([[[1., 0.], [100., 0.]]])
        patched = torch.tensor([[[1., 0.], [-100., 0.]]])
        result = compare(clean, patched, selection, component, (True, False))
        self.assertEqual(result["cosine_similarity"], 1)
        self.assertEqual(result["relative_l2_change"], 0)
        result = compare(clean*0, clean*0, selection, component, (True, True))
        self.assertIsNone(result["relative_l2_change"])
        self.assertEqual(result["relative_l2_status"], "undefined_clean_zero_norm")

    def test_fresh_run_forbids_old_artifact_access_and_resumes_exactly(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            cfg = config(root)
            adapter = tiny()
            original = builtins.open
            def guard(path, *args, **kwargs):
                text = str(path)
                if any(name in text for name in ("ablation_data", "induction_data.json", "loss_data.json", "graph_analysis.ipynb")) or text == "main.py":
                    raise AssertionError(f"Attempted legacy access: {text}")
                return original(path, *args, **kwargs)
            with patch("builtins.open", guard), patch("io.open", guard), Job("fresh", root=root / "logs") as job:
                run(cfg, yaml(cfg), job, adapter=adapter)
            output = Path(cfg.output_directory)
            first = (output / "similarities.csv").read_bytes()
            behavior = (output / "behavior.json").read_bytes()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["state"], "completed")
            self.assertEqual(manifest["schema_version"], "fresh-reference-v2")
            self.assertEqual(manifest["work_count"], 4)
            self.assertEqual(len(manifest["shards"]), 1)
            cfg.execution.resume = True
            with Job("resume", root=root / "logs") as job, patch.object(adapter, "forward", side_effect=AssertionError("completed work must be skipped")):
                run(cfg, yaml(cfg), job, adapter=adapter)
            status = json.loads((job.directory / "status.json").read_text())
            self.assertEqual(status["counters"]["examples"], {"completed": 0, "total": 2, "skipped": 2, "failed": 0})
            self.assertEqual((output / "similarities.csv").read_bytes(), first)
            self.assertEqual((output / "behavior.json").read_bytes(), behavior)
            cfg.task.seed += 1
            with Job("incompatible", root=root / "logs") as job, self.assertRaisesRegex(ValueError, "identity differs"):
                run(cfg, yaml(cfg), job, adapter=adapter)

    def test_interrupted_prefix_resume_and_corruption_rejected(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            cfg = config(root)
            cfg.execution.examples_per_shard = 1
            adapter = tiny()
            from nnets_transformer.execution.artifacts import Artifacts
            commit = Artifacts.commit
            def stop(store, key, record):
                commit(store, key, record)
                raise KeyboardInterrupt("simulated interruption after commit")
            with Job("interrupt", root=root / "logs") as job, patch.object(Artifacts, "commit", stop), self.assertRaises(KeyboardInterrupt):
                run(cfg, yaml(cfg), job, adapter=adapter)
            manifest = json.loads((Path(cfg.output_directory) / "manifest.json").read_text())
            self.assertEqual(manifest["state"], "interrupted")
            self.assertEqual(len(manifest["shards"]), 1)
            cfg.execution.resume = True
            with Job("resume", root=root / "logs") as job:
                run(cfg, yaml(cfg), job, adapter=adapter)
            resumed = (Path(cfg.output_directory) / "similarities.csv").read_bytes()
            fresh = config(root / "independent")
            fresh.execution.examples_per_shard = 1
            with Job("independent", root=root / "logs") as job:
                run(fresh, yaml(fresh), job, adapter=adapter)
            self.assertEqual(resumed, (Path(fresh.output_directory) / "similarities.csv").read_bytes())
            shard = next((Path(cfg.output_directory) / "shards").glob("*.zip"))
            shard.write_bytes(b"corrupted")
            with Job("corrupt", root=root / "logs") as job, self.assertRaisesRegex(ValueError, "checksum"):
                run(cfg, yaml(cfg), job, adapter=adapter)

    def test_fresh_analysis_is_cpu_only_and_preserves_partial_coverage(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            cfg = config(root)
            with Job("fresh", root=root / "logs") as job:
                run(cfg, yaml(cfg), job, adapter=tiny())
            code = """
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'transformer_lens'}:
        raise AssertionError('Analysis loaded a model library: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from nnets_transformer.cli import main
main()
"""
            command = [sys.executable, "-u", "-c", code, "--job-id", "fresh-analysis", "--log-root", str(root / "logs"),
                       "+experiment=analysis", f"input_matrix={Path(cfg.output_directory) / 'similarities.csv'}",
                       f"output_directory={root / 'analysis'}"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            manifest = json.loads((root / "analysis" / "manifest.json").read_text())
            self.assertEqual(manifest["scoring_version"], "induction-repeat-target-logprob-v1")
            self.assertEqual(manifest["graph"]["measurement_status"], "partial")
            self.assertLess(manifest["graph"]["measurement_coverage"], 1)
            self.assertEqual(manifest["inventory_provenance"], "tl-existing-sites-v2")
            (Path(cfg.output_directory) / "similarities.csv").write_text("tampered")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum failed", result.stdout+result.stderr)

    def test_composed_fresh_preset_and_yaml_logging(self):
        cfg, _ = load("configs", overrides=["+experiment=induction_smoke", "logging.progress_interval_seconds=0.1"])
        self.assertEqual(cfg.command, "run")
        self.assertEqual(cfg.task.name, "induction")
        self.assertEqual(cfg.scope.components[1].head, 0)
        self.assertEqual(cfg.logging.progress_interval_seconds, .1)


if __name__ == "__main__":
    unittest.main()

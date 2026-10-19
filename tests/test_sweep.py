"""T05 sweep contracts: one load per checkpoint, recorded skips, cell resume."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_fresh_runner import tiny, retained_directory
from nnets_transformer.config import load
from nnets_transformer.execution.sweep import run_sweep
from nnets_transformer.observability import Job

REVISIONS = {"step0": "0"*40, "step8": "8"*40}


class SweepContracts(unittest.TestCase):
    def configure(self, root, *extra):
        config, _ = load("configs", overrides=["command=sweep", "scope.resolution=component",
            "scope.components=[{hook: hook_embed}, {hook: blocks.0.attn.hook_q, head: 0}]",
            "sweep.checkpoints=[step0,step8]", "sweep.tasks=[induction]",
            "sweep.replacements=[zero,mean,donor,donor_resample]", "sweep.examples=3", "sweep.calibration_examples=2",
            "task.pattern_length=4", f"sweep.output_root={root}/sweep", "execution.backend=optimized",
            "execution.example_batch_size=1", "execution.examples_per_shard=1", "sweep.compact=cosine16", *extra])
        return config

    def test_checkpoint_grouping_skips_and_resume(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            loads = []
            def load_adapter(model, resources, job):
                loads.append(model.revision)
                return tiny()
            config = self.configure(root)
            # Induction cells inherit pattern_length from task YAML; force the tiny vocabulary size.
            from nnets_transformer.execution import sweep as module
            original = module._load
            def small(schema, path, overrides=None):
                value = original(schema, path, overrides)
                if hasattr(value, "pattern_length"):
                    value.pattern_length = 4
                return value
            with patch("nnets_transformer.models.checkpoints.resolve_revisions", lambda m, c: {k: REVISIONS[k] for k in c}), \
                 patch("nnets_transformer.models.load_adapter", load_adapter), patch.object(module, "_load", small), \
                 Job("sweep", root=root / "logs") as job:
                run_sweep(config, "configs", job)
            self.assertEqual(loads, [REVISIONS["step0"], REVISIONS["step8"]])
            manifest = json.loads((root / "sweep" / "pythia-14m" / "sweep.json").read_text())
            self.assertEqual(len(manifest["cells"]), 6)
            self.assertTrue(all(c["state"] == "completed" for c in manifest["cells"]))
            self.assertEqual([(s["replacement"], s["checkpoint"]) for s in manifest["skipped"]], [("donor", "all")])
            self.assertEqual(set(manifest["fixed_evaluation_data"]), {"induction/zero", "induction/mean", "induction/donor_resample"})
            cell = root / "sweep" / "pythia-14m" / "step8" / "induction" / "mean"
            self.assertEqual(json.loads((cell / "manifest.json").read_text())["compaction"]["mode"], "cosine16")
            self.assertFalse(list((cell / "shards").glob("*.zip")))
            saved = json.loads((cell / "manifest.json").read_text())
            self.assertEqual(saved["identity"]["configuration"]["model"]["revision"], REVISIONS["step8"])
            self.assertNotIn("sweep", saved["identity"]["configuration"])
            # Rerunning without resume is refused; resuming loads nothing.
            loads.clear()
            with patch("nnets_transformer.models.load_adapter", load_adapter), Job("again", root=root / "logs") as job, \
                 self.assertRaises(FileExistsError):
                run_sweep(config, "configs", job)
            config.execution.resume = True
            with patch("nnets_transformer.models.load_adapter", load_adapter), patch.object(module, "_load", small), \
                 Job("resume", root=root / "logs") as job:
                run_sweep(config, "configs", job)
            self.assertEqual(loads, [])
            self.assertEqual(len(json.loads((root / "sweep" / "pythia-14m" / "sweep.json").read_text())["attempts"]), 2)

    def test_failed_cell_is_recorded_and_others_continue(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            config = self.configure(root, "sweep.replacements=[zero,mean]")
            from nnets_transformer.execution import sweep as module
            original, runner = module._load, None
            def small(schema, path, overrides=None):
                value = original(schema, path, overrides)
                if hasattr(value, "pattern_length"):
                    value.pattern_length = 4
                return value
            import nnets_transformer.execution.runner as runner_module
            real_run = runner_module.run
            def flaky(cell, yaml, job, adapter=None):
                if cell.replacement.name == "mean" and cell.model.checkpoint == "step0":
                    raise ValueError("deliberate cell failure")
                return real_run(cell, yaml, job, adapter=adapter)
            with patch("nnets_transformer.models.checkpoints.resolve_revisions", lambda m, c: {k: REVISIONS[k] for k in c}), \
                 patch("nnets_transformer.models.load_adapter", lambda *a: tiny()), patch.object(module, "_load", small), \
                 patch.object(runner_module, "run", flaky), Job("flaky", root=root / "logs") as job, \
                 self.assertRaisesRegex(RuntimeError, "1 sweep cell"):
                run_sweep(config, "configs", job)
            states = {(c["checkpoint"], c["replacement"]): c["state"]
                      for c in json.loads((root / "sweep" / "pythia-14m" / "sweep.json").read_text())["cells"]}
            self.assertEqual(states, {("step0", "zero"): "completed", ("step0", "mean"): "failed",
                                      ("step8", "zero"): "completed", ("step8", "mean"): "completed"})
            # A rerun retries only the failed cell.
            config.execution.resume = True
            with patch("nnets_transformer.models.load_adapter", lambda *a: tiny()), patch.object(module, "_load", small), \
                 Job("retry", root=root / "logs") as job:
                run_sweep(config, "configs", job)
            states = {c["state"] for c in json.loads((root / "sweep" / "pythia-14m" / "sweep.json").read_text())["cells"]}
            self.assertEqual(states, {"completed"})

    def test_incompatible_cells_can_be_rejected_early(self):
        with retained_directory() as tmp:
            config = self.configure(Path(tmp), "sweep.skip_incompatible=false")
            with patch("nnets_transformer.models.checkpoints.resolve_revisions", lambda m, c: {k: REVISIONS[k] for k in c}), \
                 patch("nnets_transformer.models.load_adapter", side_effect=AssertionError("no loading before validation")), \
                 Job("reject", root=Path(tmp) / "logs") as job, self.assertRaisesRegex(ValueError, "Incompatible sweep cell"):
                run_sweep(config, "configs", job)

    def test_pilot_preset_composes(self):
        config, _ = load("configs", overrides=["+experiment=pilot", "model=pythia_160m"])
        self.assertEqual(config.command, "sweep")
        self.assertEqual(config.model.id, "EleutherAI/pythia-160m")
        self.assertEqual(config.model.n_layers, 12)
        self.assertEqual(len(config.sweep.checkpoints), 12)
        self.assertEqual(config.sweep.tasks, ["induction", "ioi", "gendered_pronoun", "greater_than", "sva"])
        with self.assertRaisesRegex(ValueError, "multiple of example_batch_size"):
            load("configs", overrides=["+experiment=pilot", "execution.examples_per_shard=6"])


if __name__ == "__main__":
    unittest.main()

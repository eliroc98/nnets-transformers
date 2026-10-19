"""Post-completion compaction keeps what analysis needs and makes runs final."""
import json
from pathlib import Path
import shutil
import unittest

import numpy as np
import pandas as pd

from test_fresh_runner import tiny, config, yaml, retained_directory
from nnets_transformer.execution.runner import run
from nnets_transformer.execution.artifacts import compact, read_chunk, iter_records, load_dataset
from nnets_transformer.analysis.artifacts import read_fresh
from nnets_transformer.analysis.export import normalize
from nnets_transformer.config import Graph, Replacement
from nnets_transformer.observability import Job


class CompactionContracts(unittest.TestCase):
    def make(self, root):
        cfg = config(root / "run")
        cfg.scope.components = []
        cfg.replacement = Replacement(name="mean")
        cfg.task.calibration_examples = 2
        cfg.execution.examples_per_shard = 1
        with Job("compaction-fixture", root=root / "logs") as job:
            run(cfg, yaml(cfg), job, adapter=tiny())
        return Path(cfg.output_directory)

    def test_cosine16_preserves_analysis_inputs_and_is_final(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            original = self.make(root)
            copy = root / "compacted"
            shutil.copytree(original, copy)
            manifest = json.loads((original / "manifest.json").read_text())
            cosines = np.concatenate([read_chunk(original / s["path"])[1][..., 0] for s in manifest["shards"].values()])
            records = list(iter_records(original, manifest))
            behavior = json.loads((original / "behavior.json").read_text())
            with Job("compact", root=root / "logs") as job:
                result = compact(copy, "cosine16", job)
                self.assertEqual(compact(copy, "cosine16", job), result)  # idempotent
            self.assertFalse(list((copy / "shards").glob("*.zip")))
            self.assertFalse((copy / "dataset.json").exists())
            self.assertEqual((copy / "similarities.csv").read_bytes(), (original / "similarities.csv").read_bytes())
            self.assertEqual(load_dataset(copy), load_dataset(original))
            self.assertIsNotNone(read_fresh(copy / "similarities.csv"))  # all remaining checksums verify
            saved = json.loads((copy / "behavior.json").read_text())
            self.assertEqual(saved["summaries"], behavior["summaries"])
            rows = pd.read_parquet(copy / "behavior_measurements.parquet")
            expected = pd.DataFrame(behavior["measurements"])
            pd.testing.assert_frame_equal(rows, expected)
            import zipfile, io
            with zipfile.ZipFile(copy / "per_example_cosine.zip") as bundle:
                stored = np.load(io.BytesIO(bundle.read("cosine.npy")))
                index = json.loads(bundle.read("records.json"))["rows"]
            self.assertEqual(stored.dtype, np.float16)
            np.testing.assert_allclose(stored.astype(np.float32), cosines, atol=1e-3)
            self.assertEqual([r[0] for r in index], [r["work_key"] for r in records])
            self.assertLessEqual(result["per_example"]["max_abs_rounding_error"], 1e-3)
            # Exports agree with the uncompacted run on every table the dashboard reads.
            with Job("export", root=root / "logs") as job:
                normalize([original], root / "export-a", Graph(resolution="component"), job, [.5, .7], include_activations=False)
                normalize([copy], root / "export-b", Graph(resolution="component"), job, [.5, .7], include_activations=False)
            for table in ("checkpoints", "components", "edges", "behavior", "roles"):
                a = pd.read_parquet(root / "export-a" / f"{table}.parquet").drop(columns=["run_id", "trajectory_id"], errors="ignore")
                b = pd.read_parquet(root / "export-b" / f"{table}.parquet").drop(columns=["run_id", "trajectory_id"], errors="ignore")
                if table == "behavior":
                    a, b = a[sorted(a.columns)], b[sorted(b.columns)]
                pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
            # Final: no resume, no per-record activation reads.
            cfg = config(root / "run")
            cfg.scope.components = []
            cfg.replacement = Replacement(name="mean")
            cfg.task.calibration_examples = 2
            cfg.execution.examples_per_shard = 1
            cfg.output_directory, cfg.execution.resume = str(copy), True
            with Job("resume-compacted", root=root / "logs") as job, self.assertRaisesRegex(ValueError, "compacted after completion"):
                run(cfg, yaml(cfg), job, adapter=tiny())
            with self.assertRaisesRegex(ValueError, "Compacted run"):
                list(iter_records(copy, json.loads((copy / "manifest.json").read_text())))

    def test_cosine_only_runs_label_uncomputed_fields(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            cfg = config(root / "cosine")
            cfg.scope.components = []
            cfg.execution.backend = "optimized"
            cfg.execution.activation_metrics = "cosine"
            with Job("cosine-run", root=root / "logs") as job:
                run(cfg, yaml(cfg), job, adapter=tiny())
            reference = config(root / "full")
            reference.scope.components = []
            reference.execution.backend = "optimized"
            with Job("full-run", root=root / "logs") as job:
                run(reference, yaml(reference), job, adapter=tiny())
            fast_root, full_root = Path(cfg.output_directory), Path(reference.output_directory)
            fast_manifest = json.loads((fast_root / "manifest.json").read_text())
            self.assertEqual(fast_manifest["shard_format"]["computed_fields"], ["cosine_similarity"])
            fast = list(iter_records(fast_root, fast_manifest))
            full = list(iter_records(full_root, json.loads((full_root / "manifest.json").read_text())))
            for a, b in zip(fast, full):
                for x, y in zip(a["activations"], b["activations"]):
                    self.assertAlmostEqual(x["cosine_similarity"], y["cosine_similarity"], places=6)
                    self.assertEqual(x["relative_l2_status"], "not_computed")
                    self.assertIsNone(x["relative_l2_change"])
            a = pd.read_csv(fast_root / "similarities.csv", index_col=0)
            b = pd.read_csv(full_root / "similarities.csv", index_col=0)
            pd.testing.assert_frame_equal(a, b, atol=1e-6)
            record = compact(fast_root, "cosine16")
            self.assertEqual(record["removed_fields"], [])

    def test_means_mode_and_refusals(self):
        with retained_directory() as tmp:
            root = Path(tmp)
            original = self.make(root)
            copy = root / "means"
            shutil.copytree(original, copy)
            compact(copy, "means")
            self.assertFalse((copy / "per_example_cosine.zip").exists())
            self.assertIsNotNone(read_fresh(copy / "similarities.csv"))
            tampered = root / "tampered"
            shutil.copytree(original, tampered)
            (tampered / "behavior.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                compact(tampered, "cosine16")
            self.assertTrue(list((tampered / "shards").glob("*.zip")))
            with self.assertRaises(ValueError):
                compact(original, "float8")


if __name__ == "__main__":
    unittest.main()

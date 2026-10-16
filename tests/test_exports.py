import json
from pathlib import Path
import unittest
import uuid
import pandas as pd
from nnets_transformer.analysis.longitudinal import upper_tail, jaccard, role_dynamics
from nnets_transformer.analysis.export import normalize
from nnets_transformer.config import Graph
from nnets_transformer.observability import Job


class ExportContracts(unittest.TestCase):
    def test_role_ties_empty_sets_and_checkpoint_order(self):
        self.assertEqual(upper_tail({"a": 1, "b": 1}), set())
        self.assertIsNone(jaccard(set(), set()))
        frame = pd.DataFrame([dict(run_id="x", step=s, threshold=.7, resolution="component",
            graph_version="influence-v2", metric="in_strength", component=c, value=v)
            for s, values in [(2, [3, 1, 0]), (0, [0, 3, 1])] for c, v in zip("abc", values)])
        roles = role_dynamics(frame, .5)
        self.assertEqual(roles.step.tolist(), [0, 2])
        self.assertEqual(roles.iloc[1].churn, 1)

    def test_verified_normalized_exports_and_figures(self):
        source = Path("runs/fresh-induction-final")
        if not source.exists():
            self.skipTest("Optional local fresh integration artifact absent")
        root = Path("artifacts/export-validation") / uuid.uuid4().hex
        with Job("export", root=root / "logs") as job:
            normalize([str(source)], root / "export", Graph(), job, [.3, .7])
        manifest = json.loads((root / "export/manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], "normalized-experiment-v1")
        for name, path in manifest["tables"].items():
            frame = pd.read_parquet(root / "export" / path)
            self.assertIsInstance(frame, pd.DataFrame)
        behavior = pd.read_parquet(root / "export/behavior.parquet")
        self.assertEqual(len(behavior), 6)
        summary = pd.read_parquet(root / "export/checkpoints.parquet")
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(summary.measurement_status == "partial"))
        self.assertTrue((root / "export/report.html").stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from nnets_transformer.config import load

ROOT = Path("artifacts/validation-t02") / uuid.uuid4().hex


class ConfigurationContracts(unittest.TestCase):
    def test_compose_and_resolve_without_chdir(self):
        before = Path.cwd()
        cfg, yaml = load("configs", overrides=["+experiment=historical", "resources=h100", "resources.disable_optional_vision=true"])
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(cfg.command, "reference")
        self.assertEqual(cfg.resources.device, "cuda")
        self.assertTrue(cfg.resources.disable_optional_vision)
        self.assertEqual(len(cfg.model.revision), 40)
        self.assertEqual(yaml.task.scoring_version, "legacy-induction-token2205-v1")

    def test_invalid_combinations_rejected_early(self):
        for override in ["replacement.name=invalid", "replacement.estimator=gradient", "scope.positions=last",
                         "graph.resolution=mixed", "resources.dtype=int8", "model.revision=step1000",
                         "resources.device=cuda:1", "task.name=ioi", "graph.similarity_threshold=2"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load("configs", overrides=[override])
        for override in ["unknown_field=7", "model.n_layers=not-an-integer"]:
            with self.subTest(override=override), self.assertRaises(Exception):
                load("configs", overrides=[override])

    def test_cpu_analysis_forbids_model_imports_and_preserves_input(self):
        import hashlib
        source = Path("ablation_data/step1000.csv")
        if not source.exists():
            self.skipTest("Local historical artifacts are optional")
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        output = ROOT / "analysis"
        code = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformer_lens', 'transformers'}:
        raise AssertionError('CPU artifact analysis attempted model import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from nnets_transformer.cli import main
main()
'''
        result = subprocess.run([sys.executable, "-u", "-c", code, "--job-id", "t02-cpu-guard",
                "--log-root", str(ROOT / "logs"), "+experiment=analysis", f"input_matrix={source}", f"output_directory={output}"],
                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["graph"]["graph_version"], "influence-v2")
        self.assertEqual(manifest["num_nodes"], 44)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)
        self.assertTrue((output / "resolved.yaml").exists())
        # Reusing an output path fails, preserving completed artifacts.
        second = subprocess.run([sys.executable, "-u", "-c", code, "--job-id", "t02-cpu-guard",
                "--log-root", str(ROOT / "logs"), "+experiment=analysis", f"input_matrix={source}", f"output_directory={output}"],
                capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
        print(f"CPU analysis artifacts/logs: {ROOT.resolve()}")


if __name__ == "__main__":
    unittest.main()

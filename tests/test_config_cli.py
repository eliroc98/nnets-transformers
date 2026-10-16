from pathlib import Path
import unittest

from nnets_transformer.config import load


class ConfigurationContracts(unittest.TestCase):
    def test_compose_and_resolve_without_chdir(self):
        before = Path.cwd()
        cfg, yaml = load("configs", overrides=["+experiment=pilot", "model=pythia_160m"])
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(cfg.command, "sweep")
        self.assertEqual(cfg.resources.device, "cuda")
        self.assertTrue(cfg.resources.disable_optional_vision)
        self.assertEqual(cfg.model.cache_dir, "models/hf-cache")
        self.assertEqual(yaml.task.scoring_version, "tigges-2024-ioi-v1")
        for removed in ("+experiment=historical", "command=reference", "graph.version=legacy-notebook-v1"):
            with self.subTest(removed=removed), self.assertRaises(Exception):
                load("configs", overrides=[removed])

    def test_invalid_combinations_rejected_early(self):
        for override in ["replacement.name=invalid", "replacement.estimator=gradient", "scope.positions=last",
                         "graph.resolution=mixed", "resources.dtype=int8", "model.revision=step1000",
                         "resources.device=cuda:1", "task.name=ioi", "graph.similarity_threshold=2"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load("configs", overrides=[override])
        for override in ["unknown_field=7", "model.n_layers=not-an-integer"]:
            with self.subTest(override=override), self.assertRaises(Exception):
                load("configs", overrides=[override])


if __name__ == "__main__":
    unittest.main()

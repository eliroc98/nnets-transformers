import unittest
import torch
# Text-only tests: ignore the environment's unrelated, unlocked vision backend.
from transformers.utils import import_utils
import_utils._torchvision_available = False
from transformer_lens import HookedTransformer, HookedTransformerConfig
from nnets_transformer.execution.reference import serial_intervention, observation_hooks
import main as legacy


class ReferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = HookedTransformer(HookedTransformerConfig(
            n_layers=1, n_heads=2, d_model=16, d_head=8, d_mlp=32, n_ctx=8,
            d_vocab=2210, act_fn="gelu", normalization_type="LN", device="cpu", seed=7,
            positional_embedding_type="rotary", rotary_dim=8, parallel_attn_mlp=True,
        ))
        cls.model.eval()
        cls.tokens = torch.tensor([[1, 4, 3, 4]])

    def test_legacy_equivalence_whole_and_head(self):
        self.assertEqual(observation_hooks(self.model), legacy.get_hook_list(self.model, ln=False))
        for head in (None, 1):
            new, score = serial_intervention(self.model, self.tokens, "blocks.0.attn.hook_q", head)
            old, logit, loss, clean_logit, clean_loss = legacy.get_zero_ablation_scores(
                self.model, self.tokens, "blocks.0.attn.hook_q", observation_hooks(self.model), head)
            self.assertEqual(new, old)
            self.assertEqual(list(score.values()), [logit, loss, clean_logit, clean_loss])

    def test_identity_and_hook_cleanup(self):
        _, score = serial_intervention(self.model, self.tokens, "blocks.0.attn.hook_q", 1, replacement="identity")
        self.assertEqual(score["clean_logit"], score["patched_logit"])
        self.assertEqual(score["clean_loss"], score["patched_loss"])
        self.assertTrue(all(not h.fwd_hooks for h in self.model.hook_dict.values()))
        def fail(value, hook):
            raise RuntimeError("deliberate observation failure")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with self.model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", fail)]):
                serial_intervention(self.model, self.tokens, "blocks.0.attn.hook_q", 1)
        self.assertTrue(all(not h.fwd_hooks for h in self.model.hook_dict.values()))

    def test_invalid_scope(self):
        with self.assertRaises(ValueError):
            serial_intervention(self.model, self.tokens, "blocks.0.mlp.hook_post", 0)


if __name__ == "__main__":
    unittest.main()

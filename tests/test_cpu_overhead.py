"""T05f equivalence against the previous scoring and TransformerLens hook paths."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from test_fresh_runner import tiny, config
from nnets_transformer.records import Scoring
from nnets_transformer.metrics.behavioral import score_batch, _batch_indices, _candidate_indices
from nnets_transformer.models.inputs import padded_inputs
from nnets_transformer.tasks.induction import Induction
from nnets_transformer.tasks.prepared import PreparedTasks
from nnets_transformer.interventions.replacements import Zero


class CPUOverheadContracts(unittest.TestCase):
    def test_cached_group_scoring_matches_previous_dense_reduction(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            logits = torch.randn(5, 3, 211, generator=torch.Generator().manual_seed(9)).to(device)
            specs = [Scoring("probability_group_difference", i % 3,
                             tuple(range(i, 80+i)), tuple(range(100, 103+i))) for i in range(5)]
            selected = logits[torch.arange(5, device=device), torch.tensor([s.position for s in specs], device=device)].float()
            weights = torch.zeros_like(selected)
            for row, spec in enumerate(specs):
                weights[row, list(spec.targets)] = 1
                weights[row, list(spec.contrasts)] = -1
            expected = (selected.softmax(-1)*weights).sum(-1)
            torch.testing.assert_close(score_batch(logits, specs), expected, atol=0, rtol=0)
            hits = _batch_indices.cache_info().hits
            torch.testing.assert_close(score_batch(logits, specs), expected, atol=0, rtol=0)
            self.assertGreater(_batch_indices.cache_info().hits, hits)
            hits = _candidate_indices.cache_info().hits
            score_batch(logits.flip(0), specs[::-1])
            self.assertGreater(_candidate_indices.cache_info().hits, hits)
            # Other scoring contracts and heterogeneous prediction positions.
            for metric in ("logit_difference", "target_log_probability"):
                group = [Scoring(metric, i % 3, (i,), (i+1,) if metric == "logit_difference" else ()) for i in range(5)]
                from nnets_transformer.metrics.behavioral import score
                reference = torch.stack([score(logits[i:i+1], s) for i, s in enumerate(group)])
                torch.testing.assert_close(score_batch(logits, group), reference, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "one scoring contract"):
            score_batch(logits, [specs[0], Scoring("logit_difference", 0, (0,), (1,))])

    def test_direct_hooks_match_tl_cache_and_preserve_external_hooks_after_error(self):
        adapter = tiny()
        examples, _ = Induction().prepare(adapter, config(Path("unused")).task)
        examples[1] = replace(examples[1], tokens=examples[1].tokens[:-1], attention_mask=(True, False, True, True, True, True, True),
                              scoring=replace(examples[1].scoring, position=6))
        observed = adapter.selections("head")
        chosen = next(s for s in observed if s.head == 1)
        tokens, masks = padded_inputs(examples, 8, "cpu")
        self.assertEqual(tokens[1, -1], 0)
        self.assertFalse(masks[1, 1])
        self.assertFalse(masks[1, -1])
        def replace_hook(value, hook):
            Zero().apply(value[:1], chosen, adapter.components[chosen.hook])
        names = list(dict.fromkeys(s.hook for s in observed))
        with torch.inference_mode(), adapter.model.hooks(fwd_hooks=[(chosen.hook, replace_hook)]):
            expected, cache = adapter.model.run_with_cache(tokens, attention_mask=masks, names_filter=names)
        actual, fresh = adapter.forward_batch(examples, observed, [(chosen, Zero()), None], pad_length=8)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name in names:
            torch.testing.assert_close(fresh[name], cache[name], atol=0, rtol=0)
        point = adapter.model.hook_dict[chosen.hook]
        external = point.register_forward_hook(lambda module, args, value: None)
        before = {n: tuple(p._forward_hooks) for n, p in adapter.model.hook_dict.items()}
        class Fail:
            def apply(self, *args):
                raise RuntimeError("deliberate failure")
        with patch.object(adapter.model, "reset_hooks", side_effect=AssertionError("global reset")):
            adapter.forward_batch(examples, observed)
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                adapter.forward_batch(examples, observed, [(chosen, Fail()), None])
        self.assertEqual(before, {n: tuple(p._forward_hooks) for n, p in adapter.model.hook_dict.items()})
        external.remove()

    def test_prepared_cache_invalidates_and_copies_condition_data(self):
        adapter = tiny()
        adapter.provenance["tokenizer_files"] = {"tokenizer.json": "fixture"}
        task = config(Path("unused")).task
        prepared = PreparedTasks()
        with patch.object(Induction, "prepare", autospec=True, side_effect=Induction.prepare) as prepare:
            examples, dataset = prepared.prepare(adapter, task)
            dataset["sample_manifest"][0]["roles"]["test"] = "mutation"
            adapter.provenance["tokenizer_revision"] = "second-checkpoint"
            again, manifest = prepared.prepare(adapter, task)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(examples, again)
            self.assertNotIn("test", manifest["sample_manifest"][0]["roles"])
            self.assertEqual(manifest["tokenizer_revision"], "second-checkpoint")
            prepared.prepare(adapter, replace(task, seed=task.seed+1))
            adapter.provenance["tokenizer_files"] = {"tokenizer.json": "changed"}
            prepared.prepare(adapter, task)
            self.assertEqual(prepare.call_count, 3)

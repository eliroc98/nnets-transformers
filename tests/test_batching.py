from dataclasses import replace
from pathlib import Path
import unittest
import torch
from test_fresh_runner import tiny, config
from nnets_transformer.execution.runner import serial_records
from nnets_transformer.execution.batching import records
from nnets_transformer.tasks.induction import Induction
from nnets_transformer.interventions.replacements import Zero, Identity


class BatchingContracts(unittest.TestCase):
    def test_serial_agreement_independent_slices_padding_and_forward_reuse(self):
        adapter = tiny()
        cfg = config(Path('unused'))
        cfg.task.examples = 3
        examples, dataset = Induction().prepare(adapter, cfg.task)
        shorter, _ = Induction().prepare(adapter, replace(cfg.task, pattern_length=3))
        examples[1] = replace(shorter[1], id=examples[1].id)
        sources = adapter.selections('head')[:7]
        observations = adapter.selections('head')
        work = [(f'{e}-{i}', example, source) for e, example in enumerate(examples) for i, source in enumerate(sources)]
        reference = {key: record for key, _, record in serial_records(work, {w[0] for w in work}, adapter, observations, Zero(), dataset)}
        serial_count = adapter.forward_count
        optimized = {key: record for key, _, record in records(work, {w[0] for w in work}, adapter, observations, Zero(), dataset, cfg.execution)}
        optimized_count = adapter.forward_count-serial_count
        self.assertEqual(serial_count, 42)
        self.assertEqual(optimized_count, 7)
        for key, expected in reference.items():
            actual = optimized[key]
            self.assertAlmostEqual(actual['behavior']['patched_score'], expected['behavior']['patched_score'], places=5)
            for observed, wanted in zip(actual['activations'], expected['activations']):
                for name in ('cosine_similarity', 'relative_l2_change'):
                    if wanted[name] is None:
                        self.assertIsNone(observed[name])
                    else:
                        self.assertAlmostEqual(observed[name], wanted[name], places=5)
        self.assertTrue(all(not hook.fwd_hooks for hook in adapter.model.hook_dict.values()))
        # Different hooks/heads in one forward remain isolated by batch slice.
        logits, cache = adapter.forward_batch(examples[:2], observations,
            [(sources[0], Identity()), (sources[1], Zero())])
        baseline, _ = adapter.forward(examples[0], observations)
        torch.testing.assert_close(logits[0, :len(examples[0].tokens)], baseline[0], atol=2e-6, rtol=2e-6)

    def test_cosine_only_metrics_match_full_metrics(self):
        import numpy as np
        adapter = tiny()
        cfg = config(Path('unused'))
        cfg.task.examples = 3
        examples, dataset = Induction().prepare(adapter, cfg.task)
        shorter, _ = Induction().prepare(adapter, replace(cfg.task, pattern_length=3))
        examples[1] = replace(shorter[1], id=examples[1].id)   # right padding exercised
        for resolution in ('component', 'head'):
            observations = adapter.selections(resolution)
            sources = observations[:6]
            work = [(f'{e}-{i}', example, s) for e, example in enumerate(examples) for i, s in enumerate(sources)]
            full = {k: r for k, _, r in records(work, {w[0] for w in work}, adapter, observations, Zero(), dataset, cfg.execution)}
            cosine_execution = replace(cfg.execution, activation_metrics='cosine')
            fast = {k: r for k, _, r in records(work, {w[0] for w in work}, adapter, observations, Zero(), dataset, cosine_execution)}
            for key, record in full.items():
                expected = np.array([a['cosine_similarity'] for a in record['activations']], dtype=np.float32)
                array = fast[key]['activations']
                self.assertEqual(array.shape, (len(observations), 4))
                np.testing.assert_allclose(array[:, 0], expected, atol=1e-6)
                self.assertTrue(np.isnan(array[:, 1:]).all())
                self.assertEqual(fast[key]['behavior'], record['behavior'])

    def test_pending_only_and_hook_cleanup_on_error(self):
        adapter = tiny()
        cfg = config(Path('unused'))
        examples, dataset = Induction().prepare(adapter, cfg.task)
        sources = adapter.selections('component')[:2]
        work = [(f'{e}-{i}', example, source) for e, example in enumerate(examples) for i, source in enumerate(sources)]
        observed = list(records(work, {'1-0'}, adapter, sources, Zero(), dataset, cfg.execution))
        self.assertEqual([r[0] for r in observed], ['1-0'])
        self.assertEqual(adapter.forward_count, 2)
        class Fail:
            def apply(self, *args):
                raise RuntimeError('deliberate batch failure')
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            adapter.forward_batch(examples, sources, [(sources[0], Fail()), None])
        self.assertTrue(all(not hook.fwd_hooks for hook in adapter.model.hook_dict.values()))

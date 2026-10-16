"""Bounded H100 serial/optimized/precision comparison; no research sweep."""
import time
from pathlib import Path
import torch
import numpy as np
from nnets_transformer.config import load
from nnets_transformer.observability import Job, atomic_json
from nnets_transformer.models import load_adapter
from nnets_transformer.tasks.induction import Induction
from nnets_transformer.execution.runner import serial_records
from nnets_transformer.execution.batching import records
from nnets_transformer.interventions.replacements import Zero

with Job('t03-profile') as job:
    config, _ = load('configs', overrides=['+experiment=induction_smoke', 'resources=h100',
        'resources.disable_optional_vision=true', 'task.examples=4'])
    loaded = time.monotonic()
    adapter = load_adapter(config.model, config.resources, job)
    load_seconds = time.monotonic()-loaded
    examples, dataset = Induction().prepare(adapter, config.task)
    observations = adapter.selections('component')
    sources = observations[:8]
    work = [(f'{e}-{i}', example, selection) for e, example in enumerate(examples) for i, selection in enumerate(sources)]
    pending = {key for key, _, _ in work}
    adapter.forward(examples[0], observations)  # warmup, excluded
    def measure(backend):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        forwards = adapter.forward_count
        started = time.monotonic()
        if backend == 'serial':
            result = list(serial_records(work, pending, adapter, observations, Zero(), dataset))
        else:
            result = list(records(work, pending, adapter, observations, Zero(), dataset, config.execution))
        torch.cuda.synchronize()
        return {key: record for key, _, record in result}, {'seconds': time.monotonic()-started,
                'forwards': adapter.forward_count-forwards, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    reference, serial = measure('serial')
    optimized, fp32 = measure('optimized')
    adapter.model.to(torch.bfloat16)
    adapter.forward(examples[0], observations)
    reduced, bf16 = measure('optimized')
    def errors(actual):
        score_error = max(abs(reference[k]['behavior']['patched_score']-actual[k]['behavior']['patched_score']) for k in reference)
        reference_cos = np.array([[o['cosine_similarity'] for o in reference[k]['activations']] for k in sorted(reference)])
        actual_cos = np.array([[o['cosine_similarity'] for o in actual[k]['activations']] for k in sorted(reference)])
        flips = {str(t): int(np.count_nonzero((reference_cos < t) != (actual_cos < t))) for t in [.1, .3, .5, .7]}
        from scipy.stats import spearmanr
        rank = float(spearmanr(reference_cos.ravel(), actual_cos.ravel()).statistic)
        return {'max_score_absolute_error': score_error, 'max_cosine_absolute_error': float(abs(reference_cos-actual_cos).max()),
                'threshold_flips': flips, 'cosine_spearman': rank}
    fp32.update(errors(optimized))
    bf16.update(errors(reduced))
    report = {'device': torch.cuda.get_device_name(), 'load_seconds': load_seconds, 'examples': len(examples),
              'interventions_per_example': len(sources), 'serial_float32': serial, 'optimized_float32': fp32,
              'optimized_bfloat16': bf16, 'speedup_float32': serial['seconds']/fp32['seconds'],
              'speedup_bfloat16': serial['seconds']/bf16['seconds'],
              'bf16_default': False, 'precision_policy': 'FP32 default; BF16 opt-in requires per-study sensitivity checks'}
    assert fp32['max_score_absolute_error'] < 1e-4 and fp32['max_cosine_absolute_error'] < 1e-5, report
    assert bf16['max_score_absolute_error'] < .15 and bf16['max_cosine_absolute_error'] < .03, report
    atomic_json(job.directory/'profile.json', report)
    print(report)
    job.artifact(job.directory/'profile.json')

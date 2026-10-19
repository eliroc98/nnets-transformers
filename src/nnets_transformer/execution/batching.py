"""Bounded selective caches and independent example/intervention batches."""
from dataclasses import asdict
import torch
from nnets_transformer.metrics.activation import compare_batch
from nnets_transformer.metrics.behavioral import score_batch
from nnets_transformer.models.inputs import padded_inputs


def records(work, pending, adapter, observations, replacement, dataset, execution, references=None):
    # Batch boundaries are fixed over the full work list, so a resumed run recomputes
    # exactly the batches (and padding) an uninterrupted run would.
    all_ids = list(dict.fromkeys(example.id for _, example, _ in work))
    by_example = {}
    for key, example, selection in work:
        if key in pending:
            by_example.setdefault(example.id, []).append((key, example, selection))
    for offset in range(0, len(all_ids), execution.example_batch_size):
        ids = [i for i in all_ids[offset:offset+execution.example_batch_size] if i in by_example]
        if not ids:
            continue
        examples = [by_example[example_id][0][1] for example_id in ids]
        position = {example.id: i for i, example in enumerate(examples)}
        length = max(len(e.tokens) for e in examples)
        if references is not None:
            recipients, clean_cache, clean_scores, corrupt_scores, replacements = references.batch(examples, observations, length)
        else:
            clean_logits, clean_cache = adapter.forward_batch(examples, observations, pad_length=length)
            clean_scores = score_batch(clean_logits, [e.scoring for e in examples]).cpu().tolist()
            corrupt_scores = [None]*len(examples)
            recipients, replacements = examples, [replacement]*len(examples)
            del clean_logits
        items = [item for example_id in ids for item in by_example[example_id]]
        _, recipient_masks = padded_inputs(recipients, length, clean_cache[observations[0].hook].device)
        batch_size = execution.intervention_batch_size
        for start in range(0, len(items), batch_size):
            batch = items[start:start+batch_size]
            original_examples = [e for _, e, _ in batch]
            batch_examples = [recipients[position[e.id]] for e in original_examples]
            logits, patched_cache = adapter.forward_batch(batch_examples, observations,
                [(s, replacements[position[e.id]]) for _, e, s in batch], pad_length=length)
            patched_scores = score_batch(logits, [e.scoring for e in batch_examples]).cpu().tolist()
            indices = torch.tensor([position[e.id] for e in batch_examples], device=logits.device)
            masks = recipient_masks.index_select(0, indices)
            if getattr(execution, "activation_metrics", "full") == "cosine":
                import numpy as np
                from nnets_transformer.metrics.activation import cosine_batch
                cosines = cosine_batch(clean_cache, patched_cache, observations, adapter.components, masks, indices)
                # [observations, fields] with only cosine computed; other fields are marked not computed.
                activations = []
                for row in cosines:
                    array = np.full((len(observations), 4), np.nan, dtype=np.float32)
                    array[:, 0] = row
                    activations.append(array)
            else:
                activations = compare_batch(clean_cache, patched_cache, observations, adapter.components, masks, indices)
            for (key, example, selection), patched, activation in zip(batch, patched_scores, activations):
                clean = clean_scores[position[example.id]]
                from nnets_transformer.interventions.references import behavior
                behavioral = behavior(example, dataset, clean, corrupt_scores[position[example.id]], patched, references.config) if references else {
                    "metric": example.scoring.metric, "units": dataset["units"], "clean_score": clean, "patched_score": patched, "signed_change": patched-clean}
                yield key, example, {"schema_version": "fresh-observation-v1", "example_id": example.id,
                    "intervention": asdict(selection), "intervention_id": selection.id, "behavior": behavioral,
                    "activations": activation}
            del logits, patched_cache
        del clean_cache

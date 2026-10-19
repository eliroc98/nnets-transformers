"""Per-example FP32 behavioral scores, anchored to prepared clean labels."""
from functools import lru_cache
import torch


def score(logits, specification):
    # The serial API accepts one prepared example; batching belongs to T03.
    selected = logits[0, specification.position].float()
    if specification.metric == "target_log_probability":
        return selected.log_softmax(-1)[specification.targets[0]]
    if specification.metric == "logit_difference":
        return selected[specification.targets[0]] - selected[specification.contrasts[0]]
    if specification.metric == "probability_group_difference":
        probabilities = selected.softmax(-1)
        return probabilities[list(specification.targets)].sum() - probabilities[list(specification.contrasts)].sum()
    raise ValueError(f"Unsupported metric: {specification.metric}")


@lru_cache(maxsize=128)
def _batch_indices(specifications, device):
    """Bounded, reusable indices; construct on CPU and transfer once per batch plan."""
    if not specifications or len({s.metric for s in specifications}) != 1:
        raise ValueError("A batch must use one scoring contract")
    positions = torch.tensor([s.position for s in specifications], device=device)
    rows = torch.arange(len(specifications), device=device)
    targets = torch.tensor([s.targets[0] for s in specifications], device=device)
    metric = specifications[0].metric
    contrasts = None
    group = None
    if metric == "logit_difference":
        contrasts = torch.tensor([s.contrasts[0] for s in specifications], device=device)
    elif metric == "probability_group_difference":
        candidates = [_candidate_indices(s, device) for s in specifications]
        group = (torch.tensor([row for row, s in enumerate(specifications)
                               for _ in range(len(s.targets)+len(s.contrasts))], device=device),
                 torch.cat([c[0] for c in candidates]), torch.cat([c[1] for c in candidates]))
    return positions, rows, targets, contrasts, group


@lru_cache(maxsize=1024)
def _candidate_indices(specification, device):
    """Transfer each example's candidates once, even across different batch plans."""
    s = specification
    return (torch.tensor(s.targets+s.contrasts, device=device, dtype=torch.long),
            torch.tensor((1.,)*len(s.targets)+(-1.,)*len(s.contrasts), device=device, dtype=torch.float32))


def score_batch(logits, specifications):
    """Full-vocabulary FP32 reductions with cached, batched candidate indexing."""
    specifications = tuple(specifications)
    positions, rows, targets, contrasts, group = _batch_indices(specifications, logits.device)
    selected = logits[rows, positions].float()
    metric = specifications[0].metric
    if metric == "target_log_probability":
        return selected.log_softmax(-1)[rows, targets]
    if metric == "logit_difference":
        return selected[rows, targets]-selected[rows, contrasts]
    if metric == "probability_group_difference":
        weights = torch.zeros_like(selected)
        candidate_rows, candidate_ids, signs = group
        weights[candidate_rows, candidate_ids] = signs
        return (selected.softmax(-1)*weights).sum(-1)
    raise ValueError(metric)

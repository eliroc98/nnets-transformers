"""Per-example FP32 behavioral scores, anchored to prepared clean labels."""
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


def score_batch(logits, specifications):
    """Vectorized vocabulary reductions; candidate groups may differ by example."""
    device = logits.device
    positions = torch.tensor([s.position for s in specifications], device=device)
    selected = logits[torch.arange(len(specifications), device=device), positions].float()
    if len({s.metric for s in specifications}) != 1:
        raise ValueError("A batch must use one scoring contract")
    metric = specifications[0].metric
    targets = torch.tensor([s.targets[0] for s in specifications], device=device)
    rows = torch.arange(len(specifications), device=device)
    if metric == "target_log_probability":
        return selected.log_softmax(-1)[rows, targets]
    if metric == "logit_difference":
        contrasts = torch.tensor([s.contrasts[0] for s in specifications], device=device)
        return selected[rows, targets]-selected[rows, contrasts]
    if metric == "probability_group_difference":
        weights = torch.zeros_like(selected)
        for row, spec in enumerate(specifications):
            weights[row, list(spec.targets)] = 1
            weights[row, list(spec.contrasts)] = -1
        return (selected.softmax(-1)*weights).sum(-1)
    raise ValueError(metric)

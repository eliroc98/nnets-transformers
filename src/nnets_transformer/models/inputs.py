"""Bounded input tensor preparation shared by model forwards and metric masks."""
from functools import lru_cache
import torch


@lru_cache(maxsize=16)
def _padded(tokens, masks, length, device):
    # One transfer each, with no per-row GPU assignment or implicit index tensors.
    return (torch.tensor([t + (0,) * (length-len(t)) for t in tokens], device=device, dtype=torch.long),
            torch.tensor([m + (False,) * (length-len(m)) for m in masks], device=device, dtype=torch.bool))


def padded_inputs(examples, length, device):
    return _padded(tuple(e.tokens for e in examples), tuple(e.attention_mask for e in examples),
                   length, torch.device(device))

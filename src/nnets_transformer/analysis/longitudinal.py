"""Shared CPU role definitions; no inherited historical phase boundaries."""
import math
import numpy as np
import pandas as pd


def upper_tail(values, quantile=.95):
    """Strictly above a linear quantile; ties at the threshold are excluded."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0,1]")
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return set()
    return set(series.index[series > series.quantile(quantile, interpolation="linear")])


def jaccard(first, second):
    union = first | second
    return len(first & second) / len(union) if union else None


def role_dynamics(components, quantile=.95):
    """Membership, turnover and final-set convergence per compatible graph key."""
    rows = []
    keys = ["run_id", "threshold", "resolution", "graph_version", "metric"]
    for key, group in components.groupby(keys, dropna=False):
        steps = sorted(group.step.unique())
        sets = {step: upper_tail(group[group.step == step].set_index("component").value, quantile) for step in steps}
        previous = None
        for step in steps:
            selected = sets[step]
            rows.append(dict(zip(keys, key)) | {"step": int(step), "members": sorted(selected),
                "size": len(selected), "churn": None if previous is None else
                    (1-jaccard(previous, selected) if previous | selected else None),
                "final_jaccard": jaccard(selected, sets[steps[-1]]), "quantile": quantile,
                "tie_policy": "strict_above_linear_quantile"})
            previous = selected
    return pd.DataFrame(rows)


def checkpoint_similarity(components, metric, quantile=.95):
    subset = components[components.metric == metric]
    sets = {step: upper_tail(group.set_index("component").value, quantile) for step, group in subset.groupby("step")}
    steps = sorted(sets)
    return steps, [[jaccard(sets[a], sets[b]) for b in steps] for a in steps]

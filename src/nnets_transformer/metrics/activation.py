"""Propagation metrics over valid sequence positions, with zero-norm status."""
import torch
import torch.nn.functional as F

ACTIVATION_VERSION = "masked-cosine-relative-l2-v1"


def compare(clean, patched, selection, component, attention_mask):
    if selection.head is not None:
        clean = clean.select(component.head_axis, selection.head)
        patched = patched.select(component.head_axis, selection.head)
    clean, patched = clean.float(), patched.float()
    if clean.shape != patched.shape or clean.ndim not in {3, 4}:
        raise ValueError("Unexpected activation axes or reference shape")
    valid = torch.tensor([attention_mask], dtype=torch.bool, device=clean.device)
    while valid.ndim < clean.ndim-1:
        valid = valid.unsqueeze(-1)
    valid = valid.expand(clean.shape[:-1])
    cosine = F.cosine_similarity(clean, patched, dim=-1)[valid].mean()
    clean_norm = clean.norm(dim=-1)[valid]
    patched_norm = patched.norm(dim=-1)[valid]
    numerator = (patched-clean).square().sum(-1)[valid].sum().sqrt()
    denominator = clean.square().sum(-1)[valid].sum().sqrt()
    values = torch.stack((cosine, numerator, denominator, (clean_norm == 0).float().mean(),
                          (patched_norm == 0).float().mean())).cpu().tolist()
    cosine, numerator, denominator, clean_zero, patched_zero = values
    return {"cosine_similarity": cosine,
            "relative_l2_change": numerator / denominator if denominator > 1e-12 else None,
            "relative_l2_status": "valid" if denominator > 1e-12 else "undefined_clean_zero_norm",
            "clean_zero_norm_fraction": clean_zero, "patched_zero_norm_fraction": patched_zero}


def compare_batch(clean_cache, patched_cache, observations, components, masks, indices):
    """Reduce positions/features on device; transfer one compact result tensor."""
    results = []
    for selection in observations:
        clean = clean_cache[selection.hook].index_select(0, indices)
        patched = patched_cache[selection.hook]
        component = components[selection.hook]
        if selection.head is not None:
            clean = clean.select(component.head_axis, selection.head)
            patched = patched.select(component.head_axis, selection.head)
        clean, patched = clean.float(), patched.float()
        valid = masks
        while valid.ndim < clean.ndim-1:
            valid = valid.unsqueeze(-1)
        valid = valid.expand(clean.shape[:-1])
        axes = tuple(range(1, valid.ndim))
        count = valid.sum(axes).clamp_min(1)
        cosine = (F.cosine_similarity(clean, patched, dim=-1)*valid).sum(axes)/count
        numerator = ((patched-clean).square().sum(-1)*valid).sum(axes).sqrt()
        denominator = (clean.square().sum(-1)*valid).sum(axes).sqrt()
        clean_zero = ((clean.norm(dim=-1) == 0)*valid).sum(axes)/count
        patched_zero = ((patched.norm(dim=-1) == 0)*valid).sum(axes)/count
        results.append(torch.stack((cosine, numerator, denominator, clean_zero, patched_zero), -1))
    values = torch.stack(results, 1).cpu().tolist()
    return [[{"observation_id": selection.id, "cosine_similarity": cosine,
              "relative_l2_change": numerator/denominator if denominator > 1e-12 else None,
              "relative_l2_status": "valid" if denominator > 1e-12 else "undefined_clean_zero_norm",
              "clean_zero_norm_fraction": clean_zero, "patched_zero_norm_fraction": patched_zero}
             for selection, (cosine, numerator, denominator, clean_zero, patched_zero) in zip(observations, row)]
            for row in values]


def cosine_batch(clean_cache, patched_cache, observations, components, masks, indices):
    """Cosine-only reduction, same definition as compare_batch's cosine: one feature-axis cosine
    map per hook (shared by its head observations), masked mean over valid positions (and heads).
    Returns a [slices, observations] float32 numpy array."""
    maps, values = {}, []
    for selection in observations:
        if selection.hook not in maps:
            clean = clean_cache[selection.hook].index_select(0, indices).float()
            maps[selection.hook] = F.cosine_similarity(clean, patched_cache[selection.hook].float(), dim=-1)
        cosine = maps[selection.hook]
        if selection.head is not None:
            cosine = cosine.select(components[selection.hook].head_axis, selection.head)
        valid = masks
        while valid.ndim < cosine.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.expand(cosine.shape)
        axes = tuple(range(1, cosine.ndim))
        values.append((cosine*valid).sum(axes)/valid.sum(axes).clamp_min(1))
    return torch.stack(values, 1).cpu().numpy()

"""Deliberately serial FP32 reference for the historical single-passage task.

No baseline reuse or intervention batching: two forwards per intervention.
The fixed token ID belongs only to the versioned historical scoring protocol.
"""
import torch
import torch.nn.functional as F

SCORING_VERSION = "legacy-induction-token2205-v1"


def observation_hooks(model):
    return [name for name in model.hook_dict if not any(part in name for part in
            ("pattern", ".hook_attn_", "attn.hook_result", "_input", "resid", "hook_mlp", "mlp.hook_pre", "ln"))]


def selections(model):
    for name in observation_hooks(model):
        yield name, None
        if "attn" in name:
            for head in range(model.cfg.n_heads):
                yield name, head


def serial_intervention(model, tokens, hook_name, head=None, *, replacement="zero"):
    """All-position whole-component/head replacement and historical reductions.

    Identity replacement is a validation control. Hooks are scoped by the model
    context managers, including when an observation hook raises an exception.
    """
    if replacement not in {"zero", "identity"}:
        raise ValueError("Reference supports zero or identity only")
    if head is not None and (head < 0 or head >= model.cfg.n_heads or "attn" not in hook_name):
        raise ValueError("Invalid attention head")
    observed = observation_hooks(model)
    if hook_name not in observed:
        raise ValueError(f"Unsupported historical intervention: {hook_name}")
    diffs = {}
    with torch.no_grad():
        (clean_logits, clean_loss), cache = model.run_with_cache(tokens, return_type="both")
        def replace(value, hook):
            if replacement == "zero":
                if head is None:
                    value[:] = 0
                else:
                    value[:, :, head, :] = 0
        def observe(value, hook):
            diffs[hook.name] = F.cosine_similarity(value, cache[hook.name], dim=-1).mean().item()
            if hook.name.endswith(("hook_q", "hook_k", "hook_v", "hook_z", "hook_result")):
                for h in range(model.cfg.n_heads):
                    diffs[f"{hook.name}.{h}"] = F.cosine_similarity(value[:, :, h, :], cache[hook.name][:, :, h, :], dim=-1).mean().item()
        logits, loss = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, replace)] + [(n, observe) for n in observed], return_type="both")
        diffs["logit"] = F.cosine_similarity(logits, clean_logits, dim=-1).mean().item()
        return diffs, {
            "patched_logit": logits[0, -1, 2205].item(), "patched_loss": loss.item(),
            "clean_logit": clean_logits[0, -1, 2205].item(), "clean_loss": clean_loss.item(),
        }

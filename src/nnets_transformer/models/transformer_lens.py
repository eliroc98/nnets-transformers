"""Initial TransformerLens adapter for existing component and attention-head sites."""
from dataclasses import asdict
from pathlib import Path
import hashlib

import torch
from nnets_transformer.records import Component, Selection
from .inputs import padded_inputs

INVENTORY_VERSION = "tl-existing-sites-v2"
PYTHIA_TOKENS_PER_STEP = 2_097_152


class TransformerLensAdapter:
    def __init__(self, model, provenance):
        self.model = model.eval()
        self.tokenizer = model.tokenizer
        self.provenance = provenance
        self.vocab_size = model.cfg.d_vocab
        self.context_length = model.cfg.n_ctx
        self.query_heads = model.cfg.n_heads
        self.key_value_heads = model.cfg.n_key_value_heads or self.query_heads
        if self.key_value_heads != self.query_heads:
            raise ValueError("Grouped-query attention requires a separately validated adapter")
        components = []
        for hook in ("hook_embed", "hook_pos_embed"):
            if hook in model.hook_dict:
                components.append(Component(hook, -1))
        for layer in range(model.cfg.n_layers):
            for site in ("q", "k", "v", "rot_q", "rot_k", "z"):
                hook = f"blocks.{layer}.attn.hook_{site}"
                if hook in model.hook_dict:
                    heads = self.key_value_heads if site in {"k", "v", "rot_k"} else self.query_heads
                    components.append(Component(hook, layer, heads, 2))
            hook = f"blocks.{layer}.mlp.hook_post"
            if hook in model.hook_dict:
                components.append(Component(hook, layer))
        self.inventory = tuple(components)
        self.components = {c.hook: c for c in components}

    def selections(self, resolution):
        if resolution not in {"component", "head", "mixed"}:
            raise ValueError("Invalid component resolution")
        result = []
        for component in self.inventory:
            if resolution != "head" or component.head_count is None:
                result.append(Selection(component.hook))
            if resolution != "component" and component.head_count is not None:
                result.extend(Selection(component.hook, h) for h in range(component.head_count))
        return result

    def validate_selection(self, selection):
        if selection.hook not in self.components:
            raise ValueError(f"Unsupported component: {selection.hook}")
        component = self.components[selection.hook]
        if selection.head is not None and (component.head_count is None or
                type(selection.head) is not int or not 0 <= selection.head < component.head_count):
            raise ValueError(f"Invalid attention head: {selection}")
        return component

    def forward(self, example, observations, intervention=None):
        return self.forward_batch([example], observations, [intervention])

    def forward_batch(self, examples, observations, interventions=None, pad_length=None):
        """One independent intervention per example slice; right padding is masked."""
        if not examples:
            raise ValueError("Empty forward batch")
        interventions = interventions or [None] * len(examples)
        if len(interventions) != len(examples):
            raise ValueError("Intervention/example batch mismatch")
        for example in examples:
            example.validate(self.vocab_size, self.context_length)
        for selection in observations:
            self.validate_selection(selection)
        length = pad_length or max(len(e.tokens) for e in examples)
        if length > self.context_length or length < max(len(e.tokens) for e in examples):
            raise ValueError("Invalid padded context length")
        device = self.model.cfg.device
        tokens, mask = padded_inputs(examples, length, device)
        grouped = {}
        for index, intervention in enumerate(interventions):
            if intervention is not None:
                selection, replacement = intervention
                component = self.validate_selection(selection)
                grouped.setdefault(selection.hook, []).append((index, selection, replacement, component))
        def make_hook(entries):
            def replace(value, hook):
                for index, selection, replacement, component in entries:
                    replacement.apply(value[index:index+1], selection, component)
            return replace
        hooks = [(name, make_hook(entries)) for name, entries in grouped.items()]
        wanted = list(dict.fromkeys(s.hook for s in observations))
        cache, handles = {}, []
        def capture(name):
            def save(value, hook):
                cache[name] = value.detach()
            return save
        def register(name, callback):
            point = self.model.hook_dict[name]
            # Native removable handles preserve unrelated/permanent TL hooks and
            # avoid scanning/resetting every model hook point after each forward.
            handles.append(point.register_forward_hook(
                lambda module, args, value: callback(value, hook=module)))
        try:
            for name, callback in hooks:
                register(name, callback)
            # Cache after the replacement, as with TL run_with_cache inside hooks().
            for name in wanted:
                register(name, capture(name))
            with torch.inference_mode():
                logits = self.model(tokens, attention_mask=mask)
        finally:
            for handle in reversed(handles):
                handle.remove()
        self.forward_count = getattr(self, "forward_count", 0) + 1
        return logits, {name: cache[name] for name in wanted}


def load(config, resources, job):
    if resources.disable_optional_vision:
        from transformers.utils import import_utils
        import_utils._torchvision_available = False
        job.event("compatibility", message="Optional Transformers vision imports explicitly disabled")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    job.stage("checkpoint_loading", model=config.id, checkpoint=config.checkpoint, revision=config.revision)
    from .checkpoints import download_patterns
    # Exactly the weight format the loader uses (also what a sweep prefetches), so no redundant download.
    model_path = Path(snapshot_download(config.id, revision=config.revision, cache_dir=config.cache_dir,
        allow_patterns=download_patterns(config.id, config.revision), max_workers=8))
    tokenizer_path = model_path if config.tokenizer_revision == config.revision else Path(snapshot_download(
        config.id, revision=config.tokenizer_revision, cache_dir=config.cache_dir, allow_patterns=["*.json", "*.txt"]))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    preprocessing = dict(fold_ln=False, center_unembed=False, center_writing_weights=False,
                         fold_value_biases=False, refactor_factored_attn_matrices=False)
    model = HookedTransformer.from_pretrained(config.id, revision=config.revision, tokenizer=tokenizer,
        cache_dir=config.cache_dir, device=resources.device, dtype=getattr(torch, resources.dtype), **preprocessing)
    if model.cfg.n_layers != config.n_layers or not model.cfg.parallel_attn_mlp:
        raise ValueError("Loaded model does not match the configured Pythia architecture")
    import re
    seed = re.search(r"-seed(\d)$", config.id)
    step = int(config.checkpoint.removeprefix("step"))
    provenance = {
        "model_id": config.id, "model_revision": config.revision,
        "tokenizer_revision": config.tokenizer_revision, "checkpoint": config.checkpoint,
        "stage": "pretraining", "step": step,
        # Pythia: 1,024 sequences x 2,048 tokens per optimizer step for every scale.
        "tokens_seen": step*PYTHIA_TOKENS_PER_STEP, "tokens_seen_convention": "pythia-2097152-tokens-per-step",
        "training_seed": f"seed{seed.group(1)}" if seed else "standard-release",
        "deduplicated_data": "-deduped" in config.id,
        "preprocessing": preprocessing, "dtype": resources.dtype, "device": resources.device,
        "optional_vision_disabled": resources.disable_optional_vision,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "query_heads": model.cfg.n_heads, "key_value_heads": model.cfg.n_key_value_heads or model.cfg.n_heads,
        "architecture": "parallel", "inventory_version": INVENTORY_VERSION,
        "tokenizer_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tokenizer_path.iterdir()
                            if p.suffix in {".json", ".txt"} and p.is_file()},
    }
    adapter = TransformerLensAdapter(model, provenance)
    provenance["components"] = [asdict(c) for c in adapter.inventory]
    return adapter

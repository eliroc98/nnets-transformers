"""Lazy model registry; artifact readers never load inference dependencies."""

def load_adapter(config, resources, job):
    from nnets_transformer.config import PYTHIA
    if not PYTHIA.fullmatch(config.id):
        raise ValueError(f"Unvalidated model adapter: {config.id}")
    from .transformer_lens import load
    return load(config, resources, job)

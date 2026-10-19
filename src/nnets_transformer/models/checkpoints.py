"""Checkpoint branch resolution to immutable commits (no model imports)."""


def branches(model_id):
    from huggingface_hub import HfApi
    return {ref.name: ref.target_commit for ref in HfApi().list_repo_refs(model_id).branches}


def resolve_revisions(model_id, checkpoints):
    available = branches(model_id)
    missing = [c for c in checkpoints if c not in available]
    if missing:
        raise ValueError(f"{model_id} lacks checkpoint branches: {missing}")
    return {c: available[c] for c in checkpoints}


def evict(cache_dir, revisions, job):
    """Delete downloaded snapshots for revisions (bounded checkpoint storage)."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir(cache_dir)
    strategy = info.delete_revisions(*revisions)
    job.event("checkpoint_cache_evicted", revisions=list(revisions), freed_bytes=strategy.expected_freed_size)
    strategy.execute()


def download_patterns(model_id, revision):
    """Config/tokenizer files plus one weight format: safetensors when present (what the loader
    uses), otherwise the PyTorch .bin files (single file or shards)."""
    from huggingface_hub import HfApi
    files = [s.rfilename for s in HfApi().model_info(model_id, revision=revision).siblings]
    safetensors = [f for f in files if f.endswith(".safetensors")]
    weights = safetensors or [f for f in files if f.endswith(".bin")]
    if not weights:
        raise ValueError(f"{model_id}@{revision} has no weight files")
    return ["*.json", "*.txt", *weights]


def prefetch(model_id, revision, cache_dir, workers=8):
    """Download one checkpoint into the shared cache (safe to run while another checkpoint computes)."""
    import time
    from huggingface_hub import snapshot_download
    started = time.perf_counter()
    snapshot_download(model_id, revision=revision, cache_dir=cache_dir,
                      allow_patterns=download_patterns(model_id, revision), max_workers=workers)
    return time.perf_counter() - started

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

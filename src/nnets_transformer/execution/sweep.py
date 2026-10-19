"""Checkpoint-grouped sweeps: load each checkpoint once, run every task x replacement cell.

Each cell is an ordinary fresh run (its own resolved YAML, manifest, shards and
resume). The sweep adds immutable checkpoint resolution, fixed-evaluation-data
checks across checkpoints, recorded skips and an atomic sweep manifest.
"""
from copy import deepcopy
import json
from pathlib import Path

from omegaconf import OmegaConf

from nnets_transformer.observability import atomic_json

SCHEMA = "checkpoint-sweep-v1"


def _load(schema, path, overrides=None):
    merged = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(path), overrides or {})
    return OmegaConf.to_object(merged)


def slug(model_id):
    return model_id.split("/")[-1]


def cells(config, config_dir, revisions):
    """Validated cell configurations, plus explicit records of incompatible skips."""
    from nnets_transformer.config import Task, Replacement, validate
    sweep, planned, skipped = config.sweep, [], []
    task_overrides = {k: v for k, v in (("examples", sweep.examples), ("calibration_examples", sweep.calibration_examples)) if v is not None}
    for checkpoint in sweep.checkpoints:
        for task_name in sweep.tasks:
            task = _load(Task, Path(config_dir) / "task" / f"{task_name}.yaml", task_overrides)
            for replacement_name in sweep.replacements:
                cell = deepcopy(config)
                cell.command = "run"
                cell.task = task
                cell.replacement = _load(Replacement, Path(config_dir) / "replacement" / f"{replacement_name}.yaml")
                cell.model.checkpoint = checkpoint
                cell.model.revision = cell.model.tokenizer_revision = revisions[checkpoint]
                cell.output_directory = str(Path(sweep.output_root) / slug(cell.model.id) / checkpoint / task_name / replacement_name)
                cell.execution.resume = config.execution.resume
                cell.sweep = type(config.sweep)()
                key = {"checkpoint": checkpoint, "task": task_name, "replacement": replacement_name}
                try:
                    validate(cell)
                except ValueError as exc:
                    if not sweep.skip_incompatible:
                        raise ValueError(f"Incompatible sweep cell {key}: {exc}") from exc
                    if checkpoint == sweep.checkpoints[0]:
                        skipped.append({**key, "checkpoint": "all", "reason": str(exc)})
                    continue
                planned.append((key, cell))
    return planned, skipped


def completed(cell):
    path = Path(cell.output_directory) / "manifest.json"
    return path.is_file() and json.loads(path.read_text()).get("state") == "completed"


def evaluation_fingerprint(output):
    """Evaluation/calibration/donor content, excluding per-checkpoint revision labels."""
    from .artifacts import fingerprint, load_dataset
    dataset = load_dataset(output)
    for name in ("tokenizer_revision", "dataset_sha256"):
        dataset.pop(name, None)
    return fingerprint(dataset)


class _Download:
    """One background checkpoint download (daemon thread: never delays an interrupted exit)."""
    def __init__(self, function, *args):
        import threading
        self.result, self.error = None, None
        def target():
            try:
                self.result = function(*args)
            except Exception as exc:  # reported when awaited; the loader then downloads normally
                self.error = exc
        self.thread = threading.Thread(target=target, name="prefetch", daemon=True)
        self.thread.start()

    def wait(self):
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.result


def run_sweep(config, config_dir, job):
    from .runner import run
    from nnets_transformer.tasks.prepared import PreparedTasks
    from nnets_transformer.models.checkpoints import resolve_revisions, evict
    sweep = config.sweep
    prepared_tasks = PreparedTasks()
    root = Path(sweep.output_root)
    job.stage("checkpoint_resolution", model=config.model.id, checkpoints=len(sweep.checkpoints))
    # One manifest per model: several models may share an output root.
    manifest_path = root / slug(config.model.id) / "sweep.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if previous is not None and not config.execution.resume:
        raise FileExistsError(f"{root} has a sweep manifest; pass execution.resume=true to continue it")
    if previous is not None and previous["model_id"] == config.model.id:
        # Branches can move; resumed sweeps keep the commits they started with.
        revisions = {c: previous["revisions"][c] for c in sweep.checkpoints if c in previous["revisions"]}
        missing = [c for c in sweep.checkpoints if c not in revisions]
        if missing:
            revisions.update(resolve_revisions(config.model.id, missing))
    else:
        revisions = resolve_revisions(config.model.id, sweep.checkpoints)
    planned, skipped = cells(config, config_dir, revisions)
    for key, cell in planned:
        if Path(cell.output_directory).exists() and not config.execution.resume and not completed(cell):
            raise FileExistsError(f"{cell.output_directory} exists; pass execution.resume=true")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": SCHEMA, "model_id": config.model.id, "revisions": revisions,
             "tasks": sweep.tasks, "replacements": sweep.replacements, "checkpoints": sweep.checkpoints, "compact": sweep.compact,
             "cells": [{**key, "output_directory": cell.output_directory, "state": "completed" if completed(cell) else "pending"}
                       for key, cell in planned],
             "skipped": skipped, "attempts": [*(previous or {}).get("attempts", []), str(job.directory)],
             "fixed_evaluation_data": {}}
    atomic_json(manifest_path, state)
    for record in skipped:
        job.event("sweep_cell_skipped", **record)
    sweep_resume = job.resume_command
    total = len(planned)
    done = sum(cell["state"] == "completed" for cell in state["cells"])
    job.progress("cells", 0, total, skipped=done)
    fingerprints = dict(previous.get("fixed_evaluation_data", {})) if previous else {}
    from .artifacts import compact
    if sweep.compact != "none":
        # Completed cells left uncompacted (e.g. interrupted right after completion).
        for key, cell in planned:
            if completed(cell):
                compact(cell.output_directory, sweep.compact, job)
    processed = 0
    failures = []
    # Completed cells must agree with the evaluation data recorded for their condition.
    for index, (key, cell) in enumerate(planned):
        condition = f"{key['task']}/{key['replacement']}"
        if completed(cell) and fingerprints.setdefault(condition, evaluation_fingerprint(cell.output_directory)) != \
                evaluation_fingerprint(cell.output_directory):
            state["cells"][index]["state"] = "inconsistent"
            failures.append({**key, "error": "completed cell's evaluation data differs from its condition's other checkpoints"})
    state["fixed_evaluation_data"] = fingerprints
    atomic_json(manifest_path, state)
    def remaining(checkpoint):
        return [(i, key, cell) for i, (key, cell) in enumerate(planned) if key["checkpoint"] == checkpoint and not completed(cell)
                and state["cells"][i]["state"] != "inconsistent"]
    work = [c for c in sweep.checkpoints if remaining(c)]
    downloads = {}
    def schedule(position):
        # Download the next checkpoint while this one computes; downloads then cost no GPU time.
        if sweep.prefetch and position < len(work) and work[position] not in downloads:
            from nnets_transformer.models.checkpoints import prefetch
            downloads[work[position]] = _Download(prefetch, config.model.id, revisions[work[position]], config.model.cache_dir)
    schedule(0)
    import time
    for position, checkpoint in enumerate(work):
        todo = remaining(checkpoint)
        if checkpoint in downloads:
            started = time.perf_counter()
            try:
                seconds = downloads.pop(checkpoint).wait()
                job.event("prefetch_ready", checkpoint=checkpoint, download_seconds=seconds,
                          waited_seconds=time.perf_counter()-started)
            except Exception as exc:
                job.event("prefetch_failed", severity="WARNING", checkpoint=checkpoint, error=str(exc))
        schedule(position+1)
        if not todo:
            continue
        from nnets_transformer.models import load_adapter
        model_config = deepcopy(todo[0][2].model)
        adapter = load_adapter(model_config, config.resources, job)
        try:
            for index, key, cell in todo:
                cell.execution.resume = Path(cell.output_directory).exists()
                yaml = OmegaConf.to_yaml(OmegaConf.structured(cell))
                job.stage("sweep_cell", **key)
                try:
                    run(cell, yaml, job, adapter=adapter, prepared_tasks=prepared_tasks)
                    condition = f"{key['task']}/{key['replacement']}"
                    current = evaluation_fingerprint(cell.output_directory)
                    if fingerprints.setdefault(condition, current) != current:
                        raise ValueError(f"Evaluation data changed across checkpoints for {condition}")
                    if sweep.compact != "none":
                        compact(cell.output_directory, sweep.compact, job)
                except Exception as exc:  # one failing condition must not discard the rest of the grid
                    failures.append({**key, "error": f"{type(exc).__name__}: {exc}"})
                    state["cells"][index].update(state="failed", error=failures[-1]["error"])
                    atomic_json(manifest_path, state)
                    job.event("sweep_cell_failed", severity="ERROR", **failures[-1])
                    continue
                finally:
                    # Cells set their own resume command; the sweep's command resumes everything.
                    job.resume_command = sweep_resume
                state["cells"][index]["state"] = "completed"
                state["fixed_evaluation_data"] = fingerprints
                atomic_json(manifest_path, state)
                processed += 1
                job.progress("cells", processed, total, skipped=done)
        finally:
            del adapter
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        if sweep.evict_checkpoint_cache:
            evict(config.model.cache_dir, [revisions[checkpoint]], job)
    job.resume_command = sweep_resume
    job.event("sweep_summary", cells=total, completed_now=processed, previously_completed=done, skipped=len(skipped),
              failed=len(failures), manifest=str(manifest_path.resolve()))
    job.artifact(manifest_path)
    if failures:
        listed = "; ".join(f"{f['checkpoint']}/{f['task']}/{f['replacement']}: {f['error']}" for f in failures[:5])
        raise RuntimeError(f"{len(failures)} sweep cell(s) failed (see sweep.json); others completed. {listed}")

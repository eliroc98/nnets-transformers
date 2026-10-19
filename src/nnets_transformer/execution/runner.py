"""Fresh serial experiment execution; no legacy imports or result dependencies."""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from .artifacts import Artifacts, digest
from nnets_transformer.observability import atomic_json
from nnets_transformer.records import Selection


def run(config, resolved_yaml, job, *, adapter=None):
    output = Path(config.output_directory)
    if output.exists() and not config.execution.resume:
        raise FileExistsError(output)
    if adapter is None:
        from nnets_transformer.models import load_adapter
        adapter = load_adapter(config.model, config.resources, job)
    from nnets_transformer.tasks import get_task
    from nnets_transformer.interventions.replacements import get_replacement
    from nnets_transformer.metrics.behavioral import score
    from nnets_transformer.metrics.activation import compare, ACTIVATION_VERSION
    job.stage("dataset_preparation", task=config.task.name)
    examples, dataset = get_task(config.task.name).prepare(adapter, config.task)
    observations = adapter.selections(config.scope.resolution)
    sources = [Selection(s.hook, s.head) for s in config.scope.components] if config.scope.components else observations
    if len({s.id for s in sources}) != len(sources):
        raise ValueError("Duplicate intervention selections")
    for selection in sources:
        adapter.validate_selection(selection)
    from nnets_transformer.interventions.references import donors
    from .artifacts import fingerprint
    dataset["donor_manifest"] = donors(examples, config.replacement)
    excluded = [e.id for e in examples if dataset["donor_manifest"].get(e.id, {}).get("example_id", e.id) is None]
    if excluded:
        if len(excluded) == len(examples):
            raise ValueError("No evaluation example has a length/mask-aligned non-self donor")
        examples = [e for e in examples if e.id not in excluded]
        job.event("donor_exclusions", count=len(excluded), example_ids=excluded)
        # Recorded only when present, so unaffected conditions keep identical evaluation data.
        dataset["excluded_examples"] = [{"example_id": e, "reason": "no length/mask-aligned non-self donor"} for e in excluded]
        dataset["evaluated_example_count"] = len(examples)
    from nnets_transformer.interventions.references import convention
    dataset["replacement_convention"] = convention(config.replacement)
    dataset.pop("dataset_sha256")
    dataset["dataset_sha256"] = fingerprint(dataset)
    inventory = {"version": adapter.provenance.get("inventory_version"),
                 "components": [asdict(c) for c in adapter.inventory],
                 "observations": [asdict(s) for s in observations], "interventions": [asdict(s) for s in sources]}
    work = [(f"e{e:06d}-i{i:06d}", example, selection) for e, example in enumerate(examples) for i, selection in enumerate(sources)]
    # One committed shard per example chunk, holding every intervention for those examples.
    size = config.execution.examples_per_shard
    if config.execution.backend == "optimized" and size % config.execution.example_batch_size:
        raise ValueError("examples_per_shard must be a multiple of example_batch_size for exact resume")
    planned = {}
    for e in range(0, len(examples), size):
        planned[f"e{e:06d}-e{min(e+size, len(examples))-1:06d}"] = [key for key, _, _ in work[e*len(sources):(e+size)*len(sources)]]
    chunk_of = {key: chunk for chunk, keys in planned.items() for key in keys}
    scientific_config = asdict(config)
    for name in ("output_directory", "logging", "input_matrix", "export", "sweep"):
        scientific_config.pop(name, None)
    scientific_config["execution"].pop("resume")
    scientific_config["model"].pop("cache_dir")
    package = Path(__file__).parents[1]
    source_files = [package / "records.py", package / "config.py"]
    for folder in ("models", "tasks", "interventions", "metrics", "execution"):
        source_files.extend((package / folder).rglob("*.py"))
    identity = {"configuration": scientific_config, "model": adapter.provenance,
        "dataset_sha256": dataset["dataset_sha256"], "inventory": inventory,
        "code_hashes": {str(p.relative_to(package)): digest(p) for p in sorted(source_files)},
        "python": sys.version, "dependencies": {name: importlib.metadata.version(name) for name in
            ("torch", "transformer-lens", "transformers")}, "activation_metric_version": ACTIVATION_VERSION,
        "scoring_version": dataset["scoring_version"], "estimator": "exact", "execution_backend": config.execution.backend}
    store = Artifacts(output, identity, resolved_yaml, dataset, inventory, planned,
                      job, resume=config.execution.resume)
    import shlex
    job.resume_command = shlex.join(["poetry", "run", "nnets-transformer", "--config-dir", str(output.resolve()),
                                    "--config-name", "resolved", "execution.resume=true",
                                    f"output_directory={output.resolve()}"])
    reused = sum(len(planned[key]) for key in store.manifest["shards"])
    completed = 0
    processed_examples = set()
    last_update = time.monotonic()
    job.stage("interventions", replacement=config.replacement.name, checkpoint=config.model.checkpoint)
    job.progress("interventions", 0, len(work), skipped=reused)
    try:
        pending = {key for chunk, keys in planned.items() if chunk not in store.manifest["shards"] for key in keys}
        before_forwards = getattr(adapter, "forward_count", 0)
        from nnets_transformer.interventions.references import References
        references = References(adapter, examples, dataset, sources, config.replacement, job) if pending else None
        if config.execution.backend == "optimized":
            from .batching import records
            generated = records(work, pending, adapter, observations, None, dataset, config.execution, references=references)
        else:
            generated = serial_records(work, pending, adapter, observations, None, dataset, references=references)
        buffers = {}
        for key, example, record in generated:
            chunk = chunk_of[key]
            buffers.setdefault(chunk, []).append({"work_key": key, **record})
            if len(buffers[chunk]) == len(planned[chunk]):
                order = {k: i for i, k in enumerate(planned[chunk])}
                store.commit(chunk, sorted(buffers.pop(chunk), key=lambda r: order[r["work_key"]]))
            completed += 1
            processed_examples.add(example.id)
            if time.monotonic()-last_update >= config.logging.progress_interval_seconds or completed+reused == len(work):
                job.progress("interventions", completed, len(work), skipped=reused)
                last_update = time.monotonic()
        if buffers:
            raise RuntimeError(f"Incomplete shard chunks after execution: {sorted(buffers)}")
        job.stage("export")
        export(store, sources, observations)
        store.finish("completed")
        job.progress("examples", len(processed_examples), len(examples), skipped=len(examples)-len(processed_examples))
        job.event("scientific_summary", metric=dataset["scoring_version"], activation_metric=ACTIVATION_VERSION,
                  completed_interventions=completed, reused_interventions=reused,
                  forward_count=getattr(adapter, "forward_count", 0)-before_forwards, sample_count=len(examples))
    except BaseException as exc:
        store.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    finally:
        store.close()


def serial_records(work, pending, adapter, observations, replacement, dataset, references=None):
    from nnets_transformer.metrics.behavioral import score
    from nnets_transformer.metrics.activation import compare
    for key, example, selection in work:
        if key not in pending:
            continue
        if references is not None:
            from nnets_transformer.interventions.references import behavior
            recipients, clean_cache, clean_scores, corrupt_scores, replacements = references.batch([example], observations, len(example.tokens))
            patched_logits, patched_cache = adapter.forward(recipients[0], observations, (selection, replacements[0]))
            patched_score = float(score(patched_logits, example.scoring).cpu())
            record = {"schema_version": "fresh-observation-v1", "example_id": example.id,
                "intervention": asdict(selection), "intervention_id": selection.id,
                "behavior": behavior(example, dataset, clean_scores[0], corrupt_scores[0], patched_score, references.config),
                "activations": [{"observation_id": target.id, **compare(clean_cache[target.hook],
                    patched_cache[target.hook], target, adapter.components[target.hook], example.attention_mask)} for target in observations]}
        else:
            clean_logits, clean_cache = adapter.forward(example, observations)
            patched_logits, patched_cache = adapter.forward(example, observations, (selection, replacement))
            clean_score = float(score(clean_logits, example.scoring).cpu())
            patched_score = float(score(patched_logits, example.scoring).cpu())
            record = {"schema_version": "fresh-observation-v1", "example_id": example.id,
                "intervention": asdict(selection), "intervention_id": selection.id,
                "behavior": {"metric": example.scoring.metric, "units": dataset["units"],
                    "clean_score": clean_score, "patched_score": patched_score,
                    "signed_change": patched_score-clean_score},
                "activations": [{"observation_id": target.id, **compare(clean_cache[target.hook],
                    patched_cache[target.hook], target, adapter.components[target.hook], example.attention_mask)}
                    for target in observations]}
        yield key, example, record


def export(store, sources, observations):
    """Similarity means and behavioral summaries; shards preserve every per-example observation."""
    import csv
    import os
    import numpy as np
    meta, values = store.arrays()
    rows = {source.id: [] for source in sources}
    for index, record in enumerate(meta):
        rows[record["intervention_id"]].append(index)
    behavior = [{"example_id": r["example_id"], "intervention_id": r["intervention_id"], **r["behavior"]} for r in meta]
    temporary = store.root / "similarities.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["intervention", *(target.id for target in observations)])
        for source in sources:
            # FP64 mean over examples of stored FP32 per-example cosine similarities.
            writer.writerow([source.id, *values[rows[source.id], :, 0].astype(np.float64).mean(0).tolist()])
    os.replace(temporary, store.root / "similarities.csv")
    summaries = {}
    epsilon = store.manifest["identity"]["configuration"]["replacement"]["recovery_epsilon"]
    for source in sources:
        selected = [behavior[i] for i in rows[source.id]]
        if selected and all(row.get("corrupt_score") is not None for row in selected):
            clean = sum(r["clean_score"] for r in selected)/len(selected)
            corrupt = sum(r["corrupt_score"] for r in selected)/len(selected)
            patched = sum(r["patched_score"] for r in selected)/len(selected)
            gap = clean-corrupt
            summaries[source.id] = {"clean_score": clean, "corrupt_score": corrupt, "patched_score": patched,
                "recovery": (patched-corrupt)/gap if abs(gap)>epsilon else None,
                "signed_noising": (patched-clean)/gap if abs(gap)>epsilon else None,
                "status": "valid" if abs(gap)>epsilon else "undefined_small_gap", "normalization": "dataset-means"}
    atomic_json(store.root / "behavior.json", {"summaries": summaries,"aggregation": "per_example_per_intervention",
                "comparison_to_historical_results": False, "measurements": behavior})

"""Logged configuration inspection, bounded reference execution and CPU analysis."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from nnets_transformer.observability import Job, Interrupted, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--job-id", default="configured-job")
    parser.add_argument("--log-root")
    parser.add_argument("--heartbeat-seconds", type=float)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    try:
        with Job(args.job_id, root=args.log_root or "logs", heartbeat_seconds=args.heartbeat_seconds if args.heartbeat_seconds is not None else 10.) as job:
            job.stage("configuration")
            from .config import load
            from omegaconf import OmegaConf
            config, resolved = load(args.config_dir, args.config_name, args.overrides)
            if config.command == "run" and "resolve" in {config.model.revision, config.model.tokenizer_revision}:
                # Record the immutable checkpoint commit before anything is saved.
                from .models.checkpoints import resolve_revisions
                commit = resolve_revisions(config.model.id, [config.model.checkpoint])[config.model.checkpoint]
                job.event("revision_resolved", model=config.model.id, checkpoint=config.model.checkpoint, revision=commit)
                for name in ("revision", "tokenizer_revision"):
                    if getattr(config.model, name) == "resolve":
                        setattr(config.model, name, commit)
                        resolved.model[name] = commit
            job.configure(console_level=config.logging.console_level, file_level=config.logging.file_level,
                          heartbeat_seconds=args.heartbeat_seconds if args.heartbeat_seconds is not None else config.logging.heartbeat_interval_seconds,
                          timezone=config.logging.timezone, root=args.log_root or config.logging.root,
                          rotation_bytes=config.logging.rotation_bytes, telemetry_seconds=config.logging.telemetry_interval_seconds)
            yaml = OmegaConf.to_yaml(resolved, resolve=True)
            config_path = job.directory / "resolved.yaml"
            config_path.write_text(yaml)
            job.artifact_root = str(Path(config.output_directory).resolve())
            job.event("resolved_configuration", message=f"Resolved config: {config_path}; artifact root: {job.artifact_root}", config_path=str(config_path), config_sha256=hashlib.sha256(yaml.encode()).hexdigest(), artifact_root=job.artifact_root)
            invocation_path = job.directory / "invocation.json"
            invocation = json.loads(invocation_path.read_text())
            invocation.update(artifact_root=job.artifact_root, resolved_config=str(config_path),
                              config_sha256=hashlib.sha256(yaml.encode()).hexdigest())
            atomic_json(invocation_path, invocation)
            job.artifact(config_path)
            if config.command == "inspect":
                print(yaml)
            elif config.command == "run":
                from .execution.runner import run
                run(config, yaml, job)
            elif config.command == "reference":
                from .execution.historical import run
                run(SimpleNamespace(checkpoint=config.model.checkpoint, revision=config.model.revision,
                    output=Path(config.output_directory), device=config.resources.device,
                    historical_root=Path(config.task.historical_root), cache_dir=Path(config.model.cache_dir),
                    disable_optional_vision=config.resources.disable_optional_vision), job, resolved_yaml=yaml)
            elif config.command == "export":
                from .analysis.export import normalize
                normalize(config.export.inputs, config.output_directory, config.graph, job, config.export.thresholds, config.export.quantile)
            elif config.command == "analyze":
                analyze(config, yaml, job)
            elif config.command == "sweep":
                import shlex
                # Cells read task/replacement YAMLs from the original config directory; changed
                # scientific settings are rejected by each cell's identity check on resume.
                overrides = [o for o in args.overrides if not o.startswith("execution.resume=")]
                job.resume_command = shlex.join(["poetry", "run", "nnets-transformer", "--config-dir", str(args.config_dir.resolve()),
                    "--config-name", args.config_name, "--job-id", args.job_id, *overrides, "execution.resume=true"])
                from .execution.sweep import run_sweep
                run_sweep(config, args.config_dir, job)
    except Interrupted as exc:
        raise SystemExit(128+exc.signum)


def analyze(config, yaml, job):
    # Only CPU artifact dependencies; importing this CLI never imports models.
    import pandas as pd
    import networkx as nx
    from .analysis.graphs import Component, build_graph, centralities
    job.stage("analysis", graph_version=config.graph.version)
    source = Path(config.input_matrix)
    from .analysis.artifacts import read_fresh
    fresh = read_fresh(source)
    frame = pd.read_csv(source, index_col=0)
    inventory = fresh["inventory"] if fresh else [Component.parse(n, config.model.n_layers)
                for n in dict.fromkeys([*frame.index, *frame.columns])]
    graph = build_graph(frame, inventory=inventory, threshold=config.graph.similarity_threshold,
                        version=config.graph.version, architecture=fresh["architecture"] if fresh else config.model.architecture,
                        resolution=config.graph.resolution, epsilon=config.graph.epsilon)
    metrics = centralities(graph)
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved.yaml").write_text(yaml)
    atomic_json(output / "graph.json", nx.node_link_data(graph, edges="edges"))
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "manifest.json", {
        "schema_version": "graph-analysis-v1", "state": "completed", "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "graph": graph.graph,
        "inventory_provenance": fresh["inventory_provenance"] if fresh else "union of unmanifested source/observation labels; original inventory unknown",
        "dataset_sha256": fresh["dataset_sha256"] if fresh else None,
        "aggregation": fresh["aggregation"] if fresh else "unknown input aggregation",
        "source_run_identity": fresh["manifest"]["identity_sha256"] if fresh else None,
        "artifacts": {"graph": "graph.json", "metrics": "metrics.json", "resolved_config": "resolved.yaml"},
        "num_nodes": graph.number_of_nodes(), "num_edges": graph.number_of_edges(),
        "log_directory": str(job.directory), "scoring_version": fresh["scoring_version"] if fresh else None,
        "resolved_config": str((output / "resolved.yaml").resolve()),
    })
    job.progress("graphs", 1, 1)
    job.artifact(output / "manifest.json")


if __name__ == "__main__":
    main()

"""Normalize fresh/legacy-compatible measurements and analyses for CPU readers."""
from pathlib import Path
import hashlib
import json
import os
import pandas as pd
import networkx as nx
from .artifacts import read_fresh
from .graphs import build_graph, centralities
from .longitudinal import role_dynamics
from nnets_transformer.observability import atomic_json

SCHEMA = "normalized-experiment-v1"


def parquet(path, frame):
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def normalize(roots, output, graph_config, job, thresholds=None, quantile=.95):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not roots:
        raise ValueError("At least one input run is required")
    output.mkdir(parents=True)
    summaries, components, edges, behaviors, activations, inventories, sources = [], [], [], [], [], [], []
    seen = set()
    for number, root in enumerate(map(Path, roots), 1):
        job.stage("normalize", source=str(root))
        source = root / "similarities.csv"
        fresh = read_fresh(source)
        if fresh is None:
            raise ValueError("normalize requires a verified fresh run; use import-legacy for trusted pickle artifacts")
        manifest = fresh["manifest"]
        run_id = manifest["identity_sha256"]
        if run_id in seen:
            raise ValueError("Duplicate scientific run identity")
        seen.add(run_id)
        identity = manifest["identity"]
        model = identity["model"]
        configuration = identity["configuration"]
        common = {"run_id": run_id, "model_id": model.get("model_id", configuration["model"]["id"]),
            "model_revision": model["model_revision"], "tokenizer_revision": model["tokenizer_revision"],
            "checkpoint": model.get("checkpoint", configuration["model"]["checkpoint"]),
            "step": int(model.get("step", configuration["model"]["checkpoint"].removeprefix("step"))),
            "stage": model.get("stage"), "tokens_seen": model.get("tokens_seen"),
            "training_seed": model.get("training_seed"), "task": configuration["task"]["name"],
            "replacement": configuration["replacement"]["name"], "scoring_version": identity["scoring_version"],
            "dataset_sha256": fresh["dataset_sha256"], "dtype": model.get("dtype", "float32")}
        # Training runs/checkpoints share a trajectory only when condition keys
        # agree; checkpoint revision itself is deliberately omitted.
        trajectory_fields = {key: value for key, value in common.items() if key not in
            {"run_id", "model_revision", "checkpoint", "step", "tokens_seen"}}
        common["trajectory_id"] = hashlib.sha256(json.dumps(trajectory_fields, sort_keys=True).encode()).hexdigest()
        sample_scores = {}
        from nnets_transformer.execution.artifacts import iter_records
        for record in iter_records(root, manifest):
            behavioral = common | {"example_id": record["example_id"], "intervention_id": record["intervention_id"], **record["behavior"]}
            behaviors.append(behavioral)
            sample_scores[record["example_id"]] = record["behavior"]["clean_score"]
            for observation in record["activations"]:
                activations.append(common | {"example_id": record["example_id"], "source": record["intervention_id"],
                                             "target": observation["observation_id"], **observation})
        frame = pd.read_csv(source, index_col=0)
        for component in fresh["inventory"]:
            inventories.append(common | {"component": component.id, "layer": component.layer,
                                          "head": component.head, "site": component.site})
        for threshold in thresholds or [graph_config.similarity_threshold]:
            graph = build_graph(frame, inventory=fresh["inventory"], threshold=threshold, version=graph_config.version,
                architecture=fresh["architecture"], resolution=graph_config.resolution, epsilon=graph_config.epsilon)
            key = common | {"threshold": threshold, "resolution": graph_config.resolution, "graph_version": graph_config.version,
                "metric_version": graph.graph["metric_version"], "measurement_status": graph.graph["measurement_status"],
                "measurement_coverage": graph.graph["measurement_coverage"]}
            summaries.append(key | {"num_nodes": len(graph), "active_nodes": graph.graph["active_nodes"],
                "num_edges": graph.number_of_edges(), "density": nx.density(graph),
                "admissible_density": graph.graph["admissible_density"],
                "clean_score": sum(sample_scores.values())/len(sample_scores) if sample_scores else None,
                "sample_count": len(sample_scores)})
            for metric, values in centralities(graph).items():
                for component, value in values.items():
                    node = graph.nodes[component]
                    components.append(key | {"component": component, "layer": node.get("layer"), "head": node.get("head"),
                        "metric": metric, "value": value, "units": "strength" if "strength" in metric else "dimensionless"})
            for a, b, values in graph.edges(data=True):
                edges.append(key | {"source": a, "target": b, **values})
        sources.append({"path": str(root.resolve()), "run_id": run_id, "manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                        "attempts": manifest["attempts"]})
        job.progress("runs", number, len(roots))
    tables = {"checkpoints": pd.DataFrame(summaries), "components": pd.DataFrame(components),
              "edges": pd.DataFrame(edges, columns=list(edges[0]) if edges else ["run_id", "source", "target", "threshold", "graph_version", "resolution", "strength", "distance"]),
              "behavior": pd.DataFrame(behaviors), "activations": pd.DataFrame(activations), "inventory": pd.DataFrame(inventories)}
    # Role continuity is across checkpoints within a condition, not arbitrary
    # independent run IDs. Restore the individual run IDs after calculation.
    role_input = tables["components"].copy()
    role_input["run_id"] = role_input.trajectory_id
    tables["roles"] = role_dynamics(role_input, quantile)
    for name, table in tables.items():
        parquet(output / f"{name}.parquet", table)
        job.artifact(output / f"{name}.parquet")
    from .figures import report
    report(tables, output / "report.html")
    atomic_json(output / "manifest.json", {"schema_version": SCHEMA, "state": "completed", "sources": sources,
        "tables": {name: f"{name}.parquet" for name in tables}, "roles": {"quantile": quantile, "ties": "strict_above_linear_quantile"},
        "graph": {"version": graph_config.version, "resolution": graph_config.resolution, "thresholds": thresholds or [graph_config.similarity_threshold],
                  "distance": "inverse_strength", "epsilon": graph_config.epsilon},
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        "log_directory": str(job.directory)})
    job.artifact(output / "manifest.json")
    return output

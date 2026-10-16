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


KEYS = ["run_id", "trajectory_id", "checkpoint", "step", "tokens_seen"]
GRAPH_KEYS = ["threshold", "resolution", "graph_version"]


def behavior_summary(rows, epsilon):
    """Per-intervention dataset means; recovery uses mean scores (undefined for a negligible gap)."""
    frame = pd.DataFrame(rows)
    out = []
    for intervention, group in frame.groupby("intervention_id", sort=False):
        clean, patched = group.clean_score.mean(), group.patched_score.mean()
        corrupt = group.corrupt_score.mean() if "corrupt_score" in group and group.corrupt_score.notna().all() else None
        gap = None if corrupt is None else clean-corrupt
        valid = gap is not None and abs(gap) > epsilon
        out.append({"intervention_id": intervention, "metric": group.metric.iloc[0], "units": group.units.iloc[0],
                    "direction": group.direction.iloc[0] if "direction" in group else None, "sample_count": len(group),
                    "clean_score": clean, "corrupt_score": corrupt, "patched_score": patched,
                    "signed_change": group.signed_change.mean(), "recovery": (patched-corrupt)/gap if valid else None,
                    "recovery_status": "valid" if valid else ("undefined_small_gap" if gap is not None else "unpaired"),
                    "normalization": "dataset-means"})
    return out


def normalize(roots, output, graph_config, job, thresholds=None, quantile=.95, include_activations=True,
              behavior_rows=True, slim=False):
    """Normalized tables for CPU readers.

    include_activations: per-example activation rows (examples x sources x targets; large).
    behavior_rows: per-example behavior rows; otherwise per-intervention summaries (per-example
      rows stay in each run). slim: key columns only in component/edge/inventory tables (the
      checkpoint table keeps full provenance), and edges stored once at the highest threshold
      with their similarity, since lower-threshold edge sets are exact subsets.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not roots:
        raise ValueError("At least one input run is required")
    output.mkdir(parents=True)
    thresholds = sorted(thresholds or [graph_config.similarity_threshold])
    frames = {name: [] for name in ("checkpoints", "components", "edges", "behavior", "activations", "inventory")}
    sources = []
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
        # Checkpoints share a trajectory only when the condition agrees. Per-checkpoint
        # labels (model/tokenizer commit, and the dataset hash that embeds the tokenizer
        # commit) are replaced by content identities: tokenizer files and evaluation data.
        from nnets_transformer.execution.artifacts import load_dataset, iter_records
        dataset = load_dataset(root)
        convention = dataset.get("replacement_convention") or {"replacement": common["replacement"]}
        common["replacement_condition"] = "/".join(str(convention[k]) for k in
            ("replacement", "direction", "donor_policy", "mean") if convention.get(k) is not None)
        from nnets_transformer.execution.artifacts import fingerprint
        common["evaluation_sha256"] = fingerprint({k: v for k, v in dataset.items() if k not in {"tokenizer_revision", "dataset_sha256"}})
        files = model.get("tokenizer_files")
        common["tokenizer_sha256"] = fingerprint(files) if files else model["tokenizer_revision"]
        trajectory_fields = {key: value for key, value in common.items() if key not in
            {"run_id", "model_revision", "tokenizer_revision", "dataset_sha256", "checkpoint", "step", "tokens_seen"}}
        common["trajectory_id"] = hashlib.sha256(json.dumps(trajectory_fields, sort_keys=True).encode()).hexdigest()
        summaries, components, edges, behaviors, activations, inventories = [], [], [], [], [], []
        keys = {k: common[k] for k in KEYS} if slim else common
        sample_scores = {}
        if manifest.get("compaction"):
            # Per-record activations were removed; behavior rows are kept in Parquet.
            rows = pd.read_parquet(root / "behavior_measurements.parquet").to_dict("records")
            for row in rows:
                behaviors.append(row)
                sample_scores[row["example_id"]] = row["clean_score"]
            records = ()
        else:
            records = iter_records(root, manifest)
        for record in records:
            behaviors.append({"example_id": record["example_id"], "intervention_id": record["intervention_id"], **record["behavior"]})
            sample_scores[record["example_id"]] = record["behavior"]["clean_score"]
            for observation in (record["activations"] if include_activations else ()):
                activations.append(keys | {"example_id": record["example_id"], "source": record["intervention_id"],
                                             "target": observation["observation_id"], **observation})
        frame = pd.read_csv(source, index_col=0)
        for component in fresh["inventory"]:
            inventories.append(keys | {"component": component.id, "layer": component.layer,
                                          "head": component.head, "site": component.site})
        for threshold in thresholds:
            graph = build_graph(frame, inventory=fresh["inventory"], threshold=threshold, version=graph_config.version,
                architecture=fresh["architecture"], resolution=graph_config.resolution, epsilon=graph_config.epsilon)
            full = common | {"threshold": threshold, "resolution": graph_config.resolution, "graph_version": graph_config.version,
                "metric_version": graph.graph["metric_version"], "measurement_status": graph.graph["measurement_status"],
                "measurement_coverage": graph.graph["measurement_coverage"]}
            key = keys | {k: full[k] for k in GRAPH_KEYS} if slim else full
            summaries.append(full | {"num_nodes": len(graph), "active_nodes": graph.graph["active_nodes"],
                "num_edges": graph.number_of_edges(), "density": nx.density(graph),
                "admissible_density": graph.graph["admissible_density"],
                "clean_score": sum(sample_scores.values())/len(sample_scores) if sample_scores else None,
                "sample_count": len(sample_scores)})
            for metric, values in centralities(graph).items():
                for component, value in values.items():
                    node = graph.nodes[component]
                    components.append(key | {"component": component, "layer": node.get("layer"), "head": node.get("head"),
                        "metric": metric, "value": value, "units": "strength" if "strength" in metric else "dimensionless"})
            if not slim or threshold == thresholds[-1]:
                for a, b, values in graph.edges(data=True):
                    edges.append(key | {"source": a, "target": b, **values})
        epsilon = configuration["replacement"].get("recovery_epsilon", 1e-6)
        behaviors = [keys | row for row in (behaviors if behavior_rows else behavior_summary(behaviors, epsilon))]
        for name, rows in (("checkpoints", summaries), ("components", components), ("edges", edges),
                           ("behavior", behaviors), ("activations", activations), ("inventory", inventories)):
            if rows:
                frames[name].append(pd.DataFrame(rows))
        sources.append({"path": str(root.resolve()), "run_id": run_id, "manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                        "attempts": manifest["attempts"]})
        job.progress("runs", number, len(roots))
    empty = {"edges": ["run_id", "trajectory_id", "step", "source", "target", "threshold", "graph_version", "resolution",
                       "similarity", "strength", "distance"]}
    tables = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=empty.get(name, []))
              for name, parts in frames.items()}
    del frames
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
        "graph": {"version": graph_config.version, "resolution": graph_config.resolution, "thresholds": thresholds,
                  "distance": "inverse_strength", "epsilon": graph_config.epsilon,
                  **({"edges_threshold": thresholds[-1], "edges_rule": "edges at a lower threshold t are rows with similarity < t"} if slim else {})},
        "layout": {"slim": slim, "behavior": "per-example rows" if behavior_rows else "per-intervention dataset-mean summaries",
                   "activations": include_activations},
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        "log_directory": str(job.directory)})
    job.artifact(output / "manifest.json")
    return output

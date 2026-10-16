"""Explicit import of trusted legacy pickles into the normalized dashboard contract.

Source files are never modified. Legacy semantics are labeled, not repaired:
graph version `legacy-notebook-v1`, strength used as shortest-path distance in
the old closeness (`closeness_legacy`), fixed-passage token-2205 behavior, and no
endpoint identities in `edge_weights.pickle` (so no network view).
"""
import hashlib
import json
import pickle
from pathlib import Path
import re

import pandas as pd

from nnets_transformer.analysis.longitudinal import role_dynamics
from nnets_transformer.observability import atomic_json

METRICS = {"in_strength_dist": "in_strength", "out_strength_dist": "out_strength", "betweenness_dist": "betweenness",
           "closeness_dist": "closeness_legacy", "pagerank_dist": "pagerank"}
LIMITATIONS = [
    "Legacy graph semantics (legacy-notebook-v1): strength transformed once in the notebook; isolated nodes dropped from retained-edge graphs.",
    "closeness_legacy used strength as shortest-path distance; it is not the corrected directed closeness.",
    "Behavior is the historical fixed passage scored on vocabulary token 2205 (legacy-induction-token2205-v1).",
    "edge_weights.pickle has no endpoint identities: network exploration is unavailable for this export.",
    "Active-node counts, admissible-pair density, tokens seen and measurement coverage are unknown.",
]


def parse_component(name):
    """Legacy component names -> (layer, head, is_module_level); head is -1 for module level."""
    if name in ("emb", "pos_emb"):
        return -1, -1, True
    if name == "logits":
        return 90, -1, True
    if name.startswith("mlp"):
        digits = re.findall(r"\d+", name)
        if not digits:
            raise ValueError(f"cannot parse layer from MLP component {name!r}")
        return int(digits[-1]), -1, True
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "attn":
        raise ValueError(f"unrecognised component name {name!r}")
    layer = int(parts[2])
    if len(parts) == 3:
        return layer, -1, True
    return layer, int(parts[3]), False


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_legacy(source, output, job, quantile=.95):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    files = {name: source / name for name in ("all_metrics.pickle", "all_distributions.pickle")}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    job.stage("legacy_import", source=str(source.resolve()))
    hashes = {name: _sha(path) for name, path in files.items()}
    with files["all_metrics.pickle"].open("rb") as handle:
        metrics = pd.DataFrame(pickle.load(handle))
    with files["all_distributions.pickle"].open("rb") as handle:
        distributions = pd.DataFrame(pickle.load(handle))
    trajectory = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    common = {"run_id": trajectory, "trajectory_id": trajectory, "model_id": "EleutherAI/pythia-14m",
              "model_revision": None, "tokenizer_revision": None, "stage": "pretraining", "tokens_seen": None,
              "training_seed": "standard-release", "task": "legacy_induction", "replacement": "zero",
              "scoring_version": "legacy-induction-token2205-v1", "dataset_sha256": None, "dtype": "float32",
              "resolution": "mixed", "graph_version": "legacy-notebook-v1", "metric_version": "legacy-notebook-v1",
              "measurement_status": "legacy-unknown", "measurement_coverage": None}
    checkpoints = metrics.rename(columns={"induction_log": "clean_score", "induction_loss": "legacy_induction_loss"})
    checkpoints = checkpoints.assign(**common, checkpoint=checkpoints.step.map(lambda s: f"step{int(s)}"),
                                     active_nodes=None, admissible_density=None, sample_count=1)
    if checkpoints.duplicated(["step", "threshold"]).any():
        raise ValueError("Legacy metrics contain duplicate step/threshold rows")
    parsed = distributions.component.map(parse_component)
    base = distributions.assign(layer=parsed.map(lambda p: p[0]), head=parsed.map(lambda p: None if p[2] else p[1]))
    components = []
    for column, metric in METRICS.items():
        if column in base:
            part = base[["step", "threshold", "component", "layer", "head", column]].rename(columns={column: "value"})
            components.append(part.assign(metric=metric, units="strength" if "strength" in metric else "dimensionless"))
    components = pd.concat(components, ignore_index=True).assign(**common)
    components["checkpoint"] = components.step.map(lambda s: f"step{int(s)}")
    inventory = base.drop_duplicates("component")[["component", "layer", "head"]].assign(**common, site=None)
    role_input = components.copy()
    tables = {"checkpoints": checkpoints, "components": components, "inventory": inventory,
              "roles": role_dynamics(role_input, quantile),
              "edges": pd.DataFrame(columns=["trajectory_id", "step", "threshold", "source", "target", "strength", "distance"])}
    output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
        job.artifact(output / f"{name}.parquet")
    atomic_json(output / "manifest.json", {"schema_version": "normalized-experiment-v1", "state": "completed", "legacy": True,
        "limitations": LIMITATIONS, "sources": [{"path": str(path.resolve()), "sha256": hashes[name], "run_id": trajectory}
                                                for name, path in files.items()],
        "tables": {name: f"{name}.parquet" for name in tables},
        "roles": {"quantile": quantile, "ties": "strict_above_linear_quantile"},
        "graph": {"version": "legacy-notebook-v1", "resolution": "mixed", "thresholds": sorted(checkpoints.threshold.unique().tolist()),
                  "distance": "legacy: strength used as distance"},
        "files": {p.name: _sha(p) for p in output.iterdir() if p.is_file() and p.suffix == ".parquet"},
        "log_directory": str(job.directory)})
    job.artifact(output / "manifest.json")
    job.progress("legacy_tables", len(tables), len(tables))
    return output

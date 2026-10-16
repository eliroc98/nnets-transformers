"""Validated, serializable dashboard view state (saved views and shareable URLs)."""
from dataclasses import asdict, dataclass, field, fields
import json
from urllib.parse import parse_qs, urlencode

VIEW_SCHEMA = "dashboard-view-v1"
URL_FIELDS = ("export_id", "trajectory_id", "threshold", "metric", "step", "axis", "log_x")


@dataclass
class ViewState:
    export_id: str | None = None
    trajectory_id: str | None = None
    threshold: float | None = None
    metric: str | None = None
    step: int | None = None
    pinned: list = field(default_factory=list)
    axis: str = "step"
    log_x: bool = True
    quantile: float = .95
    compare: list = field(default_factory=list)
    compare_column: str = "density"
    compare_step: int | None = None
    network_min_strength: float | None = None
    network_max_edges: int = 300
    network_neighborhood: bool = False
    selected_component: str | None = None

    def to_json(self, provenance=None):
        return json.dumps({"schema_version": VIEW_SCHEMA, "state": asdict(self), "provenance": provenance or {}},
                          indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version", VIEW_SCHEMA) != VIEW_SCHEMA:
            raise ValueError("Unsupported saved-view schema")
        raw = data.get("state", data)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown view fields: {unknown}")
        state = cls(**raw)
        if state.axis not in {"step", "tokens"}:
            raise ValueError("axis must be step or tokens")
        if not 0 <= state.quantile <= 1 or state.network_max_edges < 1:
            raise ValueError("Invalid quantile or network edge limit")
        return state

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def query(self):
        values = {k: getattr(self, k) for k in URL_FIELDS if getattr(self, k) is not None}
        values["log_x"] = int(self.log_x)
        return "?" + urlencode(values)

    @classmethod
    def from_query(cls, search):
        raw = {k: v[-1] for k, v in parse_qs((search or "").lstrip("?")).items() if k in URL_FIELDS}
        converted = {}
        for key, value in raw.items():
            if key == "threshold":
                converted[key] = float(value)
            elif key == "step":
                converted[key] = int(value)
            elif key == "log_x":
                converted[key] = value not in {"0", "false", "False"}
            else:
                converted[key] = value
        return cls.from_dict({"state": converted})


def reconcile(state, repository):
    """Keep requested selections that exist in the catalog; explain everything dropped."""
    notes = []
    catalog = repository.catalog
    if state.export_id not in catalog:
        if state.export_id is not None:
            notes.append(f"Export {state.export_id} is unavailable; showing the first available export.")
        state.export_id = next(iter(catalog), None)
    if state.export_id is None:
        return state, notes + ["No completed normalized exports were found."]
    trajectories = list(repository.trajectories(state.export_id).trajectory_id)
    if state.trajectory_id not in trajectories:
        if state.trajectory_id is not None:
            notes.append("Requested trajectory is unavailable in this export.")
        state.trajectory_id = trajectories[0] if trajectories else None
    if state.trajectory_id is None:
        return state, notes
    thresholds = repository.thresholds(state.export_id, state.trajectory_id)
    if state.threshold not in thresholds:
        if state.threshold is not None:
            notes.append(f"Threshold {state.threshold} is unavailable; precomputed thresholds: {thresholds}.")
        state.threshold = 0.7 if 0.7 in thresholds else thresholds[-1]
    metrics = repository.metrics(state.export_id, state.trajectory_id)
    if state.metric not in metrics:
        if state.metric is not None:
            notes.append(f"Metric {state.metric} is unavailable.")
        state.metric = "in_strength" if "in_strength" in metrics else (metrics[0] if metrics else None)
    steps = repository.steps(state.export_id, state.trajectory_id)
    if state.step not in steps:
        if state.step is not None:
            notes.append(f"Checkpoint step {state.step} was not recorded; selecting the last recorded checkpoint.")
        state.step = steps[-1] if steps else None
    if state.compare_step is not None and state.compare_step not in steps:
        notes.append(f"Comparison step {state.compare_step} was not recorded.")
        state.compare_step = None
    components = set(repository.select(state.export_id, "components", state.trajectory_id).component.unique())
    dropped = [c for c in state.pinned if c not in components]
    if dropped:
        notes.append(f"Pinned components not in this trajectory: {dropped}")
    state.pinned = [c for c in state.pinned if c in components]
    others = set(trajectories)
    state.compare = [t for t in state.compare if t in others and t != state.trajectory_id]
    if state.axis == "tokens":
        rows = repository.select(state.export_id, "checkpoints", state.trajectory_id)
        if "tokens_seen" not in rows or rows.tokens_seen.isna().any():
            notes.append("Tokens seen are unknown for this trajectory; using training steps.")
            state.axis = "step"
    return state, notes

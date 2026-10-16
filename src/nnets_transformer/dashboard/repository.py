"""Read-only, CPU-only queries over normalized experiment exports.

An export is a directory written by `analysis.export.normalize` (or the explicit
legacy importer) with a completed `normalized-experiment-v1` manifest. Files are
checksum-verified before use; results are cached per manifest hash, so a new
artifact version is never served from a stale cache entry.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import pandas as pd

SCHEMA = "normalized-experiment-v1"
TABLES = ("checkpoints", "components", "edges", "behavior", "inventory", "roles")
# Trajectory identity: the condition keys that define one training-progression series.
TRAJECTORY_FIELDS = ("trajectory_id", "model_id", "training_seed", "task", "replacement", "replacement_condition",
                     "scoring_version", "evaluation_sha256", "dtype", "graph_version", "resolution")


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Repository:
    def __init__(self, roots, cache_entries=32):
        if cache_entries < 1:
            raise ValueError("cache_entries must be positive")
        self.roots = [Path(r) for r in roots]
        self.cache_entries = cache_entries
        self._cache = OrderedDict()
        self._verified = {}
        self._lock = threading.Lock()
        self.catalog = {}
        self.problems = []
        self.refresh()

    # -- catalog -----------------------------------------------------------
    def refresh(self):
        """Rescan roots; only completed, checksum-valid exports are browsable."""
        catalog, problems = {}, []
        for root in self.roots:
            if not root.exists():
                problems.append({"path": str(root), "problem": "artifact root does not exist"})
                continue
            candidates = [root] if (root / "manifest.json").is_file() else sorted(p.parent for p in root.glob("*/manifest.json"))
            for path in candidates:
                try:
                    entry = self._entry(path)
                except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                    problems.append({"path": str(path), "problem": str(exc)})
                    continue
                if entry is not None:
                    if entry["id"] in catalog:
                        problems.append({"path": str(path), "problem": f"duplicate export id {entry['id']}"})
                        continue
                    catalog[entry["id"]] = entry
        with self._lock:
            self.catalog, self.problems = catalog, problems
            live = {(e["id"], e["manifest_sha256"]) for e in catalog.values()}
            for key in [k for k in self._cache if k[:2] not in live]:
                del self._cache[key]
        return catalog

    def _entry(self, path):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA:
            return None
        if manifest.get("state") != "completed":
            raise ValueError("export is not completed")
        sha = _sha(manifest_path)
        for name, expected in manifest.get("files", {}).items():
            if Path(name).name != name:
                raise ValueError(f"invalid file name {name!r}")
            file = path / name
            stat = file.stat()
            key = (str(file.resolve()), stat.st_mtime_ns, stat.st_size)
            if self._verified.get(key) != expected:
                if _sha(file) != expected:
                    raise ValueError(f"checksum failed: {name}")
                self._verified[key] = expected
        tables = {name: path / file for name, file in manifest.get("tables", {}).items()}
        missing = [name for name in ("checkpoints", "components") if name not in tables]
        if missing:
            raise ValueError(f"missing required tables: {missing}")
        return {"id": f"{path.name}-{sha[:10]}", "name": path.name, "path": str(path.resolve()), "manifest_sha256": sha,
                "manifest": manifest, "tables": tables, "legacy": bool(manifest.get("legacy", False)),
                "limitations": manifest.get("limitations", [])}

    def entry(self, export_id):
        try:
            return self.catalog[export_id]
        except KeyError:
            raise KeyError(f"Unknown or no longer available export: {export_id}") from None

    # -- tables --------------------------------------------------------------
    def table(self, export_id, name):
        entry = self.entry(export_id)
        key = (export_id, entry["manifest_sha256"], name)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        if name not in entry["tables"]:
            frame = pd.DataFrame()
        else:
            frame = pd.read_parquet(entry["tables"][name])
        with self._lock:
            self._cache[key] = frame
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
        return frame

    def available(self, export_id, name):
        return name in self.entry(export_id)["tables"] and not self.table(export_id, name).empty

    def trajectories(self, export_id):
        checkpoints = self.table(export_id, "checkpoints")
        fields = [f for f in TRAJECTORY_FIELDS if f in checkpoints.columns]
        frame = checkpoints[fields].drop_duplicates("trajectory_id").reset_index(drop=True)
        counts = checkpoints.groupby("trajectory_id").step.nunique()
        frame["checkpoints"] = frame.trajectory_id.map(counts)
        return frame

    def select(self, export_id, name, trajectory_id, **equal):
        """Rows of one trajectory, with optional exact-match filters (threshold, metric, step, ...)."""
        frame = self.table(export_id, name)
        if frame.empty:
            return frame
        key = "run_id" if name == "roles" else "trajectory_id"
        mask = frame[key] == trajectory_id
        for column, value in equal.items():
            if value is None:
                continue
            mask &= frame[column] == value
        return frame[mask]

    def steps(self, export_id, trajectory_id):
        rows = self.select(export_id, "checkpoints", trajectory_id)
        return sorted(int(s) for s in rows.step.unique())

    def thresholds(self, export_id, trajectory_id):
        rows = self.select(export_id, "checkpoints", trajectory_id)
        return sorted(float(t) for t in rows.threshold.unique())

    def metrics(self, export_id, trajectory_id):
        rows = self.select(export_id, "components", trajectory_id)
        return sorted(rows.metric.unique()) if not rows.empty else []

    def cache_size(self):
        return len(self._cache)

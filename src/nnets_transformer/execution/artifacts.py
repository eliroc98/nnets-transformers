"""Atomic chunked measurement shards and verified, exclusive resumption.

Schema v2 commits one deterministic zip per example chunk: behavioral records as
JSON plus one float32 activation array [records, observations, 4]. Readers also
accept per-intervention v1 JSON shards from earlier runs.
"""
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
from nnets_transformer.observability import atomic_json

SCHEMA = "fresh-reference-v2"
READABLE = {"fresh-reference-v1", SCHEMA}
FIELDS = ("cosine_similarity", "relative_l2_change", "clean_zero_norm_fraction", "patched_zero_norm_fraction")
_EPOCH = (1980, 1, 1, 0, 0, 0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_chunk(path, records, observation_ids):
    """Deterministic bytes: fixed zip timestamps and member order."""
    values = np.full((len(records), len(observation_ids), len(FIELDS)), np.nan, dtype=np.float32)
    meta = []
    for row, record in enumerate(records):
        if [a["observation_id"] for a in record["activations"]] != observation_ids:
            raise ValueError("Activation observations differ from the saved inventory order")
        for column, activation in enumerate(record["activations"]):
            values[row, column] = [math.nan if activation[f] is None else activation[f] for f in FIELDS]
        meta.append({k: v for k, v in record.items() if k != "activations"})
    array = io.BytesIO()
    np.save(array, values, allow_pickle=False)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in (("records.json", json.dumps(meta, sort_keys=True, allow_nan=False).encode()),
                           ("activations.npy", array.getvalue())):
            info = zipfile.ZipInfo(name, date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_chunk(path):
    with zipfile.ZipFile(path) as bundle:
        meta = json.loads(bundle.read("records.json"))
        values = np.load(io.BytesIO(bundle.read("activations.npy")), allow_pickle=False)
    return meta, values


def expand(meta, values, observation_ids):
    """Reconstruct v1-shaped per-record dictionaries for compatibility readers."""
    for record, rows in zip(meta, values):
        activations = []
        for observation, (cosine, relative, clean_zero, patched_zero) in zip(observation_ids, rows.tolist()):
            valid = not math.isnan(relative)
            activations.append({"observation_id": observation, "cosine_similarity": cosine,
                "relative_l2_change": relative if valid else None,
                "relative_l2_status": "valid" if valid else "undefined_clean_zero_norm",
                "clean_zero_norm_fraction": clean_zero, "patched_zero_norm_fraction": patched_zero})
        yield {**record, "activations": activations}


def iter_records(root, manifest):
    """Verified records of a completed v1 or v2 run, in planned order."""
    root = Path(root)
    if manifest.get("schema_version") not in READABLE:
        raise ValueError("Unknown fresh artifact schema")
    if manifest.get("compaction"):
        raise ValueError("Compacted run: per-record activations were removed; read behavior_measurements.parquet instead")
    observation_ids = None
    for key in manifest["planned_shards"]:
        shard = manifest["shards"][key]
        path = root / shard["path"]
        if digest(path) != shard["sha256"]:
            raise ValueError(f"Corrupt shard: {path}")
        if manifest["schema_version"] == "fresh-reference-v1":
            yield json.loads(path.read_text())
            continue
        if observation_ids is None:
            observation_ids = [o["hook"] if o["head"] is None else f"{o['hook']}.{o['head']}"
                               for o in json.loads((root / "inventory.json").read_text())["observations"]]
        yield from expand(*read_chunk(path), observation_ids)


class Artifacts:
    def __init__(self, root, identity, resolved_yaml, dataset, inventory, planned, job, resume=False):
        """planned maps chunk IDs to their ordered work keys."""
        self.root, self.job = Path(root), job
        self.planned = planned
        self.observation_ids = [o["hook"] if o["head"] is None else f"{o['hook']}.{o['head']}" for o in inventory["observations"]]
        existed = self.root.exists()
        if existed and not resume:
            raise FileExistsError(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / "manifest.json"
            self.identity = fingerprint(identity)
            plan = fingerprint(planned)
            if existed:
                self.manifest = json.loads(path.read_text())
                if self.manifest.get("compaction"):
                    raise ValueError("Run was compacted after completion and is final; it cannot be resumed")
                if self.manifest["schema_version"] != SCHEMA or self.manifest["identity_sha256"] != self.identity:
                    raise ValueError("Resume configuration/model/dataset/code identity differs")
                for name, checksum in self.manifest["inputs"].items():
                    if digest(self.root / name) != checksum:
                        raise ValueError(f"Resume input failed checksum: {name}")
                if self.manifest["planned_shards"] != list(planned) or self.manifest["plan_sha256"] != plan:
                    raise ValueError("Resume work plan differs")
                for key, shard in self.manifest["shards"].items():
                    if key not in planned or shard["path"] != f"shards/{key}.zip":
                        raise ValueError("Unexpected committed shard")
                    if digest(self.root / shard["path"]) != shard["sha256"]:
                        raise ValueError(f"Resume shard failed checksum: {key}")
            else:
                (self.root / "shards").mkdir()
                (self.root / "resolved.yaml").write_text(resolved_yaml)
                atomic_json(self.root / "dataset.json", dataset)
                atomic_json(self.root / "inventory.json", inventory)
                self.manifest = {"schema_version": SCHEMA, "identity_sha256": self.identity,
                    "identity": identity, "planned_shards": list(planned), "plan_sha256": plan,
                    "work_count": sum(map(len, planned.values())), "state": "running", "shards": {},
                    "inputs": {name: digest(self.root / name) for name in ("resolved.yaml", "dataset.json", "inventory.json")},
                    "attempts": [], "artifacts": {"dataset": "dataset.json", "inventory": "inventory.json",
                    "resolved_config": "resolved.yaml", "similarities": "similarities.csv", "behavior": "behavior.json"},
                    "shard_format": {"container": "zip", "records": "records.json", "activations": "activations.npy",
                                     "activation_axes": ["record", "observation", "field"], "fields": list(FIELDS),
                                     "missing": "NaN relative_l2_change marks an undefined clean zero norm"}}
            previous = self.manifest["attempts"][-1] if self.manifest["attempts"] else None
            job.previous_attempt = previous
            self.manifest["attempts"].append(str(job.directory))
            self.manifest["state"] = "running"
            atomic_json(path, self.manifest)
            job.event("resume_validated" if existed else "run_created", identity_sha256=self.identity,
                      reused_shards=len(self.manifest["shards"]), previous_attempt=previous)
        except BaseException:
            self.lock.close()
            raise

    def commit(self, key, records):
        if key in self.manifest["shards"]:
            raise ValueError("Cannot replace a committed shard")
        if [r["work_key"] for r in records] != self.planned[key]:
            raise ValueError("Chunk records differ from the planned work keys")
        path = self.root / "shards" / f"{key}.zip"
        write_chunk(path, records, self.observation_ids)
        self.manifest["shards"][key] = {"path": f"shards/{key}.zip", "sha256": digest(path), "records": len(records)}
        atomic_json(self.root / "manifest.json", self.manifest)
        self.job.artifact(path)

    def finish(self, state):
        self.manifest["state"] = state
        if state == "completed":
            self.manifest["output_hashes"] = {name: digest(self.root / name) for name in ("similarities.csv", "behavior.json")}
        atomic_json(self.root / "manifest.json", self.manifest)
        self.job.artifact(self.root / "manifest.json")

    def arrays(self):
        """Committed records' metadata and stacked activation metrics, in planned order."""
        meta, values = [], []
        for key in self.manifest["planned_shards"]:
            shard = self.manifest["shards"][key]
            if digest(self.root / shard["path"]) != shard["sha256"]:
                raise ValueError(f"Corrupt shard: {key}")
            chunk_meta, chunk_values = read_chunk(self.root / shard["path"])
            meta.extend(chunk_meta)
            values.append(chunk_values)
        return meta, np.concatenate(values) if values else np.zeros((0, len(self.observation_ids), len(FIELDS)), np.float32)

    def records(self):
        meta, values = self.arrays()
        return list(expand(meta, values, self.observation_ids))

    def close(self):
        self.lock.close()


# -- post-completion compaction ------------------------------------------------
COMPACTION = "compacted-v1"
COMPACT_MODES = {"cosine16", "means"}


def load_dataset(root):
    """dataset.json, or its gzip form after compaction."""
    import gzip
    root = Path(root)
    if (root / "dataset.json").is_file():
        return json.loads((root / "dataset.json").read_text())
    return json.loads(gzip.decompress((root / "dataset.json.gz").read_bytes()))


def _deterministic_zip(path, members):
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
    os.replace(temporary, path)


def compact(root, mode, job=None):
    """Shrink a completed run while keeping everything graph/behavior analysis reads.

    Kept: manifest, resolved config, inventory, similarities.csv (mean per-example cosine),
    behavior summaries, every per-example behavioral record (zstd Parquet), the dataset
    manifest (gzip) and, for mode `cosine16`, per-example cosine similarities as float16.
    Removed: the full float32 activation shards (relative L2 and zero-norm fields). The run
    becomes final: it can be analysed and exported, but not resumed.
    """
    import gzip
    import pandas as pd
    if mode not in COMPACT_MODES:
        raise ValueError(f"Compaction mode must be one of {sorted(COMPACT_MODES)}")
    root = Path(root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("compaction"):
        return manifest["compaction"]
    if manifest.get("state") != "completed" or manifest.get("schema_version") != SCHEMA:
        raise ValueError("Only completed fresh-reference-v2 runs can be compacted")
    for name, checksum in {**manifest["inputs"], **manifest["output_hashes"]}.items():
        if digest(root / name) != checksum:
            raise ValueError(f"Refusing to compact: checksum failed for {name}")
    observation_ids = [o["hook"] if o["head"] is None else f"{o['hook']}.{o['head']}"
                       for o in json.loads((root / "inventory.json").read_text())["observations"]]
    meta, values = [], []
    for key in manifest["planned_shards"]:
        shard = manifest["shards"][key]
        if digest(root / shard["path"]) != shard["sha256"]:
            raise ValueError(f"Refusing to compact: corrupt shard {key}")
        chunk_meta, chunk_values = read_chunk(root / shard["path"])
        meta.extend(chunk_meta)
        values.append(chunk_values)
    values = np.concatenate(values)
    behavior = json.loads((root / "behavior.json").read_text())
    if len(behavior["measurements"]) != len(meta):
        raise ValueError("Behavior measurements do not match the committed records")
    frame = pd.DataFrame(behavior["measurements"])
    frame.to_parquet(root / "behavior_measurements.parquet.tmp", compression="zstd", index=False)
    os.replace(root / "behavior_measurements.parquet.tmp", root / "behavior_measurements.parquet")
    behavior["measurements"] = {"path": "behavior_measurements.parquet", "rows": len(frame), "format": "parquet-zstd"}
    atomic_json(root / "behavior.json", behavior)
    data = (root / "dataset.json").read_bytes()
    temporary = root / "dataset.json.gz.tmp"
    temporary.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    os.replace(temporary, root / "dataset.json.gz")
    outputs = {"similarities.csv": manifest["output_hashes"]["similarities.csv"], "behavior.json": digest(root / "behavior.json"),
               "behavior_measurements.parquet": digest(root / "behavior_measurements.parquet")}
    record = {"version": COMPACTION, "mode": mode, "removed_shards": {k: s["sha256"] for k, s in manifest["shards"].items()},
              "removed_fields": ["relative_l2_change", "clean_zero_norm_fraction", "patched_zero_norm_fraction"] +
                                (["cosine_similarity (per example)"] if mode == "means" else []),
              "original_dataset_json_sha256": manifest["inputs"]["dataset.json"],
              "original_behavior_json_sha256": manifest["output_hashes"]["behavior.json"]}
    if mode == "cosine16":
        array = io.BytesIO()
        np.save(array, values[..., 0].astype(np.float16), allow_pickle=False)
        index = [[r["work_key"], r["example_id"], r["intervention_id"]] for r in meta]
        _deterministic_zip(root / "per_example_cosine.zip", [
            ("records.json", json.dumps({"columns": ["work_key", "example_id", "intervention_id"], "rows": index}).encode()),
            ("observations.json", json.dumps(observation_ids).encode()), ("cosine.npy", array.getvalue())])
        outputs["per_example_cosine.zip"] = digest(root / "per_example_cosine.zip")
        record["per_example"] = {"path": "per_example_cosine.zip", "dtype": "float16", "axes": ["record", "observation"],
                                 "max_abs_rounding_error": float(np.abs(values[..., 0] - values[..., 0].astype(np.float16)).max())}
    inputs = {k: v for k, v in manifest["inputs"].items() if k != "dataset.json"}
    inputs["dataset.json.gz"] = digest(root / "dataset.json.gz")
    manifest.update(inputs=inputs, output_hashes=outputs, compaction=record, shards={})
    manifest["artifacts"] = {**manifest["artifacts"], "dataset": "dataset.json.gz",
                             "behavior_measurements": "behavior_measurements.parquet",
                             **({"per_example_cosine": "per_example_cosine.zip"} if mode == "cosine16" else {})}
    atomic_json(path, manifest)
    # Delete only after the compacted manifest is committed.
    for key in record["removed_shards"]:
        (root / "shards" / f"{key}.zip").unlink(missing_ok=True)
    (root / "dataset.json").unlink(missing_ok=True)
    if job is not None:
        job.event("run_compacted", path=str(root), mode=mode, records=len(meta))
    return record

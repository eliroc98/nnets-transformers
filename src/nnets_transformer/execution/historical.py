"""Bounded T01 reproduction; never invokes the legacy checkpoint sweep.

Run with PYTHONPATH=src from the repository root. Outputs must be new paths.
"""
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def historical_text(path):
    # Frozen source fixture survives migration of the old main.py entry point.
    return (Path(__file__).parents[1] / "tasks/fixtures/legacy_passage.txt").read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-optional-vision", action="store_true", help="Disable unrelated Transformers vision imports when extra TorchVision has an incompatible ABI")
    parser.add_argument("--checkpoint", default="step1000")
    parser.add_argument("--revision", required=True, help="Immutable 40-character checkpoint commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--historical-root", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/historical-cache"))
    args = parser.parse_args()
    from nnets_transformer.observability import Job, Interrupted
    try:
        with Job("historical-reference", artifact_root=args.output) as job:
            run(args, job)
    except Interrupted as exc:
        raise SystemExit(128+exc.signum)


def run(args, job, resolved_yaml=None):
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise ValueError("--revision must be an immutable commit SHA")
    args.output.mkdir(parents=True, exist_ok=False)
    if resolved_yaml is not None:
        (args.output / "resolved.yaml").write_text(resolved_yaml)
    print(f"Artifact root: {args.output.resolve()}", flush=True)
    import pandas as pd
    import numpy as np
    import torch
    if args.disable_optional_vision:
        from transformers.utils import import_utils
        import_utils._torchvision_available = False
        print("Explicit compatibility mode: optional Transformers TorchVision backend disabled", flush=True)
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    from huggingface_hub import snapshot_download
    from nnets_transformer.execution.reference import SCORING_VERSION, selections, serial_intervention

    root = args.historical_root
    files = [root / "ablation_data" / f"{args.checkpoint}.csv", root / "induction_data.json", root / "loss_data.json", root / "main.py"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    historical = pd.read_csv(files[0], index_col=0)
    induction = json.loads(files[1].read_text())[args.checkpoint]
    loss = json.loads(files[2].read_text())[args.checkpoint]
    job.stage("checkpoint_loading", checkpoint=args.checkpoint, revision=args.revision)
    snapshot = snapshot_download("EleutherAI/pythia-14m", revision=args.revision, cache_dir=str(args.cache_dir),
                                 allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-14m", revision=args.revision,
                tokenizer=tokenizer, device=args.device, dtype=torch.float32,
                cache_dir=str(args.cache_dir), fold_ln=True, center_unembed=True, center_writing_weights=True)
    model.eval()
    text = historical_text(root / "main.py")
    tokens = model.to_tokens(text, padding_side="left")
    ids = list(selections(model))
    selection_ids = [n + (f".{h}" if h is not None else "") for n, h in ids]
    if set(selection_ids) != set(historical.index):
        raise ValueError("Current and historical intervention inventories differ")
    print(f"Checkpoint {args.checkpoint}@{args.revision}; {len(ids)} serial interventions; tokens={tokens.shape}", flush=True)
    manifest = {
        "schema_version": "historical-reproduction-v1", "scoring_version": SCORING_VERSION,
        "model": "EleutherAI/pythia-14m", "checkpoint": args.checkpoint, "model_revision": args.revision,
        "tokenizer_revision": args.revision, "training_tokens": None,
        "checkpoint_stage": "pretraining", "checkpoint_step": int(args.checkpoint.removeprefix("step")),
        "dataset_id": "legacy-fixed-passage-v1", "seed": None,
        "model_dimensions": {k: getattr(model.cfg, k) for k in ("n_layers", "n_heads", "n_key_value_heads", "d_model", "d_head", "d_mlp", "d_vocab")},
        "prediction_positions": [tokens.shape[1]-1], "candidate_token_ids": [2205],
        "score_units": {"logit": "raw_logit", "loss": "nats_per_next_token"},
        "artifacts": {"matrix": f"{args.checkpoint}.csv", "scores": "scores.json", "comparison": "comparison.json"},
        "historical_revision_status": "recovered current branch commit; original run revision unrecorded",
        "historical_environment": "unrecorded; comparison uses repository lockfile baseline",
        "optional_vision_disabled": args.disable_optional_vision,
        "python": sys.version, "prefix": sys.prefix, "device": args.device, "dtype": "float32",
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "versions": {n: metadata.version(n) for n in ["torch", "transformer-lens", "transformers", "pandas", "networkx"]},
        "transformer_lens_source": metadata.distribution("transformer-lens").read_text("direct_url.json"),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], text=True),
        "source_hashes": {str(p): sha(p) for p in files + [Path("poetry.lock"), Path(__file__), Path(__file__).with_name("reference.py")]},
        "tokenizer_hashes": {p.name: sha(p) for p in Path(snapshot).glob("*token*") if p.is_file()},
        "preprocessing": {"fold_ln": True, "center_unembed": True, "center_writing_weights": True, "prepend_bos": True, "padding_side": "left"},
        "passage_sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens": tokens.cpu().tolist(),
        "interventions": selection_ids, "forward_count": 2 * len(ids), "log_directory": str(job.directory),
        "state": "running",
    }
    write_json(args.output / "manifest.json", manifest)
    start = time.monotonic()
    rows, scores = {}, {}
    job.stage("interventions")
    for index, ((name, head), sid) in enumerate(zip(ids, selection_ids), 1):
        rows[sid], scores[sid] = serial_intervention(model, tokens, name, head)
        if index % 10 == 0 or index == len(ids):
            job.progress("interventions", index, len(ids))
    job.stage("persistence")
    matrix = pd.DataFrame.from_dict(rows, orient="index").reindex(index=historical.index, columns=historical.columns)
    if matrix.isna().any().any():
        raise ValueError("Observation inventories differ or measurements are missing")
    matrix.to_csv(args.output / f"{args.checkpoint}.csv")
    write_json(args.output / "scores.json", scores)
    similarity_error = float(np.abs(matrix.to_numpy() - historical.to_numpy()).max())
    errors = {"patched_logit": max(abs(s["patched_logit"] - induction[k]) for k, s in scores.items()),
              "patched_loss": max(abs(s["patched_loss"] - loss[k]) for k, s in scores.items()),
              "clean_logit": max(abs(s["clean_logit"] - induction["clean"]) for s in scores.values()),
              "clean_loss": max(abs(s["clean_loss"] - loss["clean"]) for s in scores.values())}
    report = {"matrix_shape": list(matrix.shape), "max_absolute_similarity_error": similarity_error,
              "max_absolute_score_errors": errors, "similarity_atol": 2e-5, "score_atol": 2e-4,
              "elapsed_seconds": time.monotonic()-start,
              "passed": similarity_error <= 2e-5 and all(v <= 2e-4 for v in errors.values())}
    write_json(args.output / "comparison.json", report)
    manifest.update(state="completed" if report["passed"] else "comparison_failed", comparison=report)
    write_json(args.output / "manifest.json", manifest)
    job.artifact(args.output / "manifest.json")
    job.event("scientific_comparison", **report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Compare clean source-model and TransformerLens logits on a fixed text.

Uses the local immutable checkpoint cache; validates the adapter's weight
conversion (with and without preprocessing) against the Hugging Face model.
"""
import argparse
import hashlib
import json
from pathlib import Path

from nnets_transformer.observability import Job, Interrupted

TEXT = ("When Mary and John went to the store, John gave a drink to Mary. The war lasted from the year 1732 to "
        "the year 1745. The keys to the cabinet are on the table, and the author that the guards like writes "
        "every morning. A B C D E F G H, A B C D E F G H.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-14m")
    parser.add_argument("--revision", default="5b020995bfc7aee2931b0f35bd70cf7ee8b1db62")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/hf-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-optional-vision", action="store_true")
    args = parser.parse_args()
    try:
        with Job("clean-adapter-check", artifact_root=args.output.parent) as job:
            if args.output.exists():
                raise FileExistsError(args.output)
            snapshot = args.cache_dir / ("models--" + args.model.replace("/", "--")) / "snapshots" / args.revision
            if not (snapshot / "config.json").is_file():
                raise FileNotFoundError(f"Immutable checkpoint must already be cached: {snapshot}")
            import torch
            if args.disable_optional_vision:
                from transformers.utils import import_utils
                import_utils._torchvision_available = False
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformer_lens import HookedTransformer
            job.stage("source_loading", revision=args.revision)
            hf = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.float32,
                    attn_implementation="eager").to(args.device).eval()
            tokenizer = AutoTokenizer.from_pretrained(snapshot)
            results = {}
            for processed in (False, True):
                job.stage("clean_comparison", weight_preprocessing=processed)
                model = HookedTransformer.from_pretrained(args.model,
                    tokenizer=tokenizer, revision=args.revision, device=args.device,
                    cache_dir=str(args.cache_dir), fold_ln=processed, center_unembed=False,
                    center_writing_weights=processed)
                model.eval()
                tokens = model.to_tokens(TEXT)
                with torch.no_grad():
                    logits, loss = model(tokens, return_type="both")
                    reference = hf(tokens, labels=tokens)
                    results[str(processed)] = {
                        "max_absolute_logit_error": (logits-reference.logits).abs().max().item(),
                        "tl_loss": loss.item(), "hf_loss": reference.loss.item(),
                    }
                job.progress("conditions", len(results), 2)
            report = {"model": args.model, "revision": args.revision, "text_sha256": hashlib.sha256(TEXT.encode()).hexdigest(), "optional_vision_disabled": args.disable_optional_vision,
                      "source_config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "log_directory": str(job.directory), "logit_atol": 2e-4, "loss_atol": 2e-6,
                      "results": results}
            report["passed"] = all(r["max_absolute_logit_error"] <= report["logit_atol"] and
                    abs(r["tl_loss"]-r["hf_loss"]) <= report["loss_atol"] for r in results.values())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            job.artifact(args.output)
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise SystemExit(1)
    except Interrupted as exc:
        raise SystemExit(128+exc.signum)


if __name__ == "__main__":
    main()

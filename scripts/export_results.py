"""Export completed sweep cells to dashboard/analysis tables, one export per (model, task).

    python scripts/export_results.py --runs runs/full --exports exports/full [--model pythia-14m]

Only completed (and, when the sweep compacts, compacted) cells are read, so this is safe to run
while a sweep is still going. Each export is built in a hidden folder and swapped in atomically.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from nnets_transformer.observability import Job, Interrupted


def finished(cell, compacting):
    path = cell / "manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    return manifest.get("state") == "completed" and (not compacting or bool(manifest.get("compaction")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/full"))
    parser.add_argument("--exports", type=Path, default=Path("exports/full"))
    parser.add_argument("--model", action="append", help="model folder name (repeatable); default: all")
    parser.add_argument("--thresholds", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--log-root", default="logs")
    args = parser.parse_args()
    from nnets_transformer.analysis.export import normalize
    from nnets_transformer.config import Graph
    thresholds = [float(t) for t in args.thresholds.split(",")]
    try:
        with Job("export-results", root=args.log_root) as job:
            models = sorted(p.parent for p in args.runs.glob("*/sweep.json"))
            if args.model:
                models = [m for m in models if m.name in args.model]
            groups = []
            for model in models:
                sweep = json.loads((model / "sweep.json").read_text())
                compacting = sweep.get("compact", "none") != "none"
                for task in sorted({p.name for p in model.glob("step*/*") if p.is_dir()}):
                    cells = sorted(c for c in model.glob(f"step*/{task}/*") if finished(c, compacting))
                    if cells:
                        groups.append((model.name, task, cells))
            for number, (model, task, cells) in enumerate(groups, 1):
                job.stage("export", model=model, task=task, cells=len(cells))
                target = args.exports / model / task
                building = args.exports / model / f".{task}.building-{os.getpid()}"
                shutil.rmtree(building, ignore_errors=True)
                normalize([str(c) for c in cells], building, Graph(resolution="component"), job, thresholds,
                          include_activations=False, behavior_rows=False, slim=True)
                previous = args.exports / model / f".{task}.previous-{os.getpid()}"
                if target.exists():
                    target.rename(previous)
                building.rename(target)
                shutil.rmtree(previous, ignore_errors=True)
                job.progress("exports", number, len(groups))
                print(f"exported {model}/{task}: {len(cells)} cells -> {target}", flush=True)
            if not groups:
                print(f"No completed cells under {args.runs} yet.")
    except Interrupted as exc:
        raise SystemExit(128+exc.signum)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Every main Pythia model (14M-12B) x every available checkpoint x every task x every replacement.
#
#   scripts/run_everything.sh              # run + export (resumable: rerun the same command to continue)
#   scripts/run_everything.sh export       # refresh dashboard/analysis exports now (CPU; safe while running)
#   DRY_RUN=1 scripts/run_everything.sh    # print the plan, checkpoint counts and disk estimates only
#
# Outputs: results runs/full/<model>/<checkpoint>/<task>/<replacement>/ (similarity matrix, per-example
# behavior, manifests, per-example float16 cosine); exports exports/full/<model>/<task>/ (graph metrics,
# roles, edges, behavior summaries) after each model, read by the dashboard.
#
# Run it inside tmux. Ctrl+C stops the current sweep cleanly (recorded as interrupted, resumable)
# and does not start the next model. Settings (environment variables, with defaults):
#   CUDA_VISIBLE_DEVICES=0   GPU to use
#   OUTPUT_ROOT=runs/full    results: OUTPUT_ROOT/<model>/<checkpoint>/<task>/<replacement>/
#   REPLACEMENTS="zero mean mean_global donor donor_denoising donor_resample"
#   SCOPE=component          component | head (head resolution needs far more disk and time)
#   MIN_FREE_GB=100          stop when the results disk has less free space than this
#   KEEP=auto                after each cell, compact it (keeps graphs, all behavior rows, manifests):
#                            cosine16 = also keep per-example cosine as float16; means = mean similarities only;
#                            auto = cosine16 if the model's estimate fits, else means, else skip the model
#   MODELS="..."             space-separated subset of the model IDs listed below
# Models are processed smallest first. A model whose estimated (compacted) results do not fit in the
# free space above MIN_FREE_GB is skipped (reported), and the script moves on. Downloaded checkpoints are deleted
# after use. Deduplicated and seed variants are not included (see the catalog comment).
set -uo pipefail
cd "$(dirname "$0")/.."

export PYENV_VERSION=nnets-transformer
export VIRTUAL_ENV=/data/shared/pyenv/versions/3.11.1/envs/nnets-transformer
export PATH="$VIRTUAL_ENV/bin:$PATH"
export POETRY_VIRTUALENVS_CREATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/full}"
REPLACEMENTS="${REPLACEMENTS:-zero mean mean_global donor donor_denoising donor_resample}"
SCOPE="${SCOPE:-component}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
KEEP="${KEEP:-auto}"
DRY_RUN="${DRY_RUN:-0}"

# model id | base model YAML | layers | parameters in millions (download size ~2 bytes/parameter)
# Training-seed replicates loadable here can be added as lines such as:
#   EleutherAI/pythia-160m-seed1|pythia_160m|12|162   (also -seed2, -seed3)
CATALOG="
EleutherAI/pythia-14m|pythia_14m|6|14
EleutherAI/pythia-31m|pythia_31m|6|31
EleutherAI/pythia-70m|pythia_70m|6|70
EleutherAI/pythia-160m|pythia_160m|12|162
EleutherAI/pythia-410m|pythia_410m|24|405
EleutherAI/pythia-1b|pythia_1b|16|1012
EleutherAI/pythia-1.4b|pythia_1_4b|24|1415
EleutherAI/pythia-2.8b|pythia_2_8b|32|2775
EleutherAI/pythia-6.9b|pythia_2_8b|32|6857
EleutherAI/pythia-12b|pythia_2_8b|36|11846
"

EXPORT_ROOT="${EXPORT_ROOT:-exports/full}"
mkdir -p logs/run-everything "$OUTPUT_ROOT"
if [ "${1:-}" = export ]; then
    exec poetry run python -u scripts/export_results.py --runs "$OUTPUT_ROOT" --exports "$EXPORT_ROOT"
fi
LOG="logs/run-everything/$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -i -a "$LOG") 2>&1
echo "run_everything: log $LOG; output $OUTPUT_ROOT; replacements [$REPLACEMENTS]; scope $SCOPE; GPU $CUDA_VISIBLE_DEVICES"

if [ "$DRY_RUN" != 1 ] && pgrep -f "nnets-transformer.*(command=sweep|experiment=pilot)" >/dev/null; then
    echo "Another sweep is already running (pgrep -af nnets-transformer). Wait for it to finish, then rerun." >&2
    exit 1
fi

REPLACEMENT_LIST="[$(echo $REPLACEMENTS | tr ' ' ',')]"
# Tasks x replacements; matched donors are undefined for induction and are skipped by the sweep.
MATCHED=$(for r in $REPLACEMENTS; do case $r in donor|donor_denoising) echo x;; esac; done | wc -l)
CELLS=$(( 5 * $(echo $REPLACEMENTS | wc -w) - MATCHED ))
# Compacted size per cell, fitted to 14M/160M pilot cells (component resolution, S = 1 + 7*layers sources):
#   ~310 KB dataset + 2,250 B * S behavior rows + (75 B cosine16 | 40 B means) * S^2.
# Head resolution has far more sources; the estimate is then a lower bound.
estimate_gb() {  # layers checkpoints mode
    local sources=$(( 1 + 7 * $1 )) quad=75
    [ "$3" = means ] && quad=40
    echo $(( ($2 * CELLS * (310000 + 2250 * sources + quad * sources * sources) + 999999999) / 1000000000 ))
}
free_gb() { df -BG --output=avail "$OUTPUT_ROOT" | tail -1 | tr -dc 0-9; }
checkpoints() {  # model id -> "[step0,step1,...]" sorted numerically; every available step branch
    python - "$1" <<'EOF'
import sys
from huggingface_hub import HfApi
steps = sorted(int(r.name[4:]) for r in HfApi().list_repo_refs(sys.argv[1]).branches
               if r.name.startswith("step") and r.name[4:].isdigit())
print("[" + ",".join(f"step{s}" for s in steps) + "]")
EOF
}

STOP=0
trap 'STOP=1; echo "run_everything: stop requested; finishing the current sweep shutdown"' INT TERM
LOW_DISK_FLAG="$(mktemp -u)"
if [ "$DRY_RUN" != 1 ]; then
    # Disk guard: interrupt the running sweep (resumably) if free space falls below the floor.
    ( while sleep 300; do
        if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
            touch "$LOW_DISK_FLAG"
            pkill -INT -f "nnets-transformer.*full-" || true
            exit
        fi
      done ) &
    GUARD=$!
    trap 'kill $GUARD 2>/dev/null; rm -f "$LOW_DISK_FLAG"' EXIT
fi

printf '%-34s %6s %10s %10s  %s\n' MODEL CKPTS EST_GB FREE_GB STATUS
for line in $CATALOG; do
    IFS='|' read -r ID BASE LAYERS PARAMS <<<"$line"
    if [ -n "${MODELS:-}" ] && ! [[ " $MODELS " == *" $ID "* ]]; then continue; fi
    [ "$STOP" = 1 ] && break
    if [ -e "$LOW_DISK_FLAG" ]; then echo "run_everything: free space below ${MIN_FREE_GB} GB; stopping."; break; fi
    SLUG="${ID#*/}"
    CK=$(checkpoints "$ID") || { printf '%-34s %6s %10s %10s  %s\n' "$ID" "-" "-" "-" "skipped: cannot list checkpoints"; continue; }
    N=$(echo "$CK" | tr ',' '\n' | wc -l)
    FREE=$(free_gb)
    STATE_FILE="$OUTPUT_ROOT/$SLUG/sweep.json"
    DOWNLOAD=$(( (PARAMS * 2 + 999) / 1000 ))   # one checkpoint at a time; evicted after use
    MODE=""
    for candidate in $( [ "$KEEP" = auto ] && echo cosine16 means || echo "$KEEP" ); do
        NEED=$(( $(estimate_gb "$LAYERS" "$N" "$candidate") + DOWNLOAD ))
        if [ $(( FREE - MIN_FREE_GB )) -ge "$NEED" ] || [ -f "$STATE_FILE" ]; then MODE=$candidate; break; fi
    done
    if [ -f "$STATE_FILE" ]; then   # resuming: keep the mode the model started with
        MODE=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('compact') or 'none')" "$STATE_FILE")
    fi
    if [ -z "$MODE" ]; then
        printf '%-34s %6s %10s %10s  %s\n' "$ID" "$N" "$NEED" "$FREE" "skipped: estimate exceeds free space above ${MIN_FREE_GB} GB"
        continue
    fi
    if [ "$DRY_RUN" = 1 ]; then
        printf '%-34s %6s %10s %10s  %s\n' "$ID" "$N" "$NEED" "$FREE" "would run (keep $MODE)"
        continue
    fi
    printf '%-34s %6s %10s %10s  %s\n' "$ID" "$N" "$NEED" "$FREE" "running (keep $MODE)"
    RESUME=()
    [ -f "$STATE_FILE" ] && RESUME=(execution.resume=true)
    poetry run nnets-transformer --job-id "full-$SLUG" +experiment=pilot model="$BASE" model.id="$ID" model.n_layers="$LAYERS" \
        scope="$SCOPE" "sweep.checkpoints=$CK" "sweep.replacements=$REPLACEMENT_LIST" \
        sweep.output_root="$OUTPUT_ROOT" sweep.evict_checkpoint_cache=true sweep.compact="$MODE" "${RESUME[@]}"
    STATUS=$?
    if [ "$STATUS" -eq 0 ] || [ "$STATUS" -lt 128 ]; then   # export whatever this model completed
        poetry run python -u scripts/export_results.py --runs "$OUTPUT_ROOT" --exports "$EXPORT_ROOT" --model "$SLUG" \
            || echo "run_everything: export for $ID failed; retry with: scripts/run_everything.sh export"
    fi
    if [ "$STATUS" -ne 0 ]; then
        echo "run_everything: $ID exited with $STATUS (logs/full-$SLUG/); rerun this script to resume it."
        [ "$STATUS" -ge 128 ] && STOP=1
    fi
done
echo "run_everything: done (stop=$STOP). Rerun the same command to resume or continue."

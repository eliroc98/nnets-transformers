#!/usr/bin/env bash
# One resumable Slurm job per model. Override MODELS, SLURM_MAIL_USER, AUTO_REQUEUE,
# OUTPUT_ROOT, EXPORT_ROOT, INTERVENTION_BATCH_SIZE or add sbatch options as args.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster_env.sh
mkdir -p logs/slurm
MODEL_IDS="${MODELS:-EleutherAI/pythia-14m EleutherAI/pythia-31m EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b}"
MAIL=()
[ -z "${SLURM_MAIL_USER:-}" ] || MAIL=(--mail-user="$SLURM_MAIL_USER")
for model in $MODEL_IDS; do
    memory=64G
    batch=128
    case "$model" in
        *pythia-2.8b) memory=128G; batch=64;;
        *pythia-6.9b|*pythia-12b) memory=128G; batch=32;;
    esac
    INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-$batch}" \
        sbatch --parsable --job-name="influence-${model##*/}" --mem="$memory" \
        --output="$PROJECT_DIR/logs/slurm/%x-%j.out" \
        --export="ALL,PROJECT_DIR=$PROJECT_DIR,INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE:-$batch}" \
        "${MAIL[@]}" "$@" scripts/full_model.sbatch "$model"
done

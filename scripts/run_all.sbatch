#!/bin/bash
# Submit once from the repository: mkdir -p logs/slurm; sbatch scripts/run_all.sbatch
# Each array task runs one model's complete checkpoint/task/replacement grid.
#SBATCH --job-name=nnets-full-grid
#SBATCH --output=/exa/projects/TasCN/nnets-transformer/logs/slurm/full-grid_%A_%a.out
#SBATCH --error=/exa/projects/TasCN/nnets-transformer/logs/slurm/full-grid_%A_%a.err
#SBATCH --account=TasCN
#SBATCH --partition=math
#SBATCH --array=0-9%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --get-user-env
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elisabetta.rocchetti@unimi.it
#SBATCH --signal=B:TERM@600
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
export PROJECT_DIR="/exa/projects/TasCN/nnets-transformer"
cd "$PROJECT_DIR"

MODELS=(
    EleutherAI/pythia-14m EleutherAI/pythia-31m EleutherAI/pythia-70m
    EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b
    EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b
    EleutherAI/pythia-12b
)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:?submit with sbatch}]}"
case "$MODEL" in
    *pythia-2.8b) export INTERVENTION_BATCH_SIZE=64;;
    *pythia-6.9b|*pythia-12b) export INTERVENTION_BATCH_SIZE=32;;
    *) export INTERVENTION_BATCH_SIZE=128;;
esac

echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-N/A}"
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Model: $MODEL; outputs: ${OUTPUT_ROOT:-runs/full}"

# Shared module/venv/cache setup, logging, signal forwarding and resumable requeue.
# exec preserves delivery of Slurm's batch-shell-only TERM signal.
exec bash scripts/full_model.sbatch "$MODEL"

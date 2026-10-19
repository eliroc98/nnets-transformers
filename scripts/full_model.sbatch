#!/usr/bin/env bash
# Submit through scripts/submit_full.sh so log directories already exist.
#SBATCH --account=tascn
#SBATCH --partition=math
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --requeue
#SBATCH --get-user-env
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
set -uo pipefail
cd "${PROJECT_DIR:?submit_full.sh must supply PROJECT_DIR}"
source scripts/cluster_env.sh || exit 1
export MODELS="${1:?one model ID is required}"
REQUEUE_REQUESTED=0
LOGGER=""
stop_job() {
    REQUEUE_REQUESTED=1
    echo "Slurm advance TERM: forwarding to experiment logger ($LOGGER)"
    [ -z "$LOGGER" ] || kill -TERM "$LOGGER" 2>/dev/null || true
}
trap stop_job TERM
trap '[ -z "$LOGGER" ] || kill -INT "$LOGGER" 2>/dev/null || true' INT
poetry run -- python -m nnets_transformer.observability \
    --job-id "slurm-${SLURM_JOB_ID}-${MODELS##*/}" \
    --artifact-root "${OUTPUT_ROOT:-runs/full}/${MODELS##*/}" -- \
    bash scripts/run_everything.sh &
LOGGER=$!
wait "$LOGGER"; STATUS=$?
if [ "$REQUEUE_REQUESTED" = 1 ]; then
    wait "$LOGGER" 2>/dev/null || true
    if [ "${AUTO_REQUEUE:-1}" = 1 ]; then
        poetry run -- python -m nnets_transformer.observability \
            --job-id "slurm-requeue-${SLURM_JOB_ID}" -- scontrol requeue "$SLURM_JOB_ID"
        exit $?
    fi
    exit 143
fi
exit "$STATUS"

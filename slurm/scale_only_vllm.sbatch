#!/bin/bash
# TEMPLATE - NOT TESTED (no access to the BGU cluster from where this was written).
# Fill every <PLACEHOLDER> from the cluster documentation / your own account before submitting.
# Do not submit without approval: this starts real model generation.
#SBATCH --job-name=benchson-scale
#SBATCH --partition=<PARTITION>            # a partition that has 80GB-class GPUs
#SBATCH --account=<ACCOUNT>                # if your site requires one
#SBATCH --gpus=<GPU_TYPE>:<N>              # see docs/scale_only_experiment.md section 7 for the sizing
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=<HH:MM:SS>                  # the builder resumes, so a job can be re-submitted after a timeout
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# ---- environment (cluster specific) -------------------------------------------------
# module load <PYTHON/CUDA MODULES>
# source <PATH_TO_VENV>/bin/activate       # needs vllm + this repo's requirements (jsonschema, ...)

export GRANITE_MODEL_ID="<EXACT HF ID OF GRANITE 4 H-SMALL, e.g. as used in the original run>"
export PUBLISHED_SHA12="<first 12 chars of the pinned dataset revision>"
export RUN_NAME="${RUN_NAME:-run1}"
export VLLM_BASE_URL="http://localhost:8000/v1"
export HF_HOME="${HF_HOME:-<WRITABLE_CACHE_DIR>}"     # model weights are large; do not use $HOME quota

# ---- 1. start the model server on the allocated GPU(s) ------------------------------
vllm serve "${GRANITE_MODEL_ID}" \
    --port 8000 \
    --tensor-parallel-size "${TP_SIZE:-1}" \
    --max-model-len <MAX_LEN> \
    > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    curl -sf "${VLLM_BASE_URL}/models" >/dev/null && break
    sleep 10
done
curl -sf "${VLLM_BASE_URL}/models" >/dev/null || { echo "vLLM did not start"; exit 1; }

# ---- 2. generation (resumable: finished files are skipped) --------------------------
python src/build_benchmark.py \
    --config configs/scale_only_vllm_granite.json \
    ${SMOKE_TOTAL:+--smoke-total ${SMOKE_TOTAL}} \
    --record-calls "data_scale_only/${RUN_NAME}_calls.jsonl"

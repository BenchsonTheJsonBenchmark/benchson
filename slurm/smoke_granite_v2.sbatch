#!/bin/bash
# Scale-only smoke test with IBM Granite on vLLM: 2 examples for each of Create, Fix, Modify.
#
# Before submitting, run a separate CPU preflight from the checkout:
#   bash slurm/smoke_granite_v2.sbatch --preflight cluster
# Create logs/ before submitting so Slurm can open its output file.
#SBATCH --job-name=benchson-smoke
#SBATCH --partition=main
#SBATCH --account=4th-year-proj
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/benchson-smoke-%j.out
#SBATCH --open-mode=append

set -uo pipefail
umask 077
fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 10; }

if [ "$#" -eq 2 ] && [ "$1" = "--preflight" ] && [[ "$2" = local || "$2" = cluster ]]; then
    MODE="$2"
    cd "$(dirname "$0")/.." || fail "cannot enter checkout"
elif [ "$#" -eq 0 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    MODE=job
    cd "$SLURM_SUBMIT_DIR" || fail "cannot enter Slurm submission directory"
else
    fail "use --preflight local|cluster outside Slurm"
fi

JOB="${SLURM_JOB_ID:-$$}"
[[ "$JOB" =~ ^[1-9][0-9]*$ ]] || fail "invalid job ID"
# A requeued/preempted job runs again under the SAME job ID; Slurm increments SLURM_RESTART_COUNT for every new attempt.
# Every path below is named by RUN_ID = <job>-r<attempt>, so attempts never share or overwrite logs, scratch or output.
ATTEMPT="${SLURM_RESTART_COUNT:-0}"
[[ "$ATTEMPT" =~ ^(0|[1-9][0-9]{0,5})$ ]] || fail "invalid SLURM_RESTART_COUNT"
RUN_ID="${JOB}-r${ATTEMPT}"
export GRANITE_MODEL_ID="${GRANITE_MODEL_ID:-ibm-granite/granite-4.0-h-small}"
export GRANITE_REVISION="${GRANITE_REVISION:-b8c0982bab7fde4eb48110f5a069527c008fab39}"
PUBLISHED_REVISION=7962b0947f7d96eb3a0e6b9b928d939c7f3061a6
export PUBLISHED_SHA12="${PUBLISHED_REVISION:0:12}"
# SHA-256 of the exported anchor pool (python src/export_anchor_pool.py): pins the exact set of published train/test schemas.
export ANCHOR_SHA256="${ANCHOR_SHA256:-ab09def994a78920d1df6bfa2d0c11f6e135294a46ebc33baca9307aee0327f6}"
export RUN_NAME="smoke_${RUN_ID}"
VLLM_ENV="${VLLM_ENV:-/storage/cluster/vLLM2}"
if [ "$MODE" = local ]; then
    BENCHSON_PYTHON="${BENCHSON_PYTHON:-.venv/bin/python}"
else
    BENCHSON_PYTHON="${BENCHSON_PYTHON:-$HOME/.conda/envs/benchson/bin/python}"
fi
MAX_LLM_CALLS="${MAX_LLM_CALLS:-60}"
SMOKE_TOTAL=6
PER_TASK=2
CONFIG="configs/scale_only_vllm_granite.json"
OUT_DIR="data_scale_only/${RUN_NAME}"
CALLS_LOG="data_scale_only/${RUN_NAME}_calls.jsonl"
VLLM_PYTHON="${VLLM_ENV}/bin/python"
VLLM_BIN="${VLLM_ENV}/bin/vllm"
export PATH="${VLLM_ENV}/bin:$PATH"

echo "=== Preflight ($MODE) job=$JOB attempt=$ATTEMPT run=$RUN_ID ==="
BRANCH=$(git branch --show-current) || fail "cannot identify branch"
[ "$BRANCH" = improve-data-generation ] || fail "expected improve-data-generation; found $BRANCH"
[ -x "$BENCHSON_PYTHON" ] || fail "benchson python not found: $BENCHSON_PYTHON"
[ -f "$CONFIG" ] || fail "config not found: $CONFIG"
"$BENCHSON_PYTHON" -c 'import deepdiff, jsonschema' || fail "benchson python lacks deepdiff/jsonschema"
# The schema pool comes from the published training rows (docs/schema_reuse.md); data/jsb_schemas is not needed for it.
POOL_SOURCE=$("$BENCHSON_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["reuse"]["pool_source"])' "$CONFIG") \
    || fail "cannot read reuse.pool_source from $CONFIG"
[ "$POOL_SOURCE" = published_train ] || fail "this job needs reuse.pool_source=published_train (found: $POOL_SOURCE)"
if [ "$MODE" = local ]; then
    echo "Local checks passed. Run --preflight cluster with the real baseline and cluster environments."
    exit 0
fi

BASELINE="data_published/${PUBLISHED_SHA12}/reserved_schemas.json"
"$BENCHSON_PYTHON" - "$BASELINE" "$PUBLISHED_REVISION" <<'PYBASE' || fail "missing or invalid published baseline: $BASELINE"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    baseline = json.load(stream)
if baseline["revision"] != sys.argv[2]:
    sys.exit("published baseline revision does not match the pin")
for key in ("ids", "schema_hashes"):
    for split in ("train", "test"):
        values = baseline[key][split]
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
            sys.exit(f"missing or invalid {key}.{split}")
PYBASE
ANCHOR="data_published/${PUBLISHED_SHA12}/published_train_schemas.json"
"$BENCHSON_PYTHON" - "$ANCHOR" "$BASELINE" "$PUBLISHED_REVISION" "$ANCHOR_SHA256" <<'PYANCHOR' || fail "anchor pool missing, corrupt, or not the pinned one: $ANCHOR"
import sys

sys.path.insert(0, "src")
from data_generation.anchor import load_anchor

# Verifies content hashes, ids and agreement with reserved_schemas.json (same revision, same train/test ids and hashes).
anchor = load_anchor(sys.argv[1], sys.argv[2])
if anchor["revision"] != sys.argv[3]:
    sys.exit("anchor revision does not match the pin")
if anchor["anchor_sha256"] != sys.argv[4]:
    sys.exit("anchor_sha256 does not match the pinned value")
PYANCHOR
POOL_REPORT=$("$BENCHSON_PYTHON" src/build_benchmark.py --config "$CONFIG" --smoke-total "$SMOKE_TOTAL" \
    --out-dir "$OUT_DIR" --check-pool) || fail "pool check failed"
printf '%s\n' "$POOL_REPORT" | "$BENCHSON_PYTHON" -c '
import json
import sys

report = json.load(sys.stdin)
pool, tasks = report["pool"], report["per_task"]
if pool.get("pool_source") != "published_train" or pool.get("anchor_sha256") != sys.argv[1]:
    sys.exit("pool is not the pinned published_train anchor")
if pool.get("usable_schemas", 0) < 1 or set(tasks) != {"create", "fix", "modify"}:
    sys.exit("empty pool or wrong task set")
for name, row in sorted(tasks.items()):   # each task on its own: target exactly 2 and enough reference slots for it
    if row["target"] != 2 or row["upper_bound_if_every_attempt_succeeds"] < row["target"]:
        sys.exit(f"schema pool cannot supply the 2 smoke examples for {name}")
' "$ANCHOR_SHA256" || fail "pool does not meet smoke requirements"

[ -x "$VLLM_PYTHON" ] && [ -x "$VLLM_BIN" ] || fail "vLLM environment not found: $VLLM_ENV"
for tool in ninja setsid curl; do
    command -v "$tool" >/dev/null || fail "$tool is not on PATH"
done
"$BENCHSON_PYTHON" -c 'import openai' || fail "benchson python lacks openai"
"$VLLM_PYTHON" -c 'import huggingface_hub, vllm' || fail "vLLM python lacks huggingface_hub/vllm"
if [ "$MODE" = cluster ]; then
    printf '%s\n' "$POOL_REPORT"
    echo "Cluster CPU preflight passed."
    exit 0
fi

[[ "$MAX_LLM_CALLS" =~ ^[1-9][0-9]*$ ]] && [ "$MAX_LLM_CALLS" -le 60 ] || fail "MAX_LLM_CALLS must be 1–60"
mkdir -p logs data_scale_only || fail "cannot create output directories"
for path in "$OUT_DIR" "$CALLS_LOG" "logs/vllm-${RUN_ID}.log" "logs/pool-check-${RUN_ID}.json"; do
    [[ ! -e "$path" && ! -L "$path" ]] || fail "job path already exists: $path"
done
printf '%s\n' "$POOL_REPORT" > "logs/pool-check-${RUN_ID}.json" || fail "cannot write pool report"

# SCRATCH_DIR, when provided, is an existing parent directory only.
if [ -z "${SCRATCH_DIR:-}" ]; then
    SCRATCH_DIR="/scratch/${USER}"
    mkdir -p "$SCRATCH_DIR" || fail "cannot create scratch parent"
fi
SCRATCH_PARENT=$(cd -- "$SCRATCH_DIR" && pwd -P) || fail "invalid scratch parent"
[[ "$SCRATCH_PARENT" != / && -O "$SCRATCH_PARENT" && -w "$SCRATCH_PARENT" ]] || fail "scratch parent must be owned and writable"
FREE_KB=$(df -Pk "$SCRATCH_PARENT" | awk 'NR==2 {print $4}') || fail "cannot check scratch space"
[[ "$FREE_KB" =~ ^[0-9]+$ ]] && [ "$FREE_KB" -ge 94371840 ] || fail "scratch requires 90 GiB free"
JOB_SCRATCH=""
SCRATCH_INODE=""
VLLM_PID=""
WORK_PID=""

cleanup() {
    local rc=$? pid
    trap '' HUP INT TERM
    for pid in "$WORK_PID" "$VLLM_PID"; do
        [ -n "$pid" ] || continue
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "$WORK_PID" "$VLLM_PID"; do
        [ -n "$pid" ] || continue
        if kill -0 -- "-$pid" 2>/dev/null; then
            sleep 1
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
    if [[ -n "$SCRATCH_INODE" && "$JOB_SCRATCH" == "$SCRATCH_PARENT"/benchson-smoke-"$RUN_ID".* &&
          -d "$JOB_SCRATCH" && -O "$JOB_SCRATCH" && ! -L "$JOB_SCRATCH" &&
          "$(ls -di -- "$JOB_SCRATCH" | awk '{print $1}')" == "$SCRATCH_INODE" ]]; then
        rm -rf -- "$JOB_SCRATCH" || { echo "Scratch cleanup failed: $JOB_SCRATCH" >&2; [ "$rc" -ne 0 ] || rc=3; }
    elif [ -n "$JOB_SCRATCH" ]; then
        echo "Scratch identity check failed: $JOB_SCRATCH" >&2
        [ "$rc" -ne 0 ] || rc=3
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

JOB_SCRATCH=$(mktemp -d "$SCRATCH_PARENT/benchson-smoke-$RUN_ID.XXXXXXXX") || fail "cannot create private scratch"
SCRATCH_INODE=$(ls -di -- "$JOB_SCRATCH" | awk '{print $1}') || fail "cannot identify private scratch"
[[ "$SCRATCH_INODE" =~ ^[0-9]+$ ]] || fail "cannot identify private scratch"
export HF_HOME="$JOB_SCRATCH/huggingface"
export XDG_CACHE_HOME="$JOB_SCRATCH/cache"
export VLLM_CACHE_ROOT="$JOB_SCRATCH/vllm_cache"
export TMPDIR="$JOB_SCRATCH/tmp"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" || fail "cannot create cache directories"

SERVER_PORT=$((20000 + (JOB + ATTEMPT * 1009) % 20000))   # a lingering server of an earlier attempt cannot collide
export VLLM_BASE_URL="http://127.0.0.1:${SERVER_PORT}/v1"
VLLM_API_KEY=$("$BENCHSON_PYTHON" -c 'import secrets; print(secrets.token_urlsafe(32))') || fail "could not create API key"
[ -n "$VLLM_API_KEY" ] || fail "empty API key"
export VLLM_API_KEY

echo "=== GPU ==="
nvidia-smi || exit 11

echo "=== Downloading pinned Granite model ==="
setsid "$VLLM_PYTHON" - > "$JOB_SCRATCH/model-path" <<'PY' &
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ["GRANITE_MODEL_ID"], revision=os.environ["GRANITE_REVISION"],
                        cache_dir=os.environ["HF_HOME"] + "/hub"))
PY
WORK_PID=$!
wait "$WORK_PID"
DOWNLOAD_RC=$?
WORK_PID=""
[ "$DOWNLOAD_RC" -eq 0 ] || { echo "model download failed"; exit 11; }
MODEL_PATH=$(cat "$JOB_SCRATCH/model-path")
[ -d "$MODEL_PATH" ] || { echo "invalid model path"; exit 11; }
echo "Model downloaded to: $MODEL_PATH"

echo "=== Starting vLLM on 127.0.0.1:${SERVER_PORT} ==="
# VLLM_API_KEY (exported above) is read by vLLM itself, so the key never appears in the process list.
setsid "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$GRANITE_MODEL_ID" \
    --host 127.0.0.1 --port "$SERVER_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    > "logs/vllm-${RUN_ID}.log" 2>&1 &
VLLM_PID=$!

echo "=== Waiting for vLLM ==="
READY=0
for i in $(seq 1 180); do
    if printf 'header = "Authorization: Bearer %s"\n' "$VLLM_API_KEY" \
        | curl -qsf --noproxy 127.0.0.1 --max-time 10 -K - "$VLLM_BASE_URL/models" >/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM process exited (see logs/vllm-${RUN_ID}.log)."
        exit 11
    fi
    sleep 10
done
[ "$READY" -eq 1 ] || { echo "vLLM did not become ready (see logs/vllm-${RUN_ID}.log)."; exit 11; }

# ---- 2. Smoke test ------------------------------------------------------------------------------
echo "=== Running Benchson smoke test (${SMOKE_TOTAL} examples, ${PER_TASK} per task) ==="
setsid "$BENCHSON_PYTHON" src/build_benchmark.py \
    --config "$CONFIG" \
    --smoke-total "$SMOKE_TOTAL" \
    --out-dir "$OUT_DIR" \
    --max-llm-calls "$MAX_LLM_CALLS" \
    --smoke-openai-timeout 180 \
    --record-calls "$CALLS_LOG" &
WORK_PID=$!
wait "$WORK_PID"
BUILD_RC=$?
WORK_PID=""
echo "build_benchmark.py exit code: $BUILD_RC"

# ---- 3. Judge the result ------------------------------------------------------------------------
echo "=== Checking the result ==="
"$BENCHSON_PYTHON" src/check_smoke_result.py \
    --out-dir "$OUT_DIR" --per-task "$PER_TASK" \
    --anchor-file "$ANCHOR" --calls-log "$CALLS_LOG" --build-exit-code "$BUILD_RC"
CHECK_RC=$?
echo "=== Smoke test finished: check exit code ${CHECK_RC} (0 success, 2 shortfall, 3 technical, 4 invalid) ==="
exit "$CHECK_RC"

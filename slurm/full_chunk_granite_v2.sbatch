#!/bin/bash
# Scale-only FULL RUN, one bounded CHUNK per job: 8 chunks give exactly 1,334 Create / 1,333 Fix / 1,333 Modify new
# training examples. Same generation code, prompts, sampling parameters, model revision, anchored schema pool and reuse
# settings as the smoke test and the pilot. Every limit of Controlled Schema Reuse is GLOBAL: each chunk continues from the
# verified checkpoint of the previous chunk (src/data_generation/chunking.py, docs/full_run_chunking.md).
# Based on slurm/pilot_granite_v2.sbatch (kept untouched). Nothing is submitted, merged or resumed automatically.
#
#   CHUNK_INDEX=1 bash slurm/full_chunk_granite_v2.sbatch --preflight cluster       (CPU only; then sbatch it)
#   CHUNK_INDEX=2 PARENT_CHUNK=c01_<job>-r<n> ...   (the exact, validated directory name of the previous chunk)
# Create logs/ before submitting so Slurm can open its output file. Optional: MODEL_CACHE_DIR=<approved persistent dir>.
#   exit codes: 0 SUCCESS | 2 DATA_SHORTFALL | 3 TECHNICAL_FAILURE | 4 INVALID_OUTPUT | 10 preflight | 11 vLLM/model
#SBATCH --job-name=benchson-full
#SBATCH --partition=main
#SBATCH --account=4th-year-proj
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/benchson-full-%j.out
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
# ---- which chunk (explicit; nothing is inferred)
N_CHUNKS=8
TOTALS="create=1334,fix=1333,modify=1333"
CHUNK_INDEX="${CHUNK_INDEX:-}"
[[ "$CHUNK_INDEX" =~ ^[1-8]$ ]] || fail "CHUNK_INDEX must be 1..8"
IDX2=$(printf '%02d' "$CHUNK_INDEX")
PARENT_CHUNK="${PARENT_CHUNK:-}"
if [ "$CHUNK_INDEX" -eq 1 ]; then
    [ -z "$PARENT_CHUNK" ] || fail "chunk 1 has no parent"
else
    [[ "$PARENT_CHUNK" =~ ^c[0-9]{2}_[0-9]+-r[0-9]+$ ]] || fail "PARENT_CHUNK must be the directory name of the previous chunk (c<NN>_<job>-r<n>)"
    PARENT_NUM=${PARENT_CHUNK:1:2}
    [ "$((10#$PARENT_NUM))" -eq $((CHUNK_INDEX - 1)) ] || fail "PARENT_CHUNK is chunk $((10#$PARENT_NUM)); chunk $CHUNK_INDEX needs chunk $((CHUNK_INDEX - 1))"
fi
export RUN_NAME="full_c${IDX2}_${RUN_ID}"
VLLM_ENV="${VLLM_ENV:-/storage/cluster/vLLM2}"
if [ "$MODE" = local ]; then
    BENCHSON_PYTHON="${BENCHSON_PYTHON:-.venv/bin/python}"
else
    BENCHSON_PYTHON="${BENCHSON_PYTHON:-$HOME/.conda/envs/benchson/bin/python}"
fi
# Operational limits only (docs/full_run_chunking.md): a hard cap on model calls and on the generation wall clock per chunk.
MAX_LLM_CALLS="${MAX_LLM_CALLS:-3000}"           # hard maximum 4000
GEN_TIMEOUT_SEC="${GEN_TIMEOUT_SEC:-5400}"       # generation only (after download and vLLM startup); at most 7200
MAX_MODEL_LEN=8192                               # vLLM --max-model-len AND the context guard's window (must be equal)
CONFIG="configs/scale_only_vllm_granite.json"
FULL_ROOT="data_scale_only/full"
CHUNKS_ROOT="$FULL_ROOT/chunks"
OUT_DIR="$CHUNKS_ROOT/c${IDX2}_${RUN_ID}"
CALLS_LOG="$FULL_ROOT/c${IDX2}_${RUN_ID}_calls.jsonl"
REPORT_JSON="$FULL_ROOT/c${IDX2}_${RUN_ID}_report.json"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"           # optional persistent model cache; see docs/full_run_chunking.md
VLLM_PYTHON="${VLLM_ENV}/bin/python"
VLLM_BIN="${VLLM_ENV}/bin/vllm"
export PATH="${VLLM_ENV}/bin:$PATH"

echo "=== Preflight ($MODE) job=$JOB attempt=$ATTEMPT run=$RUN_ID ==="
BRANCH=$(git branch --show-current) || fail "cannot identify branch"
[ "$BRANCH" = improve-data-generation ] || fail "expected improve-data-generation; found $BRANCH"
[ -x "$BENCHSON_PYTHON" ] || fail "benchson python not found: $BENCHSON_PYTHON"
[ -f "$CONFIG" ] || fail "config not found: $CONFIG"
"$BENCHSON_PYTHON" -c 'import deepdiff, jsonschema' || fail "benchson python lacks deepdiff/jsonschema"
QUOTA=$("$BENCHSON_PYTHON" src/chunked_run.py increments --totals "$TOTALS" --chunks "$N_CHUNKS" | "$BENCHSON_PYTHON" -c '
import json
import sys

rows = json.load(sys.stdin)["chunks"]
print(",".join(f"{task}={rows[int(sys.argv[1]) - 1][task]}" for task in ("create", "fix", "modify")))
' "$CHUNK_INDEX") || fail "cannot compute the chunk quota"
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
POOL_REPORT=$("$BENCHSON_PYTHON" src/build_benchmark.py --config "$CONFIG" \
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
totals = {k: int(v) for k, v in (item.split("=") for item in sys.argv[2].split(","))}
for name, row in sorted(tasks.items()):   # each task on its own: the FULL target, enough reference slots, feasible
    if row["target"] != totals[name] or row["upper_bound_if_every_attempt_succeeds"] < row["target"] \
            or not row["feasible_at_assumed_acceptance"]:
        sys.exit(f"schema pool cannot supply the full {totals[name]} {name} examples")
' "$ANCHOR_SHA256" "$TOTALS" || fail "pool does not meet the full-run requirements"
if [ -n "$PARENT_CHUNK" ]; then
    # The whole chain up to the parent is re-validated (hashes of every file, one pool and config, global per-schema cap,
    # duplicates, leakage, exact cumulative totals) before any GPU time is spent.
    "$BENCHSON_PYTHON" - "$CHUNKS_ROOT/$PARENT_CHUNK" "$ANCHOR" "$BASELINE" "$((CHUNK_INDEX - 1))" "$N_CHUNKS" <<'PYPARENT' \
        || fail "parent chunk missing or not valid: $CHUNKS_ROOT/$PARENT_CHUNK"
import sys

sys.path.insert(0, "src")
import chunked_run

res = chunked_run.validate_chunk(sys.argv[1], sys.argv[2], sys.argv[3])
if res["code"] != 0:
    sys.exit("parent chain does not validate: " + "; ".join((res["technical"] + res["invalid"] + res["shortfall"])[:3]))
if (res["chunk_index"], res["n_chunks"]) != (int(sys.argv[4]), int(sys.argv[5])):
    sys.exit("parent is not the expected chunk of the 8-chunk plan")
PYPARENT
fi
if [ -n "$MODEL_CACHE_DIR" ]; then
    # Opt-in persistent cache. The script never creates it and never deletes anything in it; it only lets
    # huggingface_hub add the pinned revision inside. Whether the location is APPROVED for this is the user's decision.
    "$BENCHSON_PYTHON" - "$MODEL_CACHE_DIR" <<'PYCACHE' || fail "MODEL_CACHE_DIR is not a safe, existing, private directory owned by you: $MODEL_CACHE_DIR"
import os
import stat
import sys

path = sys.argv[1]
if not os.path.isabs(path) or os.path.islink(path) or not os.path.isdir(path):
    sys.exit("must be an absolute path to an existing real directory (it is never created or followed through a symlink)")
info = os.stat(path)
if info.st_uid != os.getuid() or not os.access(path, os.W_OK):
    sys.exit("must be owned by you and writable")
if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
    sys.exit("must not be group- or world-writable")
PYCACHE
fi

[ -x "$VLLM_PYTHON" ] && [ -x "$VLLM_BIN" ] || fail "vLLM environment not found: $VLLM_ENV"
for tool in ninja setsid curl timeout; do
    command -v "$tool" >/dev/null || fail "$tool is not on PATH"
done
"$BENCHSON_PYTHON" -c 'import openai' || fail "benchson python lacks openai"
"$VLLM_PYTHON" -c 'import huggingface_hub, vllm' || fail "vLLM python lacks huggingface_hub/vllm"
if [ "$MODE" = cluster ]; then
    printf '%s\n' "$POOL_REPORT"
    echo "Cluster CPU preflight passed."
    exit 0
fi

[[ "$MAX_LLM_CALLS" =~ ^[1-9][0-9]*$ ]] && [ "$MAX_LLM_CALLS" -le 4000 ] || fail "MAX_LLM_CALLS must be 1-4000"
[[ "$GEN_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] && [ "$GEN_TIMEOUT_SEC" -le 7200 ] || fail "GEN_TIMEOUT_SEC must be 1-7200"
mkdir -p logs "$CHUNKS_ROOT" || fail "cannot create output directories"
for path in "$OUT_DIR" "$CALLS_LOG" "$REPORT_JSON" "logs/full-vllm-${RUN_ID}.log" "logs/full-pool-check-${RUN_ID}.json"; do
    [[ ! -e "$path" && ! -L "$path" ]] || fail "job path already exists: $path"
done
printf '%s\n' "$POOL_REPORT" > "logs/full-pool-check-${RUN_ID}.json" || fail "cannot write pool report"

# SCRATCH_DIR, when provided, is an existing parent directory only.
if [ -z "${SCRATCH_DIR:-}" ]; then
    SCRATCH_DIR="/scratch/${USER}"
    mkdir -p "$SCRATCH_DIR" || fail "cannot create scratch parent"
fi
SCRATCH_PARENT=$(cd -- "$SCRATCH_DIR" && pwd -P) || fail "invalid scratch parent"
[[ "$SCRATCH_PARENT" != / && -O "$SCRATCH_PARENT" && -w "$SCRATCH_PARENT" ]] || fail "scratch parent must be owned and writable"
FREE_KB=$(df -Pk "$SCRATCH_PARENT" | awk 'NR==2 {print $4}') || fail "cannot check scratch space"
NEED_KB=94371840   # 90 GiB for the Granite weights when they are downloaded into scratch
[ -z "$MODEL_CACHE_DIR" ] || NEED_KB=20971520   # 20 GiB (vLLM caches, tmp) when the weights live in MODEL_CACHE_DIR
[[ "$FREE_KB" =~ ^[0-9]+$ ]] && [ "$FREE_KB" -ge "$NEED_KB" ] || fail "scratch requires $((NEED_KB / 1048576)) GiB free"
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
    if [[ -n "$SCRATCH_INODE" && "$JOB_SCRATCH" == "$SCRATCH_PARENT"/benchson-full-"$RUN_ID".* &&
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

JOB_SCRATCH=$(mktemp -d "$SCRATCH_PARENT/benchson-full-$RUN_ID.XXXXXXXX") || fail "cannot create private scratch"
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
import json
import os
import sys

from huggingface_hub import snapshot_download

revision = os.environ["GRANITE_REVISION"]
cache = os.environ.get("MODEL_CACHE_DIR") or os.environ["HF_HOME"] + "/hub"
kw = dict(repo_id=os.environ["GRANITE_MODEL_ID"], revision=revision, cache_dir=cache)


def problems(path):
    out = []
    if os.path.basename(path.rstrip("/")) != revision:
        out.append("snapshot directory is not the pinned revision")
    if not os.path.isfile(os.path.join(path, "config.json")):
        out.append("config.json missing")
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        shards = set(json.load(open(index))["weight_map"].values())
    else:
        shards = {n for n in os.listdir(path) if n.endswith(".safetensors")}
        if not shards:
            out.append("no safetensors weights")
    for name in sorted(shards):
        f = os.path.join(path, name)
        if not os.path.exists(f) or os.path.getsize(f) == 0:   # exists() follows the cache symlink into blobs/
            out.append(f"weight file missing or empty: {name}")
    return out


path = None
try:        # already complete in the cache: no network needed
    path = snapshot_download(local_files_only=True, **kw)
    if problems(path):
        path = None
except Exception:
    path = None
if path is None:   # downloads, or resumes an interrupted download of exactly this revision
    path = snapshot_download(**kw)
    bad = problems(path)
    if bad:
        sys.exit("model snapshot incomplete: " + "; ".join(bad[:3]))
print(path)
PY
WORK_PID=$!
wait "$WORK_PID"
DOWNLOAD_RC=$?
WORK_PID=""
[ "$DOWNLOAD_RC" -eq 0 ] || { echo "model download failed"; exit 11; }
MODEL_PATH=$(cat "$JOB_SCRATCH/model-path")
[ -d "$MODEL_PATH" ] || { echo "invalid model path"; exit 11; }
echo "Model ready at: $MODEL_PATH"

echo "=== Starting vLLM on 127.0.0.1:${SERVER_PORT} ==="
# VLLM_API_KEY (exported above) is read by vLLM itself, so the key never appears in the process list.
setsid "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$GRANITE_MODEL_ID" \
    --host 127.0.0.1 --port "$SERVER_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    > "logs/full-vllm-${RUN_ID}.log" 2>&1 &
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
        echo "vLLM process exited (see logs/full-vllm-${RUN_ID}.log)."
        exit 11
    fi
    sleep 10
done
[ "$READY" -eq 1 ] || { echo "vLLM did not become ready (see logs/full-vllm-${RUN_ID}.log)."; exit 11; }

# ---- 2. One chunk ---------------------------------------------------------------------------------
echo "=== Running chunk ${CHUNK_INDEX}/${N_CHUNKS} (quota ${QUOTA}; max ${MAX_LLM_CALLS} calls, ${GEN_TIMEOUT_SEC}s) ==="
PARENT_ARGS=()
[ -z "$PARENT_CHUNK" ] || PARENT_ARGS=(--parent-chunk "$CHUNKS_ROOT/$PARENT_CHUNK")
GEN_START=$(date +%s)
setsid timeout --kill-after=60 "$GEN_TIMEOUT_SEC" "$BENCHSON_PYTHON" src/build_benchmark.py \
    --config "$CONFIG" \
    --chunks "$N_CHUNKS" --chunk-index "$CHUNK_INDEX" ${PARENT_ARGS[@]+"${PARENT_ARGS[@]}"} \
    --out-dir "$OUT_DIR" \
    --context-guard "$MAX_MODEL_LEN" \
    --max-llm-calls "$MAX_LLM_CALLS" \
    --smoke-openai-timeout 180 \
    --record-calls "$CALLS_LOG" &
WORK_PID=$!
wait "$WORK_PID"
BUILD_RC=$?
GEN_END=$(date +%s)
WORK_PID=""
echo "build_benchmark.py exit code: $BUILD_RC (2 = chunk quota not met, no checkpoint; 124 = generation wall-clock limit)"

# ---- 3. Validate the chunk and its whole chain; ONLY a fully valid chunk gets its success marker (chunk_success.json) ------------------------------------------------------
echo "=== Validating chunk ${CHUNK_INDEX} and the chain before it ==="
"$BENCHSON_PYTHON" src/chunked_run.py seal "$OUT_DIR" \
    --anchor-file "$ANCHOR" --reserved-file "$BASELINE" --require-context-guard "$MAX_MODEL_LEN" \
    --calls-log "$CALLS_LOG" --build-exit-code "$BUILD_RC"
CHECK_RC=$?

# ---- 4. Metrics report (read-only; never changes the verdict) -----------------------------------
"$BENCHSON_PYTHON" src/pilot_report.py --out-dir "$OUT_DIR" --targets "$QUOTA" --calls-log "$CALLS_LOG" \
    --build-exit-code "$BUILD_RC" --check-exit-code "$CHECK_RC" --gen-start-epoch "$GEN_START" --gen-end-epoch "$GEN_END" \
    --max-llm-calls "$MAX_LLM_CALLS" --write "$REPORT_JSON" || echo "report could not be written (verdict unaffected)"
echo "=== Chunk ${CHUNK_INDEX} finished: exit code ${CHECK_RC} (0 success, 2 shortfall, 3 technical, 4 invalid); report: ${REPORT_JSON} ==="
exit "$CHECK_RC"

#!/bin/bash
# Overnight chain link for chunk CHUNK_INDEX (2..8) of the full Scale Only run. This job does NOT generate anything itself:
# it (1) refuses unless exactly one fully valid, sealed parent chunk exists, (2) refuses to build a chunk that already
# exists as a sealed result of ANOTHER job, and only then (3) hands over, unchanged, to slurm/full_chunk_granite_v2.sbatch
# (the same script that built chunk 1: same prompts, sampling, model revision, pool, reuse limits and targets), which
# validates the parent chain once more before any GPU time is used. Submit it with slurm/submit_chunk_chain.sh, never by hand
# without a dependency. Slurm's afterok is only the first guard: it is NOT trusted to discover the parent or to judge success.
#
# Requeue/preemption: the job is re-run under the same job ID with a higher SLURM_RESTART_COUNT; the parent is looked up again
# (same answer), a new output directory <chunk>_<job>-r<attempt> is used by the chunk script, and earlier attempts stay untouched.
# If an earlier attempt of THIS job already produced a sealed chunk (preempted after sealing), nothing is rebuilt: exit 0.
#
# The #SBATCH lines below must equal those of slurm/full_chunk_granite_v2.sbatch (tests/test_chunk_chain.py checks this);
# the job name is overridden per chunk by the submitter (benchson-full-cNN).
#   exit codes: 0 chunk sealed (or already sealed by this job) | 10 refused before any work | otherwise the chunk script's
#   (2 DATA_SHORTFALL | 3 TECHNICAL_FAILURE | 4 INVALID_OUTPUT | 11 vLLM/model)
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
fail() { echo "CHAIN REFUSED: $*" >&2; exit 10; }

[ "$#" -eq 0 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] || fail "this script only runs as a Slurm job (use slurm/submit_chunk_chain.sh)"
cd "$SLURM_SUBMIT_DIR" || fail "cannot enter Slurm submission directory"
JOB="$SLURM_JOB_ID"
[[ "$JOB" =~ ^[1-9][0-9]*$ ]] || fail "invalid job ID"
ATTEMPT="${SLURM_RESTART_COUNT:-0}"
[[ "$ATTEMPT" =~ ^(0|[1-9][0-9]{0,5})$ ]] || fail "invalid SLURM_RESTART_COUNT"

# The plan (identical to the chunk script; checked against it by the tests).
N_CHUNKS=8
TOTALS="create=1334,fix=1333,modify=1333"
PUBLISHED_REVISION=7962b0947f7d96eb3a0e6b9b928d939c7f3061a6
PUBLISHED_SHA12="${PUBLISHED_REVISION:0:12}"
CHUNK_SCRIPT="slurm/full_chunk_granite_v2.sbatch"
CHUNKS_ROOT="data_scale_only/full/chunks"
BASELINE="data_published/${PUBLISHED_SHA12}/reserved_schemas.json"
ANCHOR="data_published/${PUBLISHED_SHA12}/published_train_schemas.json"

CHUNK_INDEX="${CHUNK_INDEX:-}"
[[ "$CHUNK_INDEX" =~ ^[2-8]$ ]] || fail "CHUNK_INDEX must be 2..8 (chunk 1 is submitted directly and has no parent)"
[ "$CHUNK_INDEX" -le "$N_CHUNKS" ] || fail "CHUNK_INDEX is beyond the plan"
IDX2=$(printf '%02d' "$CHUNK_INDEX")
PARENT_INDEX=$((CHUNK_INDEX - 1))
PARENT2=$(printf '%02d' "$PARENT_INDEX")
# A PARENT_CHUNK left over in the submitting shell must never choose the parent: only validated discovery (or an explicit,
# validated PARENT_CHUNK_PIN naming one of several valid sealed candidates) can.
unset PARENT_CHUNK
PIN="${PARENT_CHUNK_PIN:-}"
if [ -n "$PIN" ]; then
    [[ "$PIN" =~ ^c${PARENT2}_[0-9]+-r[0-9]+$ ]] || fail "PARENT_CHUNK_PIN must name a chunk-${PARENT_INDEX} directory (c${PARENT2}_<job>-r<n>)"
fi
BENCHSON_PYTHON="${BENCHSON_PYTHON:-$HOME/.conda/envs/benchson/bin/python}"
echo "=== Chain link: chunk $CHUNK_INDEX job=$JOB attempt=$ATTEMPT ==="
BRANCH=$(git branch --show-current) || fail "cannot identify branch"
[ "$BRANCH" = improve-data-generation ] || fail "expected improve-data-generation; found $BRANCH"
[ -x "$BENCHSON_PYTHON" ] || fail "benchson python not found: $BENCHSON_PYTHON"
[ -f "$CHUNK_SCRIPT" ] || fail "chunk script not found: $CHUNK_SCRIPT"
[ -f src/chunked_run.py ] && [ -f "$BASELINE" ] && [ -f "$ANCHOR" ] || fail "baseline, anchor or chunked_run.py missing"
[ -d "$CHUNKS_ROOT" ] || fail "no chunk directory: $CHUNKS_ROOT"
[ ! -L "$CHUNKS_ROOT" ] || fail "chunk directory must not be a symlink: $CHUNKS_ROOT"

resolve() {   # $1 index, optional $2 pin; prints the directory name on success; the full report goes to stderr
    local pin_args=()
    [ -z "${2:-}" ] || pin_args=(--pin "$2")
    "$BENCHSON_PYTHON" src/chunked_run.py resolve --root "$CHUNKS_ROOT" --index "$1" --anchor-file "$ANCHOR" \
        --reserved-file "$BASELINE" --chunks "$N_CHUNKS" --totals "$TOTALS" ${pin_args[@]+"${pin_args[@]}"}
}

# ---- 1. This chunk must not already exist as a sealed result.
OWN=$(resolve "$CHUNK_INDEX"); OWN_RC=$?
case "$OWN_RC" in
    0)  if [[ "$OWN" =~ ^c${IDX2}_${JOB}-r[0-9]+$ ]]; then
            echo "chunk $CHUNK_INDEX was already sealed by this job ($OWN, an earlier attempt): nothing to rebuild."
            exit 0
        fi
        fail "chunk $CHUNK_INDEX already exists as a sealed result of another job ($OWN); it is never rebuilt or replaced" ;;
    7)  fail "chunk $CHUNK_INDEX already exists several times as a sealed result; resolve this by hand" ;;
    6)  ;;
    *)  fail "could not examine existing chunk-$CHUNK_INDEX directories (resolve exit $OWN_RC)" ;;
esac

# ---- 2. Exactly one valid sealed parent (independent of Slurm's dependency state).
PARENT=$(resolve "$PARENT_INDEX" "$PIN"); PARENT_RC=$?
case "$PARENT_RC" in
    0) ;;
    6) fail "no fully valid, sealed chunk $PARENT_INDEX found under $CHUNKS_ROOT${PIN:+ (pinned: $PIN)}; the chain stops here" ;;
    7) fail "several valid sealed chunk-$PARENT_INDEX directories exist; choose one explicitly with PARENT_CHUNK_PIN=<dir> (see docs/overnight_chunk_chain.md)" ;;
    *) fail "could not validate chunk $PARENT_INDEX (resolve exit $PARENT_RC)" ;;
esac
[[ "$PARENT" =~ ^c${PARENT2}_[0-9]+-r[0-9]+$ ]] || fail "unexpected parent name: $PARENT"
echo "parent chunk: $PARENT (validated: success marker, hash chain, totals, global cap, duplicates, leakage)"

# ---- 3. Hand over unchanged. exec: the chunk script becomes this job's batch script and keeps its own signal handling,
# scratch cleanup and requeue isolation. It validates the parent chain again before any GPU time is used.
export CHUNK_INDEX
export PARENT_CHUNK="$PARENT"
exec bash "$CHUNK_SCRIPT"

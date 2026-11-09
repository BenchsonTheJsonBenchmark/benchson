#!/bin/bash
# Submits chunks FROM..TO (default 2..8) of the full Scale Only run as a Slurm dependency chain, run from the repository root on
# the BGU login node. DRY RUN unless --submit is given. Nothing is cancelled, deleted or overwritten by this script.
#
#   bash slurm/submit_chunk_chain.sh --after 24044333                 # print the plan (chunk 1 = running job 24044333)
#   bash slurm/submit_chunk_chain.sh --after 24044333 --submit        # submit chunks 2..8
#   bash slurm/submit_chunk_chain.sh --after none --from 4 --submit   # continue after a manually verified chunk 3 (no dependency;
#                                                                     # the job itself refuses unless a sealed chunk 3 exists)
# Every link is `sbatch --dependency=afterok:<previous> --kill-on-invalid-dep=yes` of slurm/full_chunk_chain_job.sbatch. A failed,
# cancelled or timed-out link therefore cancels everything behind it, and each link additionally validates its parent chunk itself
# (success marker, hash chain, totals, cap, duplicates, leakage) before doing any work. Job IDs are recorded in
# logs/full-chain-<after>-c<from>-c<to>.jobs (created once, never overwritten).
set -uo pipefail
umask 077
die() { echo "REFUSED: $*" >&2; exit 10; }

AFTER=""; FROM=2; TO=8; DO_SUBMIT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --after) [ $# -ge 2 ] || die "--after needs a job ID or 'none'"; AFTER="$2"; shift 2 ;;
        --from) [ $# -ge 2 ] || die "--from needs a chunk number"; FROM="$2"; shift 2 ;;
        --to) [ $# -ge 2 ] || die "--to needs a chunk number"; TO="$2"; shift 2 ;;
        --submit) DO_SUBMIT=1; shift ;;
        *) die "unknown argument: $1 (see the header of this script)" ;;
    esac
done
[[ "$AFTER" =~ ^([1-9][0-9]*|none)$ ]] || die "--after must be the numeric job ID of the previous chunk's job, or 'none'"
[[ "$FROM" =~ ^[2-8]$ && "$TO" =~ ^[2-8]$ ]] || die "--from/--to must be 2..8 (chunk 1 is submitted directly)"
[ "$FROM" -le "$TO" ] || die "--from must not exceed --to"

cd "$(dirname "$0")/.." || die "cannot enter the checkout"
WRAPPER="slurm/full_chunk_chain_job.sbatch"
[ -f "$WRAPPER" ] && [ -f slurm/full_chunk_granite_v2.sbatch ] || die "chain or chunk script missing"
[ "$(git branch --show-current)" = improve-data-generation ] || die "expected branch improve-data-generation"
command -v sbatch >/dev/null && command -v squeue >/dev/null || die "sbatch/squeue not found (run this on the login node)"
ME="${USER:-$(id -un)}"

if [ "$AFTER" != none ]; then
    # afterok on a job that has already left the queue would be treated by Slurm as satisfied whatever its outcome,
    # so the previous job must still be active.
    LINE=$(squeue -h -j "$AFTER" -o '%i|%j|%T') || die "squeue could not look up job $AFTER"
    IFS='|' read -r JID JNAME JSTATE <<<"$LINE"
    [ "${JID:-}" = "$AFTER" ] || die "job $AFTER is not in the queue (finished or unknown). If its chunk was sealed, verify it and use --after none --from <next chunk>"
    [[ "$JNAME" =~ ^benchson-full(-c[0-9]{2})?$ ]] || die "job $AFTER is named '$JNAME', not a benchson-full chunk job"
    case "$JSTATE" in PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED) ;; *) die "job $AFTER is in state $JSTATE" ;; esac
fi
ACTIVE=$(squeue -h -u "$ME" -o '%j') || die "squeue could not list your jobs"
for K in $(seq "$FROM" "$TO"); do
    NAME="benchson-full-c$(printf '%02d' "$K")"
    if grep -qxF "$NAME" <<<"$ACTIVE"; then die "a job named $NAME is already queued or running; refusing to submit a duplicate chunk"; fi
done

echo "Plan (chunks $FROM..$TO, after: $AFTER):"
PREV="$AFTER"
for K in $(seq "$FROM" "$TO"); do
    DEP="--dependency=afterok:<${PREV}>"; [ "$PREV" != none ] || DEP="(no dependency)"
    echo "  chunk $K: sbatch --parsable $DEP --kill-on-invalid-dep=yes --job-name=benchson-full-c$(printf '%02d' "$K") --export=ALL,CHUNK_INDEX=$K $WRAPPER"
    PREV="<chunk $K job>"
done
[ "$DO_SUBMIT" -eq 1 ] || { echo "Dry run only. Add --submit to submit."; exit 0; }

mkdir -p logs || die "cannot create logs/"
RECORD="logs/full-chain-${AFTER}-c$(printf '%02d' "$FROM")-c$(printf '%02d' "$TO").jobs"
( set -o noclobber; : > "$RECORD" ) 2>/dev/null || die "$RECORD already exists: this chain was submitted before (check squeue/sacct)"
PREV="$AFTER"
for K in $(seq "$FROM" "$TO"); do
    ARGS=(--parsable --kill-on-invalid-dep=yes "--job-name=benchson-full-c$(printf '%02d' "$K")" "--export=ALL,CHUNK_INDEX=$K")
    [ "$PREV" = none ] || ARGS+=("--dependency=afterok:$PREV")
    # leftovers in this shell must never reach the jobs (the chain job also ignores PARENT_CHUNK itself)
    OUT=$(env -u PARENT_CHUNK -u PARENT_CHUNK_PIN sbatch "${ARGS[@]}" "$WRAPPER") || {
        echo "sbatch failed for chunk $K; the chain stops here. Already submitted (see $RECORD) stays queued and is safe." >&2; exit 11; }
    JOBID="${OUT%%;*}"
    [[ "$JOBID" =~ ^[1-9][0-9]*$ ]] || { echo "unexpected sbatch output for chunk $K: '$OUT'; the chain stops here (see $RECORD)." >&2; exit 11; }
    echo "chunk $K job=$JOBID after=$PREV" | tee -a "$RECORD"
    PREV="$JOBID"
done
echo "Submitted chunks $FROM..$TO. Record: $RECORD. Monitor with the commands in docs/overnight_chunk_chain.md."

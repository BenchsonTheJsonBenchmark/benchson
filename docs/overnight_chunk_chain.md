# Overnight execution of chunks 2-8 (Slurm dependency chain)

Chunk 1 is submitted by hand (job 24044333, `slurm/full_chunk_granite_v2.sbatch`) and is not touched by anything below.
Chunks 2-8 run as one `afterok` chain of `slurm/full_chunk_chain_job.sbatch` jobs, created by `slurm/submit_chunk_chain.sh`.
Nothing about generation changes: prompts, sampling (temperature 0.7, max_tokens 4096, top_p 0.95), model revision, anchored
pool, Controlled Schema Reuse (cap 3), acceptance rules and train/test exclusions are those of the chunk script, which every
link hands over to unchanged. The eight quotas stay exactly 1,334 Create / 1,333 Fix / 1,333 Modify (`chunked_run.py increments`).

## Design: three independent guards per link

1. **Slurm:** `--dependency=afterok:<previous job> --kill-on-invalid-dep=yes`. If a job fails, is cancelled or times out, every job
   behind it is cancelled and never starts. `afterok` is only this first guard: it knows exit codes, not data.
2. **The link itself (`full_chunk_chain_job.sbatch`, before any GPU time):** `chunked_run.py resolve` fully validates every
   directory `c<NN>_<job>-r<attempt>` of the parent chunk (valid `chunk_success.json`, every file hash, hash chain to chunk 1,
   one pool/config/plan, global cap of 3, duplicates, train/test leakage, exact cumulative totals) and requires **exactly one**
   valid, sealed candidate. Partial, preempted, unsealed, failed, tampered, symlinked or other-plan directories are ignored and left
   untouched. None found -> exit 10 (chain stops). Several valid ones -> exit 10 (never guessed; see "Duplicate successful
   attempts"). A `PARENT_CHUNK` inherited from the submitting shell is discarded.
3. **The chunk script (`full_chunk_granite_v2.sbatch`, after `exec`):** re-validates the parent chain again before the GPU is used,
   builds the chunk in a new directory, and only a chunk that passes every final validation gets its `chunk_success.json`
   (`chunked_run.py seal`; it records the checker result and the calls-log validation). Its exit code is the checker's
   (0 only for a sealed chunk) and is passed through unchanged, so a failed or unsealed chunk never satisfies `afterok`.

Before building, the link also refuses if chunk K already exists as a sealed result of **another** job. If an earlier attempt of
**this** job already sealed chunk K (preempted after sealing), it exits 0 without rebuilding, so `afterok` is satisfied and the
next link resolves that sealed directory.

## Failure behaviour

| Situation | What happens |
|---|---|
| Chunk K ends with exit 2/3/4/11 (shortfall, technical, invalid, vLLM) | no marker; links K+1.. are cancelled (`DependencyNeverSatisfied`); output kept for inspection |
| Job K times out or is cancelled | same: dependents cancelled; partial directory kept, never a parent |
| Job K is preempted and requeued | same job ID, `SLURM_RESTART_COUNT` +1: a new directory `c<KK>_<job>-r<n>` per attempt; earlier attempts untouched; dependents keep waiting |
| Job K preempted after it sealed | the requeued attempt finds its own sealed result and exits 0 without rebuilding |
| Parent missing/unsealed/tampered at link start | exit 10 before any work; chain stops |
| A second chain is submitted by mistake | the submitter refuses if a `benchson-full-cNN` job is queued or a record file exists; a link refuses if the chunk is already sealed by another job |
| sbatch fails while submitting | the chain stops there; jobs already submitted stay queued and are safe |

## Duplicate successful attempts

Only possible if a chunk job was rerun after sealing outside this chain (for example chunk 1 requeued after sealing: chunk 1 uses the
plain script). The next link then refuses with "several valid sealed chunk-K directories". Pick one **explicitly** after looking at both
(`chunked_run.py status`), then resubmit that link:
`PARENT_CHUNK_PIN=c01_24044333-r0 sbatch --export=ALL,CHUNK_INDEX=2 --job-name=benchson-full-c02 slurm/full_chunk_chain_job.sbatch`
The pin must name a candidate that validates; anything else is refused. Nothing is deleted or renamed; the other directory is simply never used.

## Deploying (nothing is submitted by these steps)

Chunk 1 imports `src/chunked_run.py` again when it seals. Use `rsync` (writes a temporary file and renames it atomically), never `scp`
onto an existing file, and do not sync while chunk 1 is in its last minutes if you can avoid it.

```bash
# on your Mac, from the repository root (review the dry run first)
FILES="src/chunked_run.py slurm/full_chunk_chain_job.sbatch slurm/submit_chunk_chain.sh tests/test_chunk_chain.py docs/overnight_chunk_chain.md docs/full_run_chunking.md"
rsync -avnR --checksum $FILES <BGU_LOGIN_HOST>:<CHECKOUT>/     # dry run (-R keeps the relative paths)
rsync -avR --checksum $FILES <BGU_LOGIN_HOST>:<CHECKOUT>/      # real copy; no other file is changed

# on the BGU login node, in <CHECKOUT> (read-only checks)
git branch --show-current                                      # improve-data-generation
grep -c load_seal src/data_generation/chunking.py              # >= 1: the sealing version that chunk 1 uses
BP=$HOME/.conda/envs/benchson/bin/python
$BP -m py_compile src/chunked_run.py && bash -n slurm/full_chunk_chain_job.sbatch && bash -n slurm/submit_chunk_chain.sh && echo syntax-ok
$BP src/chunked_run.py increments --totals create=1334,fix=1333,modify=1333 --chunks 8 | tail -8     # sum: 1334 / 1333 / 1333
$BP src/chunked_run.py resolve --root data_scale_only/full/chunks --index 1 \
  --anchor-file data_published/7962b0947f7d/published_train_schemas.json --reserved-file data_published/7962b0947f7d/reserved_schemas.json \
  --chunks 8 --totals create=1334,fix=1333,modify=1333 ; echo "exit=$?"     # 6 while chunk 1 is unsealed (read-only)
```

## Submitting (once, while job 24044333 is still queued or running)

```bash
bash slurm/submit_chunk_chain.sh --after 24044333            # dry run: prints the 7 sbatch commands
bash slurm/submit_chunk_chain.sh --after 24044333 --submit   # submits chunks 2..8, writes logs/full-chain-24044333-c02-c08.jobs
```

The previous job must still be in the queue (a finished job would count as satisfied for `afterok` whatever its result). If chunk K-1
already finished and you have verified that it is sealed, continue with `--after none --from K --submit`: no dependency, and the link
itself still requires a valid sealed parent.

## Monitoring

```bash
squeue -u $USER -o "%.10i %.20j %.9T %.10M %.18R %E"        # %E shows the dependency of each link
sacct -D -j 24044333,<ids from logs/full-chain-*.jobs> --format=JobID%20,JobName%20,State,ExitCode,Elapsed,Start,End     # -D lists every requeued attempt
BP=$HOME/.conda/envs/benchson/bin/python
$BP src/chunked_run.py status --root data_scale_only/full/chunks      # per directory: sealed / checkpointed_unsealed / incomplete
tail -n 40 logs/benchson-full-<jobid>.out
grep -h "CHUNK RESULT\|CHAIN REFUSED\|parent chunk:" logs/benchson-full-*.out
```
A healthy end state: seven chunk directories `sealed`, `CHUNK RESULT: SUCCESS ... sealed=True` in every log.

## After a failure

Read the job log and the chunk's `generation_summary.json`; never edit or remove chunk directories. To retry chunk K (K >= 2):
`sbatch --export=ALL,CHUNK_INDEX=K --job-name=benchson-full-cKK slurm/full_chunk_chain_job.sbatch` (it finds the sealed parent itself and
refuses if chunk K is already sealed), then `bash slurm/submit_chunk_chain.sh --after <that job> --from K+1 --submit`. For chunk 1:
`CHUNK_INDEX=1 sbatch slurm/full_chunk_granite_v2.sbatch`, then the chain with `--after <that job>`. Stop the whole chain with `scancel <job ids>`
(only the jobs listed in the record file); the outputs stay.

## Remaining limits

- Requires the previous job still queued when the chain is submitted (see above); otherwise use `--after none`.
- The orchestrator cannot repair a failed chunk; it can only refuse to continue. Each link re-downloads the model (as chunk 1 does).
- Requeue behaviour depends on the cluster's preemption policy (jobs are requeued under the same ID, as observed for the pilot).
- Duplicate chunk-1 successes need the manual pin described above.

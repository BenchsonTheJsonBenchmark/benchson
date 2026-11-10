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
bash slurm/submit_chunk_chain.sh --after 24044333 --submit   # submits chunks 2..8, writes logs/full-chain-24044333-c02-c08-<UTC time>.jobs
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

## First run, job 24044333: what went wrong, and how to continue

**Cause.** Generation was fine (166/166/166, exit 0, checkpoint written). The final step then crashed because the BGU checkout still had
OLD copies of `src/check_smoke_result.py` (compares an int with the per-task quota dict: `TypeError '>' not supported between 'int' and
'dict'`) and `src/pilot_report.py` (no `--targets`: `unrecognized arguments`). The reviewed checkout already handles both; the chunk
script had no way to notice that a stale file was deployed. Fixes in this repository:

- `chunked_run.py selfcheck` (run by the chunk script before any GPU time) feeds the checker and the report tool the chunk arguments and
  fails the job early if either cannot cope; `chunked_run.py fingerprint` prints the sha256 of every runtime file into the job log.
- Regression tests reproduce both failures, and check that every option the Slurm scripts pass to `pilot_report.py`/`chunked_run.py` exists.
- Chain records are now `logs/full-chain-<after>-c<from>-c<to>-<UTC time>.jobs` (one file per submission): the old record of the cancelled
  chain cannot block a new submission. Duplicate chains are still prevented by the queue check and by the jobs themselves.

**Recovering the finished Chunk 1 without regenerating anything** (`c01_24044333-r1`, one time, explicit, nothing is executed automatically).
`recover-seal` re-runs the complete normal validation and writes the normal success marker (marked `recovery`), only if all of this holds:
the directory name matches the job/attempt; `--confirm-recovery` repeats it; the manifest says `completed` for Slurm job 24044333 and the
chunk state exists; the Slurm log segment of attempt `r1` shows exactly one `build_benchmark.py exit code: 0`; the ORIGINAL calls log
`c01_24044333-r1_calls.jsonl` has as many calls as the manifest counted and none failed; the context guard ran for the 8,192 window; the
config file is byte-identical to the one the chunk was generated with (sha256 recorded in the state); and every example, hash, cap,
duplicate, leakage and total check passes. Anything missing or different: no marker, nothing changed. Existing files are never edited.

```bash
# after the sync of the files listed in the review, on the BGU login node in <CHECKOUT>
BP=$HOME/.conda/envs/benchson/bin/python
$BP src/chunked_run.py selfcheck && $BP src/chunked_run.py fingerprint > /tmp/fingerprint.cluster      # compare with the reviewed list
$BP src/chunked_run.py recover-seal data_scale_only/full/chunks/c01_24044333-r1 \
  --anchor-file data_published/7962b0947f7d/published_train_schemas.json --reserved-file data_published/7962b0947f7d/reserved_schemas.json \
  --calls-log data_scale_only/full/c01_24044333-r1_calls.jsonl --slurm-log logs/benchson-full-24044333.out \
  --config-file configs/scale_only_vllm_granite.json --require-context-guard 8192 --confirm-recovery c01_24044333-r1 ; echo "exit=$?"
# expected: CHUNK RECOVERY: SUCCESS ... sealed=True, exit=0
```

(The command above is the strict recovery. Chunk 1 contains three historical token-count timeouts, so it is refused until the explicit
policy below is added.)

Optional afterwards (writes a NEW file, never overwrites): the metrics report that failed:
`$BP src/pilot_report.py --out-dir data_scale_only/full/chunks/c01_24044333-r1 --targets create=166,fix=166,modify=166 --calls-log data_scale_only/full/c01_24044333-r1_calls.jsonl --build-exit-code 0 --check-exit-code 0 --write data_scale_only/full/c01_24044333-r1_report.json`

**Restarting chunks 2-8 (the cancelled jobs 24044864-24044870 are not reused).**
```bash
$BP src/chunked_run.py resolve --root data_scale_only/full/chunks --index 1 \
  --anchor-file data_published/7962b0947f7d/published_train_schemas.json --reserved-file data_published/7962b0947f7d/reserved_schemas.json \
  --chunks 8 --totals create=1334,fix=1333,modify=1333 ; echo "exit=$?"        # must print c01_24044333-r1 and exit 0
squeue -u $USER -o "%.10i %.20j %.9T"                                           # no benchson-full-cNN job may be queued
bash slurm/submit_chunk_chain.sh --after none --from 2                           # dry run
bash slurm/submit_chunk_chain.sh --after none --from 2 --submit
```
`--after none` is right because chunk 1 is finished and sealed; the first new link has no dependency but refuses unless a valid sealed chunk 1
exists. Directory `c01_24044333-r0` (preempted attempt, no state) is ignored automatically. If you regenerate chunk 1 instead of recovering it
(`CHUNK_INDEX=1 sbatch slurm/full_chunk_granite_v2.sbatch`), do NOT also recover r1: two sealed chunk-1 directories make the next link refuse
until you choose one with `PARENT_CHUNK_PIN`.

## Token-count timeouts: prevention for chunks 2-8, and the one-time historical exception for Chunk 1

**What happened (inference, not proof).** Chunk 1 recorded 3 `provider_error` events (`TimeoutError: timed out`) and 9 `worker_error`
failures (3 per attempt: create, fix, modify) on the attempts `schemas__sarif-external-property-file-2.1.0-rtm.3__a1`,
`schemas__jsonld__a1`, `schemas__sarif-external-property-file-2.1.0-rtm.0__a1`. No traceback was saved. The timeouts are inferred to come
from the `/tokenize` count (30 s) because no model call failed (the calls log has 1,159 calls, none with an error) and the count runs outside
the call recorder. This fits the evidence but is not proven.

**Prevention (applies to every new chunk).**
- A failed `/tokenize` request raises `TokenCountError` (kind `timeout` or `error`), never a model-call error and never counted as a model call.
- Only a timeout of the count request is retried: at most 3 tries in total, waiting 2 s and 5 s between them. The count request timeout is 60 s
  (was 30 s). Any other count error is not retried and is a technical failure (`token_count_error`).
- Model generation is unchanged: 180 s timeout, no SDK retries, same prompts, sampling and model revision.
- After the retries are used up the attempt fails with the stage `token_count_timeout`, goes through the same ledger path as any failed attempt
  (it counts in `attempts`, `failed_attempts` and `consecutive_failures`; schema suspension/retirement limits still apply), and writes no
  example, no sidecar and no duplicate registration. 3 exhausted counts in a row still stop the whole run (`ProviderFailing`).
- The checker tolerates `token_count_timeout` only as a warning, and only if ALL hold: at most 10 per chunk; at most 1% of all token counts
  (successful + timed out); the events and the failures name exactly the same attempts; the manifest's guard counter `count_timeouts` equals the
  number of events; no example or sidecar exists for those attempts. Anything else makes the chunk a TECHNICAL_FAILURE.
- A real model timeout or connection error, or any unexpected exception in a worker, stays a `provider_error` / `worker_error` and is fatal.
- The tolerated incidents are listed in the run reports (`generation_events.jsonl`, `generation_failures.jsonl`, `generation_summary.json`),
  in the manifest's `context_guard` counters and in the checker's `tolerated` result.

**One-time historical exception for `c01_24044333-r1` only.** It is not a general exemption for `TimeoutError`. It is accepted only when
`recover-seal` is given BOTH `--authorize-historical-timeouts historical-token-count-timeouts/1` and
`--historical-timeout-ids` with exactly the three attempt ids above (constants in `src/chunked_run.py`, `HISTORICAL_POLICY`). The recovery
refuses (no marker, nothing changed) unless all of the following hold, in addition to every normal recovery rule (directory name, confirmation,
completed manifest of job 24044333, exactly one `exit code: 0` in the Slurm segment of `r1`, the original calls log complete and error-free with
as many calls as the manifest counted, config hash, hash chain, caps, duplicates, train/test exclusions, exact totals):
1. the event log holds exactly 3 `provider_error` events, one per listed id, each with the detail exactly `TimeoutError: timed out`;
2. the failure log holds exactly 9 `worker_error` failures, create/fix/modify for each id, each with the detail exactly `timed out`;
3. nothing else technical exists (any other `provider_error`, `worker_error` or `token_count_error` blocks the recovery);
4. no example file and no metadata sidecar exists for the three attempts, and the saved state lists none;
5. the saved ledger still has a row for each of the three schemas with `failed_attempts >= 1` and `attempts >= accepted_references + failed_attempts`;
6. the context-guard counters satisfy `checked == calls in the log + overflow` and the log contains no failed call.

The success marker then records the policy id, the ids, the counts, the exact detail strings, the guard counters, the SHA-256 of the calls log, the
Slurm log, the manifest, the state and the report files (events, failures, schema ledger, summary), and the statement
`origin_status: inferred_not_proven`. Later validation (resolve, parent check, merge) takes the tolerance only from that verified marker, only for
that directory name and those ids; a marker with another policy, other ids or another chunk grants nothing, and a new incident added later to
the logs makes the chunk invalid again.

```bash
# after the sync of the files listed in the review, on the BGU login node in <CHECKOUT>
BP=$HOME/.conda/envs/benchson/bin/python
$BP src/chunked_run.py recover-seal data_scale_only/full/chunks/c01_24044333-r1 \
  --anchor-file data_published/7962b0947f7d/published_train_schemas.json --reserved-file data_published/7962b0947f7d/reserved_schemas.json \
  --calls-log data_scale_only/full/c01_24044333-r1_calls.jsonl --slurm-log logs/benchson-full-24044333.out \
  --config-file configs/scale_only_vllm_granite.json --require-context-guard 8192 --confirm-recovery c01_24044333-r1 \
  --authorize-historical-timeouts historical-token-count-timeouts/1 \
  --historical-timeout-ids schemas__sarif-external-property-file-2.1.0-rtm.3__a1 schemas__jsonld__a1 schemas__sarif-external-property-file-2.1.0-rtm.0__a1 ; echo "exit=$?"
```

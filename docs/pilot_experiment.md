# Scale Only pilot: 60 examples (20 Create / 20 Fix / 20 Modify)

Purpose: before spending GPU time on the 4,000 extra examples, measure acceptance rates, model calls, generation time
and operational reliability with the real Granite model. This is still the Scale Only experiment.

## What is the same as the smoke test (nothing scientific changes)

Same config `configs/scale_only_vllm_granite.json` (original-style standard generation, original Modify, temperature 0.7 /
max_tokens 4096 / top_p 0.95, grounding 0.9, Controlled Schema Reuse settings, `published_train` pool), same Granite revision
`b8c0982bab7fde4eb48110f5a069527c008fab39`, same anchor (`ab09def9...327f6`, HF revision `7962b0947f7d...`), same SDK
timeout 180 s with `max_retries=0`, same vLLM settings, authentication, scratch isolation, process cleanup and requeue
handling. The only differences are `--smoke-total 60` (split 20/20/20 by `split_total`) and the operational limits below.

## Files

- `slurm/pilot_granite_v2.sbatch` - the pilot job (a copy of the smoke script with pilot-specific names; the smoke script is untouched).
  Run `bash slurm/pilot_granite_v2.sbatch --preflight cluster` first (CPU only).
- `src/check_smoke_result.py` - now also takes `--label` and `--require-smoke-total`; with `--per-task 20` it requires exactly
  20 valid examples per task and keeps every integrity and train/test leakage check.
- `src/pilot_report.py` - metrics report (`data_scale_only/pilot_<job>-r<n>_report.json`, never overwritten).
- `tests/test_pilot.py` - CPU-only tests.

All paths carry `pilot_` / `pilot-` and `<job>-r<attempt>` (see docs/smoke_preflight.md, "Requeue and preemption"):
`logs/pilot-vllm-<run>.log`, `logs/pilot-pool-check-<run>.json`, `data_scale_only/pilot_<run>/`, `..._calls.jsonl`,
`..._report.json`, scratch `benchson-pilot-<run>.XXXXXXXX`. Any existing path stops the job (exit 10).

## Preflight (cluster, CPU only)

Everything the smoke preflight checks (branch, config pool source, anchor integrity and SHA pin, revision, pool report
reachable) plus: every task must have `target == 20` and `upper_bound_if_every_attempt_succeeds >= 20` (each task on its own),
`MAX_LLM_CALLS` 1-800, `GEN_TIMEOUT_SEC` 1-4800, tools `ninja setsid curl timeout`, output/report/log paths not existing.

## Result and exit codes

0 success (exactly 20/20/20, all valid, no leakage, manifest `smoke_total == 60`) | 2 shortfall | 3 technical (including
call budget used up or wall-clock limit, build exit 124) | 4 invalid output | 10 preflight | 11 vLLM/model.
A partial result keeps all files for inspection; it is never resumed or merged.

## Operational limits and why

| Limit | Value | Reasoning |
| --- | --- | --- |
| Slurm `--time` | 02:00:00 | model download + vLLM start (script waits up to 30 min for vLLM) + at most 60 min generation + margin. Shorter than the smoke's 3 h so it is easier to schedule on a preemptible partition. |
| Generation wall clock | 3600 s (`GEN_TIMEOUT_SEC`, hard max 4800), enforced with `timeout`, build exit 124 | generation only, after startup. |
| `MAX_LLM_CALLS` | 600 (hard max 800) | One schema attempt needs about 4 calls typically (reference, description, naturalized instruction x up to 2); the worst case for one attempt is about 18 (3 reference attempts x up to 4 JSON-fix calls, 3 description attempts, 2 naturalizations). A mock-model build of the same 60 examples used 281 calls at about 50 percent acceptance; this is a structural estimate only, NOT a Granite measurement. 600 covers acceptance down to about 25 percent. The full experiment needs about 41 percent, so a pilot that exhausts 600 calls is itself a NO-GO signal. The budget stops cleanly and the report shows partial rates. |
| Concurrency | 8 (config, unchanged) | |

These limits change no generation parameter and no acceptance rule.

## Metrics report

Every value is tagged measured / derived / unavailable. Per task: accepted (files on disk), failed attempts
(`generation_summary.failed_attempts`), attempts (derived = accepted + failed), acceptance rate (derived), target reached.
Totals: logical model calls (call log; SDK retries are disabled, so one record = one request), failed calls, call latency
(sum, mean, p50, p95), calls per accepted example, builder vs validator calls, failure reasons by task/stage, LLM event
counts, schema attempts. Generation duration is the wall clock around `build_benchmark.py` only (download, startup and
preflight are excluded) plus the manifest's own started/finished. NOT available: LLM calls per task (a schema attempt
shares its reference and description calls between Create, Fix and Modify, and the log is not tagged by task). The report
never changes the job's verdict.

## Preemption, resume and checkpoints (investigation)

Findings from the code:

1. The reuse builder has a resume path (`_reuse_scan_existing`): when started on an output directory that already contains
   examples, it re-registers them (references, content hashes, per-schema attempt numbers and accepted counts) and continues.
2. It is NOT reliable for an interrupted run: (a) examples and sidecars are written with a plain `open(path, "w")`, so a kill
   during a write leaves a truncated JSON file and the next scan raises (tested in `tests/test_pilot.py`); (b) failed attempts,
   schema suspension/retirement state, the call budget and the event log are not persisted, so a resumed run forgets which
   schemas already failed and repeats them; (c) failed attempt numbers are not recorded, so seed labels can repeat;
   (d) the manifest/report are written only at the end, so an interrupted run has no summary.
3. The Slurm scripts deliberately never reuse an output directory. After preemption the requeued attempt writes to a new
   `-r<n>` directory and the old partial output is left untouched. Nothing is merged silently.

Consequence for the pilot: an interrupted pilot is simply rerun (about an hour of GPU at most). For the full 4,000-example
run this would waste up to the whole run, so a minimal checkpoint/resume plan (NOT implemented) is:

1. Write every example and sidecar atomically (write to a temporary name in the same directory, `os.replace`).
2. Append one line per finished schema attempt to `attempt_log.jsonl` (schema id, attempt number, accepted tasks, failure
   stage, calls used), flushed and fsynced. The ledger, failure counts and call count are rebuilt from this log.
3. A new `--resume-from <previous attempt dir>` option that copies (never moves) the previous attempt's verified examples
   into the NEW attempt directory, replays the attempt log, verifies every file (valid JSON, anchor membership, no test
   overlap, hash recorded in the log) and records the source directory and file hashes in the new manifest. Without the
   explicit option nothing is ever carried over.
4. Chunk the full run into several Slurm jobs of a few hundred examples each (the existing per-task targets as cumulative
   goals), which bounds the loss per preemption without any resume code.
Option 4 alone needs no generator change and is the recommended first step.

# Scale-Only smoke test: preflight and cluster preparation

## data/jsb_schemas (the JSONSchemaBench part of the original schema pool)

`data/jsb_schemas` is git-ignored, so a fresh clone or the cluster checkout does not have it and the builder stops with
`Dataset folder not found: .../data/jsb_schemas/test`. The authoritative copy is the one the published Benchson v2 benchmark
was built from: git commit `e7f8b22e23097ebbedfd972faeeeb0c38c8d58ff` (1308 files; tree id `32e06bd10699e6e9b82f9b2aa0b8a1857e603523`).
It reproduces the original pool exactly (1669 eligible schemas = 250 test + 1419 train). Nothing is downloaded from another source.

    python src/prepare_jsb_schemas.py --check-only    # verify only
    python src/prepare_jsb_schemas.py                 # create from the pinned commit (needs the commit in the clone)
    python src/prepare_jsb_schemas.py --fetch         # same, after `git fetch origin`

An existing directory that does not match the pinned tree id is reported and never modified.

## CPU preflight

From the Benchson checkout on the Mac:

    bash slurm/smoke_granite_v2.sbatch --preflight local

This checks the branch, Python dependencies for schema processing, config presence and pinned corpus.
It does not require the published baseline or the cluster runtime.

On the cluster, from the checkout and before requesting any GPU allocation:

    bash slurm/smoke_granite_v2.sbatch --preflight cluster

This also checks the revision and nonempty train/test exclusions in the existing
\`data_published/7962b0947f7d/reserved_schemas.json\`, verifies that the reconstructed pool can supply
2/2/2 and checks the installed OpenAI/vLLM environments and launcher commands. The pool check is an
upper-bound check. Model acceptance and GPU availability are verified only by a future smoke job.
If the baseline is absent, use the pinned importer documented in scale_only_experiment.md.

## Smoke test

The job repeats cluster preflight before starting the model. Since the job already requested a GPU,
use the standalone CPU preflight above to catch problems before GPU allocation.
The job uses a fresh private directory under the existing user-owned \`SCRATCH_DIR\` override or
\`/scratch/$USER\`. It removes only that directory after checking its identity, and stops child
process groups on normal exit, errors and trappable cancellation signals. A forced kill or node
failure can prevent cleanup. Outputs and logs stay outside scratch.

The script applies \`umask 077\`. Slurm can open its own stdout log before the script runs.
Before any approved submission, set a private umask in the submitting shell and ensure the
\`logs/\` directory exists with appropriate permissions.

The OpenAI SDK uses a 180-second timeout and zero SDK retries only for this smoke invocation.
The model-call budget defaults to 60 (it may be lowered). Authenticated readiness probes are separate
from model generation calls. The generator's own retry settings are independent of the SDK.

Job exit codes: 0 SUCCESS (exactly 2 valid examples for each task), 2 DATA_SHORTFALL,
3 TECHNICAL_FAILURE (including missing/malformed required reports or a broken call log),
4 INVALID_OUTPUT, 10 preflight failed, 11 model download or vLLM startup failed.
Trapped HUP/INT/TERM retain their signal-style exit codes.
Use \`python src/check_smoke_result.py --out-dir <dir> --calls-log <file> --build-exit-code <n>\`
to re-check a finished run on CPU.

## Anchored pool (published_train) preparation

The scale-only configs use `reuse.pool_source: "published_train"`. Before the cluster preflight, `data_published/7962b0947f7d/published_train_schemas.json`
must exist next to `reserved_schemas.json` (it is written by `import_published_benchmark.py`, or offline by
`python src/export_anchor_pool.py --baseline-dir data_published/7962b0947f7d`). It is verified against `reserved_schemas.json` on every load;
`--check-pool` and the job stop on any mismatch. `data/jsb_schemas` is no longer needed to build this pool.

## Slurm script and the published_train pool (final compatibility audit)

`slurm/smoke_granite_v2.sbatch` now matches the `published_train` pool:

- The old `prepare_jsb_schemas.py --check-only` step is gone from the script. Instead the script requires
  `reuse.pool_source == "published_train"` in the config; any other value stops the job with exit 10.
- `--preflight cluster` (and the job itself) loads `data_published/7962b0947f7d/published_train_schemas.json` with
  `load_anchor`, which checks the file hashes and that it agrees exactly with `reserved_schemas.json`. It then requires
  revision `7962b0947f7d96eb3a0e6b9b928d939c7f3061a6` and `anchor_sha256`
  `ab09def994a78920d1df6bfa2d0c11f6e135294a46ebc33baca9307aee0327f6`.
- The `--check-pool` report must say `pool_source=published_train` with the same `anchor_sha256`, a non-empty pool, and
  for each of create / fix / modify separately: `target == 2` and `upper_bound_if_every_attempt_succeeds >= 2`.
- After the build, `check_smoke_result.py --anchor-file` verifies that every produced schema is a published TRAIN
  schema and none is a published TEST schema (violation = INVALID_OUTPUT, exit 4; an unreadable anchor = exit 3).
- The job never writes to `data_published/`; it only reads the baseline and anchor files.

## Requeue and preemption (same Slurm job ID, several executions)

BGU can preempt a job and requeue it automatically; the new execution has the same `SLURM_JOB_ID` but a higher
`SLURM_RESTART_COUNT` (job 24014523: first execution preempted, second failed because `logs/vllm-24014523.log` existed).

`slurm/smoke_granite_v2.sbatch` therefore names everything by `RUN_ID=<job>-r<attempt>` (`attempt` = `SLURM_RESTART_COUNT`,
default 0, must be a plain non-negative integer or the job stops with exit 10):

| Item | Path |
| --- | --- |
| vLLM log | `logs/vllm-<job>-r<attempt>.log` |
| pool report | `logs/pool-check-<job>-r<attempt>.json` |
| generation output | `data_scale_only/smoke_<job>-r<attempt>/` and `..._calls.jsonl` |
| private scratch | `<scratch>/benchson-smoke-<job>-r<attempt>.XXXXXXXX` (mktemp, mode 0700) |
| vLLM port | `20000 + (job + attempt*1009) % 20000` |
| Slurm stdout | `logs/benchson-smoke-<job>.out`, `--open-mode=append` so earlier attempts are kept in the same file |

Nothing is deleted or overwritten: the "path already exists" checks are unchanged, so rerunning the very same attempt
(e.g. by hand with the same variables) still stops with exit 10. Scratch cleanup only removes the directory that this
attempt created (name, owner, inode check). If an attempt dies from SIGKILL (the cleanup trap cannot run), its scratch
directory stays behind; remove it manually, it is not touched by later attempts. The smoke targets stay 2/2/2 and the
`published_train` anchor checks run on every attempt.

See docs/pilot_experiment.md for the 60-example pilot script (`slurm/pilot_granite_v2.sbatch`), which follows the same rules.

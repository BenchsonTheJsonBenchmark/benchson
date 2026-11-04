# Controlled schema reuse (scale-only experiment)

## What is the same and what is different
Same as the original Benchson v2 build: schema corpus (`data/schemas` + `data/jsb_schemas`), 10 KB limit, seed 0 /
test_ratio 0.15 partition, prompts (including the `Reasoning:` lines), the three task definitions, the standard Modify
selection (`original`), validation thresholds (grounding 0.9, min_populated_fields 2, max_build_attempts 3, ...), token limits
(not touched by the code) and train gates (deterministic, no round-trip). Complex generation is off (`complex_ratio` 0) but still in the code.

**Different, and must be stated when reporting results:**
1. The original policy was **one example per task per schema**. This experiment allows **several independent examples from the same
   schema** (controlled reuse).
2. The generator model is **IBM Granite**, not the original `openai/gpt-oss-120b`. Even with identical logic this is not an exact
   reproduction of the original generation process.

## How it works
* **Pool** = the original *training* pool (partition of all 1669 eligible schemas with seed 0 / ratio 0.15, train side),
  minus every schema whose id **or** content hash is in the published **test** split, minus content duplicates inside the pool
  (keeps one). Published *train* schemas stay in the pool - that is the point of reuse. Sanity check: >= 95% of the published train
  ids must be found in the reconstructed pool, otherwise the run aborts (wrong corpus / seed). Nothing is imported from upstream.
* **Attempts.** Every attempt on a schema mints a fresh reference object with the unchanged standard prompt and feeds all
  tasks that still need examples (same design as the original: one reference -> create / fix / modify).
  File names: `<source>__<subset>__<name>__a<attempt>.json` (the record fields are identical to the published ones).
* **Distinct, not just new ids.** A minted reference equal (canonical JSON) to any published reference of that schema, or to an
  earlier new one, is rejected like the other gates (counts toward `max_build_attempts`, reason `duplicate_reference_published|new`).
  A race between concurrent attempts is caught again at commit. Per task an exact-content check (schema + task fields) also runs;
  the same JSON under a *different* schema is only counted (`reject_cross_schema_duplicates` can make it a rejection).
  The merge tool repeats all duplicate checks on the files.
* **Balance.** Work runs in passes; in each pass the schemas with the fewest (published + new) references go first
  (ties: seeded shuffle). `max_new_references_per_schema` caps new references per schema (each feeds up to 3 tasks).
* **Bounded retries / skipping.** A schema whose attempts produce nothing `max_failures_per_schema` times in a row is *suspended*
  for `suspend_passes` passes, may be suspended `max_suspensions` times, then is *retired* for the run;
  `max_attempts_per_schema` is a hard bound. Other schemas continue, so a bad schema never blocks the run, and the quota
  per task is still pursued until it is met or no schema is left (then the shortfall is reported).
* **Determinism.** Per-attempt rng label `<seed>-train-<schema id>-a<n>`; optional `sampling_seed_param` passes a per-attempt integer to the
  provider (e.g. `"seed"`), left off by default because support on the Granite endpoint is unverified. Model output itself is not deterministic.
* **Smoke test.** `--smoke-total N` uses exactly the same code path and config, only fewer examples (separate `_smoke` output).

## Logs and reports (`<output>/_generation_reports/<ts>/`)
`generation_summary.json` (section `schema_reuse`: per-schema histograms of references/examples, per-task totals, statuses,
attempts, failures, duplicate statistics, suspend/retire/cap events, plus `schema_pool` counts and `llm_events`),
`schema_ledger.jsonl` (one row per schema: attempts, accepted references, failures, status, examples per task),
`skipped_schemas.jsonl` (suspended / retired / capped events with reasons), `generation_failures.jsonl`, `generation_events.jsonl`,
and per-example sidecars `benchmark_*/_metadata/train/<id>.json` (schema id, attempt, reference hash, validation, attempts).

## Feasibility (check without a model)
`python src/build_benchmark.py --config configs/scale_only_vllm_granite.json --check-pool [--assumed-acceptance 0.7]`
prints the pool, the upper bound (pool x cap), the *required average acceptance* per attempt (assumption-free), and the expected
distribution. Using the git `e7f8b22` data as a stand-in for the HF baseline (real numbers need the real import):

| | cap 2 | cap 3 |
|---|---|---|
| usable schemas (1419 pool - 11 test-content matches - 21 pool duplicates) | 1387 | 1387 |
| max new references | 2774 | 4161 |
| average acceptance needed per attempt for 1334 Create | 48% | 32% |
| Create-capable schemas (published Create example exists) | ~930 | ~930 |

Expected distribution at 50% acceptance: ~2,670 references to mint -> ~1,280 schemas with 2 new references, ~106 with 1; published
examples per pool schema today: 0:150, 1:224, 2:379, 3:634 (so total examples per schema end up roughly 3-9). Create is the tightest task
(only ~930 schemas are known to pass its grounding gate), so cap 3 gives margin. Whether ~4,000 distinct examples are *really* reachable
depends on the acceptance of repeated attempts, which is unknown until the real smoke test; the retry policy guarantees the run ends
with a precise shortfall report instead of looping.
Rough cost: ~2,000-2,700 attempts x ~4-6 model calls each (mint, description, naturalize) = roughly 10-15k calls.

## Merge and integrity
`python src/merge_scale_dataset.py --baseline data_published/<sha12> --new data_scale_only/<run> --out-dir data_scale_7k --allow-train-schema-reuse`
rejects: any new test example, file-name clash, a schema in the published test split (content hash), an exact duplicate of a published or new
example, a new reference equal to a published reference of the same schema, changed field sets; verifies the test files are byte-identical;
writes `merge_manifest.json` (duplicate statistics, new-examples-per-schema histogram, how many new schemas were already in published train).

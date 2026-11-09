# Full Scale Only run: context-window guard, chunked generation, model cache

Scope: the 4,000 extra original-style training examples (Create 1,334 / Fix 1,333 / Modify 1,333) with Granite 4.0 H-Small.
Nothing scientific changed: same prompts, sampling parameters (temperature 0.7, max_tokens 4096, top_p 0.95), gates,
original Modify, standard-only generation, `published_train` anchor pool, Controlled Schema Reuse settings (cap 3), seed 0.

## 1. Root cause of the pilot's technical failure (job 24036543, attempt r3)

The three failed calls said: `Maximum context length is 8192 tokens. Requested output tokens: 4096. Prompt contains at least
4097 input tokens.` vLLM runs with `--max-model-len 8192` and every request asks for `max_tokens = 4096`, so a request is
accepted only if its prompt has at most 4,096 tokens (`prompt + max_tokens <= 8192`).

Why Github_hard schemas exceed that:
- The Create prompt embeds the schema with `json.dumps(schema, indent=2)`. The pool limit (10 KB) is measured on the COMPACT
  schema, but the indented text in the prompt is about 1.5x longer (median) and up to 2.1x: on the real anchor pool the largest
  compact schema is 10,042 bytes and its prompt text is 20,845 characters. Of the 1,076 usable schemas, 130 have more than
  10,000 and 38 more than 14,000 characters of indented schema (a rough range for the at-risk schemas, 4 to 12 percent;
  characters are not tokens and exact token counts need the server's tokenizer).
- Later requests grow further: a repair request carries the previous invalid answer (up to 4,096 tokens) plus the error.
- Reference JSON of the description request is added to the schema again.
I could not identify the three schema ids offline (they are in the cluster's call log). `python src/analyze_context_failures.py
<calls.jsonl>` reads that log and says for each failed call whether it was the schema's own first prompt or a grown repair prompt.

These failures were genuine capacity limits of the setup, not model errors, but they were reported as `provider_error` /
`worker_error`, so the strict checker (correctly) called the pilot a technical failure although 60/60 examples were fine.

## 2. What changed for context overflow (no scientific change)

- `src/llm/context_guard.py`: before each request the guard counts the prompt with the vLLM server's own tokenizer and chat
  template (`POST /tokenize`, exact; `OpenAIProvider.count_prompt_tokens`) and applies the server's own rule
  `prompt_tokens + max_tokens > max_model_len`. A request that does not fit raises `ContextOverflow` and is NOT sent (no model
  call, no failed call, no budget used). A request that fits is forwarded byte for byte, with the same parameters.
- The guard checks that the server reports the same `max_model_len` as configured (`--context-guard 8192`); a mismatch, or
  three failed token counts in a row, stops the run. If counting fails once the error propagates as an ordinary error
  (still technical), it is never silently skipped.
- Builder: a refused request is recorded as event `context_overflow` and failure stage `context_overflow` (listed under
  `capacity_exclusions` in `generation_summary.json` with the token numbers), separate from `provider_error`/`worker_error`.
  If the schema's own first prompt does not fit, every attempt would be identical, so the schema is retired at once without
  retrying ("context_overflow" in the ledger). If only a grown repair prompt does not fit, that build attempt fails and the next
  one starts again from the original prompt. Tasks keep their normal acceptance rules; no example is produced from a request
  that was not made.
- Checker: `context_overflow` events are a warning (expected exclusion). `provider_error` events, failed model calls and
  `worker_error` stay technical, so a server-side overflow that slips past the guard still fails the job. The full run also
  requires that the guard was enabled for the 8,192 window and counted requests (`--require-context-guard`).
- The guard is opt-in (`--context-guard`), used only by the full-run script; the working smoke and pilot scripts and the config
  are unchanged.

Option that needs your approval and is NOT implemented: raising vLLM `--max-model-len` (for example to 16384) would let the
130-or-so long-schema prompts run instead of being excluded. It changes no sampling parameter, but it changes which schemas
can produce examples and the GPU memory use, so it should be a deliberate decision (and `--context-guard` must equal it).
Alternative: leave the exclusion; capacity stays sufficient (see section 6).

## 3. Chunked generation with checkpoints

Design (`src/data_generation/chunking.py`, `src/chunked_run.py`, `slurm/full_chunk_granite_v2.sbatch`):
- 8 chunks. Chunk k builds `floor(T*k/8) - floor(T*(k-1)/8)` of each task's total T: the quotas sum exactly to
  1,334 / 1,333 / 1,333 (chunk 1: 166/166/166, and so on; `python src/chunked_run.py increments`).
- One chunk = one Slurm job, `CHUNK_INDEX=k [PARENT_CHUNK=<previous chunk dir>]`. Directory
  `data_scale_only/full/chunks/c<kk>_<job>-r<attempt>/`; calls log and report beside it. Every attempt has its own paths.
- Each chunk continues from the verified checkpoint `chunk_state.json` of its parent. It carries the per-schema ledger
  (attempts, accepted references, failures, suspensions, retirement: all limits are GLOBAL, never reset), the hash of every
  example file, the anchor/pool/config fingerprints and a hash link to the parent. The duplicate index is rebuilt from the
  files of the whole chain (every file hash re-verified first), so duplicate control is global too. Attempt numbers continue, so
  file names never collide across chunks.
- The global cap (3 new references per schema, hence at most 3 examples per task per schema) is enforced twice: by the carried
  ledger while generating, and by an independent recount from the files during validation (including a forged-but-resealed
  state, tested).
- Pass numbers restart in every chunk; a suspended schema keeps only the passes it still has to wait. The rest of a pass that a
  chunk ended in is not replayed (fewest-references-first ordering rebalances). Attempts that were still running when a chunk
  reached its quota are discarded and still count against `max_attempts_per_schema` (conservative; never reset).
- The checkpoint is written last, only when the chunk met its quota exactly. A chunk that stopped early (budget, wall clock,
  preemption, shortfall) has no state file: it is kept for inspection, can never be a parent, and is never merged.
- Validation (`chunked_run.py validate`, run by the job for its own chunk and, before any GPU time is used, for the parent
  chain): every file hash, one pool/config/plan for the whole chain, example-level checks (schema validity, grounding,
  groundedness), train-only/no-test schemas, no identical example to a published or earlier one, no repeated reference per
  schema, global cap, ledger vs files, exact cumulative totals, context-guard evidence, call log without failed calls.
- Recovery after preemption is explicit: look at `chunked_run.py status`, then start the SAME chunk index again with the SAME
  parent (new attempt directory automatically through the requeue counter, or a fresh submission). Partial output stays untouched.
  `chunked_run.py plan --chain c01_... c02_...` validates a chain you name and prints the next chunk's variables.
- Merge: `chunked_run.py merge --root ... --chain c01_... ... c08_... --anchor-file ... --out-dir NEW` copies the explicitly named,
  fully validated chain into a new directory (refuses an existing one; sources are only read; every copy is hash-checked) with
  `merged_chunks_manifest.json`. Combining it with the published baseline is the existing `src/merge_scale_dataset.py` step.
- The published baseline directory is only ever read (tested by hashing the tree before and after).

Limitation: a chunk is not resumable in the middle. Preemption loses at most one chunk (about 30 minutes of generation); the
checkpoint after each chunk is what makes the full run robust. Mid-chunk resume would additionally need atomic example writes
and an append-only attempt log (see docs/pilot_experiment.md); it is not needed with 8 short chunks.

## 4. Model cache

I cannot reach the cluster from here, so nothing about BGU storage was verified or changed. What exists:
- Default (unchanged): the pinned Granite revision is downloaded into the job's private scratch every time and removed at exit.
- Opt-in: `MODEL_CACHE_DIR=<absolute directory>`. The script never creates it, never follows a symlink, requires that you own it,
  that it is writable and not group/world-writable, never deletes anything in it (the only `rm -rf` is still the private job
  scratch), and lets `huggingface_hub` add only the pinned revision `b8c0982b...`. A complete cached snapshot is used offline
  (`local_files_only`); an incomplete one is resumed by `snapshot_download`; the snapshot must be the pinned revision directory,
  contain `config.json` and every weight shard named in the index (non-empty) or the job stops (exit 11). Concurrent jobs
  downloading the same revision use huggingface_hub's file locks. Requeue isolation is unchanged (scratch and logs are per attempt).
  Scratch needed drops from 90 to 20 GiB.
- Whether a persistent location is APPROVED and large enough (about 64 GB) is for you or the BGU admins to decide. Read-only
  checks to run on a login node before pointing `MODEL_CACHE_DIR` at it: `df -h <dir>`, `ls -ld <dir>`, `stat -f -c %T <dir>`
  (file system type), the cluster's storage/quota policy, and whether vLLM loading from that file system is fast enough (weights
  are read in place, not copied to scratch; if loading is slow, keep the scratch default).

## 5. Files

Created: `src/llm/context_guard.py`, `src/data_generation/chunking.py`, `src/chunked_run.py`, `src/analyze_context_failures.py`,
`src/llm/mock/varying_mock_provider.py` (test aid), `slurm/full_chunk_granite_v2.sbatch`, `tests/test_context_guard.py`,
`tests/test_chunking.py`, `tests/test_full_chunk_script.py`, this document.
Modified: `src/build_benchmark.py` (`--context-guard`, `--chunks/--chunk-index/--parent-chunk`, manifest fields),
`src/data_generation/benchmark_builder.py` (overflow handling, chunk restore/state), `src/data_generation/reuse.py`
(`SchemaLedger.retire`), `src/llm/openai/openai_provider.py` (`count_prompt_tokens`), `src/check_smoke_result.py` (per-task
quotas, `context_overflow` warning), `src/pilot_report.py` (`--targets`), `tests/test_smoke_preflight.py` (stub support).
Untouched: smoke and pilot scripts, configs, the published baseline.

## 6. Estimates from the measured pilot (60 examples, 196 calls, 3 failed, 206 s net generation)

- 3.27 calls and 3.43 s per accepted example. For 4,000 examples: about 13,100 calls and 13,700 s (3.8 h) of net generation,
  8 chunks of about 500 examples: about 1,630 calls and 29 minutes each, plus startup (download about 6 min measured before the
  first preemption, vLLM load; shorter download with a cache). Roughly 6 to 8 GPU-hours in total.
- Chunk limits: Slurm 3 h, generation wall clock 5,400 s (3.1x expected), `MAX_LLM_CALLS` 3,000 (1.8x expected, hard max 4,000).
- Acceptance in the pilot (small samples, wide uncertainty): Create 57.1 percent, Fix 69.0, Modify 74.1. Create needs about
  41 percent of the 3,228 reference slots (1,334/3,228); at 57 percent it needs about 2,335 attempts. If the context guard
  excludes 4 to 12 percent of schemas, capacity falls to about 2,840 to 3,100 slots, still above 2,335 but with less margin.

## 7. Remaining risks

- Capacity margin for Create after the guard excludes long schemas (see above); real acceptance may differ from the pilot's 35 attempts.
- `/tokenize` must exist on the served vLLM version (it does in current vLLM; the job stops if it does not answer).
- A SIGKILL leaves the private scratch behind (unchanged from before); chunk outputs are never deleted by any script.
- Chunking changes the scheduling slightly (pass numbering, cancelled attempts count), documented above; generation is already
  nondeterministic through concurrency, so chunks are not bit-reproducible, but every accepted example and every limit is verifiable.
- `slurm/scale_only_vllm.sbatch` (an older tracked script) fails `bash -n`; it was not touched and is not used.

## 8. Recommendation

GO for `CHUNK_INDEX=1 bash slurm/full_chunk_granite_v2.sbatch --preflight cluster` after you review this change, then a single
chunk 1 job. NO-GO for submitting all chunks at once: submit chunk k+1 only after chunk k validated (its job prints
`CHUNK RESULT: SUCCESS`) and pass its exact directory name as `PARENT_CHUNK`.

## 9. Success marker and overnight chain (added after the first review)

- A chunk is usable only with `chunk_success.json` next to its `chunk_state.json`. It is written (never overwritten) by
  `chunked_run.py seal` after every final validation passed, and binds the run identity, the hash of the validated state and of every
  example file, the exact counts, the final checker result and the calls-log validation (hash, number of calls, 0 failures).
  `chunking.load_chain` requires it for every chunk by default, so parent validation, the builder, the next job and the final merge all
  refuse an unsealed chunk. A chunk whose final validation failed keeps its output (for debugging) and has no marker.
- Chunks 2-8 can run unattended as a dependency chain: see `docs/overnight_chunk_chain.md`.

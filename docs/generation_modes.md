# Standard / complex generation modes

Branch `improve-data-generation`. This describes the configurable mix of **standard**
(original-style) and **complex** examples across Create, Fix and Modify, and how to generate
3-5 examples to inspect before scaling up.

Nothing here lowers a quality gate. `grounding_threshold` (0.9), `max_new_tokens`,
`max_schema_kb`, schema validation and the retry limits are exactly as before. The existing
Create pilot (`src/preview_create_pilot.py` and its configs) is not modified.

## What each mode means

| Task | standard | complex |
|---|---|---|
| Modify | the **original** Benchson behaviour: one edit, uniform choice, top level + one nesting level (`standard_modify_strategy: "original"`, verified against commit `2a3ae68`) | 2-3 compatible edits in one instruction: different kinds (scalar / add-remove field / array), different depths and branches, including objects inside arrays (`items[0].qty`). Every edit leaves the object schema-valid. |
| Fix | the original single-error injector (`_inject_error`, now seedable) | 2-3 errors in one instance: different violation kinds (wrong type, range, enum, format, required, additionalProperties, array size/items), different depths/branches, also inside list items. Every error breaks the schema on its own, no two overlap, and the originals are recorded so the correction is known. |
| Create | unchanged prompt and gates | the reference is requested with an extra "rich and realistic" hint and must reach configurable thresholds (leaves, depth, distinct value types, arrays of objects). The description then goes through the **unchanged** grounding (and round-trip) gates. Schemas that cannot reach the thresholds are skipped before any model call. |

A complex example that cannot be built is **recorded as a failure and never replaced by a
standard one**; the standard quota is separate.

Instance size and task complexity are measured separately (`complexity.py`): a larger
reference is not assumed to be a harder Create task.

## Configuration

A `"generation"` block in the build/preview config (see `configs/preview_complex_*.json`):

```json
"generation": {
  "standard_ratio": 0.8,
  "complex_ratio": 0.2,
  "standard_modify_strategy": "original",
  "complex": {
    "modify": {"min_operations": 2, "max_operations": 3, "attempts": 5},
    "fix":    {"min_errors": 2, "max_errors": 3, "attempts": 5},
    "create": {"min_leaves": 10, "min_depth": 3, "min_distinct_types": 3, "min_object_arrays": 0},
    "max_failures_per_example": 5
  }
}
```

* Ratios must sum to 1. Use 1/0 for 100% standard, 0.7/0.3 for 70/30.
* **No `generation` block = legacy behaviour**: nothing changes and no extra files are written.
  (Legacy modify keeps the recursive, complexity-weighted selection of the previous commits.)
* Quotas are integers computed with exact fractions: the complex total over all tasks follows
  the ratio (4 examples at 75/25 contain exactly 1 complex), then complex examples are shared
  between tasks in proportion to their size. Ties break in the order create, fix, modify.
* Counts per task/split still come from `counts` as before. The mix applies to each split's
  per-task count. Train/test stay disjoint by schema, as in the original builder.
* `max_failures_per_example` bounds the complex retries: after `value x quota` failed
  schemas for a task the complex quota is closed and reported as a shortfall.
* Randomness: every schema gets a seed label `<seed>-<split>-<n>`; standard modify uses the
  original rng stream, complex modify/fix use derived streams. Same seed + same model output
  = same data (a real LLM is of course not reproducible).

## Generating 3-5 examples (no full run)

Uses the real `BenchmarkBuilder`; only the model is swapped for tests/mock runs.

```bash
# 4 examples over create/fix/modify, 3 standard + 1 complex
python3 src/preview_generation.py --config configs/preview_complex_mock.json \
  --total-examples 4 --standard-ratio 0.75 --complex-ratio 0.25 --out-dir outputs/smoke_test

# one task, one mode, 4 examples
python3 src/preview_generation.py --config configs/preview_complex_mock.json --tasks create --mode standard --count 4
python3 src/preview_generation.py --config configs/preview_complex_mock.json --tasks create --mode complex  --count 4
python3 src/preview_generation.py --config configs/preview_complex_mock.json --tasks fix    --mode complex  --count 4
python3 src/preview_generation.py --config configs/preview_complex_mock.json --tasks modify --mode complex  --count 4
```

Flags: `--tasks`, `--count` (per task) or `--total-examples` (over the selected tasks, extra
examples to the earlier tasks), `--mode standard|complex|mixed`, `--standard-ratio`,
`--complex-ratio`, `--seed`, `--out-dir`, `--max-llm-calls` (hard stop), `--config`.
Swap in `configs/preview_complex_watsonx_granite.json` for the real model (needs `.env`).

## Outputs (inside `--out-dir`)

| Path | Content |
|---|---|
| `review.md` | the examples for manual reading: mode, complexity, checks, what changed, raw model outputs |
| `summary.json` / `generation_summary.json` | requested vs produced per task and mode, failures by reason, complexity statistics, validation failures, LLM calls, output locations |
| `generation_failures.jsonl` | one row per failed attempt (task, mode, schema, stage, detail) |
| `llm_calls.jsonl` | every prompt and raw response |
| `dataset/benchmark_<task>/{test,train}/*.json` | the examples, **same record format as before** |
| `dataset/benchmark_<task>/_metadata/<split>/<id>.json` | sidecar per example: task, mode, schema id, seed, model/provider, complexity (operations/errors: count, types, max depth, affected fields, different branches, inside list items; reference: leaves, depth, types, arrays of objects), validation results, attempts and failure reasons, items needing manual review |

Full builds write the same report to `data/_generation_reports/<timestamp>/` and the sidecars
next to each dataset; `export_hf.py` / `export_training.py` ignore them.

## Quality checks

Programmatic (`verify_modification`, `verify_fix`, builder gates):

* Modify: ground truth differs from input and validates against the full schema; operation
  count matches; no two operations overlap; each operation is effective alone; **replaying the
  recorded operations reproduces the ground truth** (an independent implementation of every
  operation); every changed leaf lies on an operation path (no unintended change); a rewritten
  instruction is kept only if it still names every field and value (otherwise the precise
  template is used).
* Fix: target valid, corrupted invalid; each recorded error breaks the schema alone; the
  validator reports a violation at/near every error; putting the recorded originals back
  reproduces the target exactly; replay matches; no change outside the error paths.
* Create: reference schema-valid and non-trivial; complexity thresholds (complex only);
  unchanged grounding ratio and optional round-trip.

Not provable by code, flagged `requires_manual_review` in every sidecar: that an instruction's
wording matches the intended edits (heuristic + optional round-trip only), that injected errors
are realistic for the domain, and that a Create description conveys every value faithfully
(grounding is literal string overlap only - see `data_generation_research_notes.md`).

## Known limitations

* The complex Create "rich reference" hint is a first wording; judge it on the first real
  samples. On the offline mock most complex Create attempts fail (the mock builds shallow
  objects), which says nothing about a real model.
* Only the first 3 elements of an array of objects are traversed; tuple-style `items`,
  `anyOf/oneOf` branches and remote `$ref`s are not followed (the validator still checks
  everything, so such schemas just yield fewer candidates).
* Delete-field is the most frequent complex Modify operation because most properties are
  optional; no per-kind balancing is applied yet.
* Round-trip validation in complex mode needs the validator model to apply 2-3 edits at once;
  enable it only after looking at the raw outputs.
* Duplicates: one example per task per schema, and identical content produced from different
  schemas is *reported* (`duplicates` in the summary, `duplicate_of` in the sidecar) but not
  removed; near-duplicates and duplicates against earlier runs are not checked.

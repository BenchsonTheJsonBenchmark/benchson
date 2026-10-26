# Data-generation research notes

Date: 2026-10-26 | Branch: `improve-data-generation`

These notes record the **new** findings from the recent investigation of Benchson's data
generation (Modify, Fix and Create). They do not re-explain the project. Each section says what
was measured, what is supported by the inspected data, and what is still only a hypothesis.

## 1. Summary

| # | Finding | Status |
|---|---|---|
| 1 | Existing Modify examples are all single-operation (one changed leaf) | Proven for all 98 inspected examples |
| 2 | The 11 inspected Fix examples all contain exactly one leaf-level error | Proven for those 11 only |
| 3 | Schema complexity is not the same as task complexity | Conceptual finding (framework, not a metric) |
| 4 | In the valid Create pilot, reference generation usually succeeded | Observed (9 schemas, 1 run each) |
| 5 | After a valid reference existed, every Create failure was `grounding_failed` | Observed in the pilot only |
| 6 | "More reference leaves = worse grounding" was not supported | Observed (counterexamples in the pilot) |
| 7 | The grounding gate may reject useful descriptions over missing literal technical strings | Hypothesis, not proven to be a bug |
| 8 | Description retries did not consistently improve grounding | Observed in 4 cases, not generalizable |
| 9 | The repeated (`--repeats 3`) run is invalid (WatsonX 429) | Not evidence; must not be used |

## 2. Modify dataset findings

All 98 existing Modify examples under the inspected dataset were analysed.

- 98/98 (100%) differ by exactly **one leaf value** between the original JSON and the ground truth.
- There are no multi-operation examples.

Edit depth:

| Depth | Examples | Share |
|---|---:|---:|
| 1 | 33 | 33.7% |
| 2 | 43 | 43.9% |
| 3 | 18 | 18.4% |
| 4 | 4 | 4.1% |
| > 4 | 0 | 0% |

77.6% of the edits are at depth 1-2.

Change type:

| Type | Examples | Share |
|---|---:|---:|
| numeric change | 32 | 32.7% |
| string change | 23 | 23.5% |
| array add | 16 | 16.3% |
| array remove | 11 | 11.2% |
| delete field | 8 | 8.2% |
| boolean change | 8 | 8.2% |

Input JSON complexity: average 12.2 leaf values (min 11, max 16); average maximum depth 3.4 (max 4).

**Interpretation.** Modify has diversity in operation *type* but almost no *compositional*
complexity. A complex schema or instance does not imply a complex Modify task: even complex
references seen in previews received trivial edits (increment one port, change one string, remove
one field, remove one array element).

Possible future direction (**not implemented**):

| Level | Definition |
|---|---|
| Easy | Original-style single operation |
| Medium | 2-3 operations |
| Hard | Multiple operations of different types, deeper locations, different branches, arrays plus scalar changes, potentially interacting constraints |

## 3. Fix dataset findings

All 11 existing Fix examples found in the inspected dataset were analysed.

- 11/11 contain exactly **one leaf-level error**; no multi-error examples.
- All 11 errors are at depth exactly 3, with essentially the same structural form under `natureItems[0]`.
- Error types are simple type/value violations: numeric value change, string value change, boolean to string, string to null, integer to string, string to integer.
- Several error patterns repeat or are nearly identical.

**Interpretation.** The inspected Fix examples have very low compositional error complexity.

**Caveat.** There are only 11 examples. These are findings about those 11, not proof about every
Fix example the system could generate.

Possible future direction (**not implemented**):

| Level | Definition |
|---|---|
| Easy | One error |
| Medium | 2-3 diverse errors |
| Hard | Multiple errors across different locations and constraint families, potentially interacting |

## 4. Complexity taxonomy

Central conceptual finding: **schema complexity != task complexity.** At least these are distinct:

| Kind | Examples |
|---|---|
| Schema complexity | property count, nesting depth, required fields, arrays, `oneOf`/`anyOf`/`allOf`, `$ref` usage |
| Instance complexity | leaf count, object depth, populated fields, arrays and their sizes, nested objects, serialized size |
| Task complexity: Modify | number of operations, operation types, edit depth, number of affected fields, edits in different branches, interactions/dependencies |
| Task complexity: Fix | number of errors, error diversity, error depth, errors in different branches, interaction between errors/constraints |
| Task complexity: Create | amount of information in the reference, structural complexity of the reference, amount/type of information the description must preserve |
| Instruction complexity | direct vs multi-part instructions, compositional instructions, dependencies between requested operations |

Conceptual representation only (**not a finalized metric**):

`Complexity(example) = f(schema complexity, instance complexity, task/transformation complexity, instruction complexity)`

## 5. Schema-pool analysis

All 700 schemas in the relevant pool were inspected. The full pool has a very large long tail,
including extremely large schemas. The preview config uses `max_schema_kb = 10`.

- Total schemas: 700
- Eligible (<= 10 KB): 362 (51.7%)

Statistics of the eligible subset:

| Property | Mean | Median | Max |
|---|---:|---:|---:|
| size (KB) | 3.75 | 3.19 | 9.97 |
| maximum depth | 6.80 | 7 | 19 |
| properties | 17.19 | 13 | 92 |
| required | 3.46 | 1 | 56 |
| arrays | 2.44 | 1 | 20 |
| oneOf | 0.57 | 0 | 22 |
| anyOf | 0.57 | 0 | 19 |
| allOf | 0.22 | 0 | 12 |
| $ref | 4.62 | 1 | 38 |

**Conclusion.** The 10 KB limit removes almost half the pool, but the remaining schemas still have
substantial structural variation. Complexity experiments can therefore run without changing the
10 KB limit first.

## 6. Temporary sampling heuristic (not a complexity definition)

To pick deliberately varied schemas for a Create pilot, a temporary score was computed from
percentile ranks across the eligible <= 10 KB schemas, averaging equal-weight ranks of: properties,
maximum depth, required, arrays, oneOf, anyOf, allOf, $ref.

**This is not an official Benchson complexity definition and must not be presented as one.** The
LOW/MEDIUM/HIGH labels are experimental sampling tiers only.

| Tier | Schemas |
|---|---|
| LOW | `typingsrc`, `one-changelog-schema-0.1`, `drupal-links-action` |
| MEDIUM | `nodehawkrc`, `devinit.schema-2.0`, `gematik-test-hcps` |
| HIGH | `aiproj-1.0`, `asconfig-schema`, `azure-iot-edgehub-deployment-1.2` |

All files are under `data/schemas/test/`.

## 7. Earlier Create preview observations

Before the controlled pilot, preview analysis suggested a possible Create bottleneck:

- Reference JSON generation often succeeded.
- Natural-language description generation repeatedly failed the grounding gate.
- Repeated description attempts consumed LLM calls.

Earlier preview observations: a reference with about 19 leaf values was accepted; about 40 leaves
failed; about 57 leaves at depth about 7 failed.

This produced the initial **hypothesis**: richer references may be harder to describe while
preserving enough information to pass grounding, which could bias accepted Create examples against
complex cases. It was a hypothesis, not a conclusion (see section 10 for what the pilot showed).

A separate, earlier failure mode: large generated JSON can be **truncated**, giving invalid JSON and
repeated failed correction attempts. Treat this separately from grounding failure.

## 8. Controlled Create complexity pilot

A dedicated runner observes each selected schema independently without changing generation logic
(see section 18).

| Setting | Value |
|---|---|
| Model | `ibm/granite-4-h-small` |
| Grounding threshold | 0.9 |
| Max retries | 2 |
| Max build attempts | 3 |
| Round-trip validation | disabled |
| Temperature | 0.7 |
| max_new_tokens | 4096 |
| Concurrency | 1 |

First valid WatsonX pilot: 9 schemas, 1 run per schema, 34 LLM calls, no provider failure.

| Tier | Accepted |
|---|---|
| LOW | 2/3 |
| MEDIUM | 1/3 |
| HIGH | 1/3 |

With n = 3 per tier and a single run per schema, **no statistically meaningful tier relationship
can be claimed.**

## 9. Reference-generation findings

Reference generation succeeded for 8/9 schemas. For 7 of those 8, it succeeded on the first build
attempt with one LLM call.

The exception is `gematik-test-hcps`: it failed after 3 build attempts and 9 LLM calls, with 9/9
failures classified as **schema violations**. There were no provider errors, no truncation, and no
invalid-JSON classifications for it.

This looks like a schema-specific failure mode worth investigating separately. It should not yet be
read as evidence about medium-complexity schemas in general.

## 10. Create grounding bottleneck and the "more leaves" hypothesis

Among the 8 schemas with a valid reference: 4 Create examples were accepted and 4 failed. **All 4
post-reference failures were `grounding_failed`.** In this pilot, once a reference existed, every
remaining Create failure happened at the description/grounding stage. This strengthens the earlier
observation, but must not be generalized without more runs.

Per-schema results:

| Schema | Tier | Ref leaves | Checked strings | Best grounding | Accepted |
|---|---|---:|---:|---:|---|
| typingsrc | LOW | 13 | 11 | 0.9091 | yes |
| one-changelog | LOW | 6 | 4 | 0.5000 | no |
| drupal-links-action | LOW | 13 | 12 | 1.0000 | yes |
| nodehawkrc | MEDIUM | 17 | 11 | 0.9091 | yes |
| devinit | MEDIUM | 13 | 12 | 0.8333 | no |
| gematik-test-hcps | MEDIUM | - | - | - | no (reference generation failed, section 9) |
| aiproj | HIGH | 42 | 31 | 0.8710 | no |
| asconfig | HIGH | 57 | 25 | 1.0000 | yes |
| azure-iot-edgehub | HIGH | 19 | 14 | 0.8571 | no |

The simple hypothesis **"more leaves = worse grounding" was not supported.** The clearest
counterexample: `asconfig` has 57 leaves and 25 checked strings, scored 1.0, and was accepted on
the first description; `one-changelog` has only 6 leaves and 4 checked strings, scored 0.5 on all
three attempts, and was rejected. Reference size alone does not explain acceptance or grounding
success, so the initial hypothesis has to be refined.

## 11. What the grounding metric actually measures

Inspection of `_grounding_ratio` (in `src/data_generation/benchmark_builder.py`) confirmed:

- It does **not** evaluate all leaf values, only some string values.
- A string is checked only if it has at least 4 characters and at least one alphabetic character.
- If no eligible strings exist, it returns 1.0 automatically.
- The default threshold is 0.9.

So the score is not "the percentage of reference information preserved". The pilot showed
rejections caused by missing literal technical strings. Examples of ungrounded strings:

- `http://json-schema.org/draft-04/schema#`
- `Bearer abcdefghijklmnopqrstuvwxyz`
- `password123`
- `proxypassword`
- `mqtt:publish`
- `mqtt:connect`
- route expressions such as `FROM /* INTO $upstream`

Example: `aiproj` had 31 checked strings and a best grounding of 0.871, with only four strings
missing on its best attempt, yet it was rejected because the threshold is 0.9.

**New research question (hypothesis, not a proven bug):** is the current grounding gate rejecting
semantically useful descriptions because they do not reproduce certain literal technical strings?

## 12. Description retries

For the four grounding failures, grounding per attempt (1st -> 2nd -> 3rd):

| Schema | Attempts |
|---|---|
| one-changelog | 0.50 -> 0.50 -> 0.50 |
| devinit | 0.83 -> 0.67 -> 0.83 |
| aiproj | 0.87 -> 0.74 -> 0.68 |
| azure-iot-edgehub | 0.86 -> 0.86 -> 0.64 |

In this small pilot, repeated independent description generation did not systematically improve
grounding; in some cases the first attempt was the best. This is an observation from four cases
and should not be generalized until repeated runs exist.

## 13. Generation cost

Quality gates affect generation cost:

- A successful Create can take as few as 2 calls (one reference + one description).
- A grounding failure took 4 calls (one reference + three descriptions).
- `gematik-test-hcps` used 9 calls and still produced no reference.

A useful future efficiency metric is **LLM calls per accepted example**. It matters before scaling
generation from roughly 2K examples toward 7K-10K.

## 14. Repeated run (`--repeats 3`): INVALID, provider limit

After the valid one-run pilot, the same 9 schemas were attempted with `--repeats 3` (intended: 27
observations). **This run is not research evidence and must not be used.**

WatsonX returned HTTP 429 immediately with `consumption_limit_reached` for
`ibm/granite-4-h-small`. The first three provider calls failed and the runner stopped automatically
after three consecutive provider failures. Therefore:

- The first three rows are provider failures, **not** Create or reference failures.
- The other 24 rows are `not_run`.
- None of it may be included in Create success rates or complexity relationships.

The valid first pilot remains the current experimental evidence.

Minor diagnostics issue: the first provider failures were labeled `other_error`, while the final
stopping error was labeled `provider_failing`. A future cleanup could classify HTTP 429 and
provider-limit errors explicitly as provider errors. It is not important enough to change before the
experiment continues.

## 15. Current interpretation

The evidence currently supports:

1. Existing Modify examples are overwhelmingly simple, single-operation tasks.
2. The inspected Fix examples are single-error tasks (11 examples).
3. Schema/instance complexity and task complexity must be treated separately.
4. In the valid Create pilot, reference generation usually succeeded.
5. Once a reference existed, grounding was the only observed Create failure stage.
6. "Larger reference = worse grounding" was not supported.
7. The content/type of the checked strings appears potentially important.
8. The grounding metric may reject descriptions for missing literal technical strings.
9. Description retries did not consistently improve grounding in the first pilot.
10. Some schemas may have schema-specific reference-generation failure modes.
11. Generation cost should be considered before scaling to 7K-10K examples.

## 16. Still hypotheses / unresolved questions

Kept separate from the findings above:

- Does Create acceptance actually decrease with schema complexity?
- Does reference complexity affect acceptance after controlling for the type and number of checked strings?
- Is the grounding threshold of 0.9 too strict?
- Are some checked strings irrelevant enough that they should not determine acceptance?
- Would a semantic grounding/fidelity metric be better than literal string matching?
- Are description retries useful over repeated runs?
- Why does `gematik-test-hcps` repeatedly produce schema-invalid references?
- How often does truncation occur across a larger schema sample?
- What ratio of original-style vs complex examples should be used? The 70/30 split discussed earlier is only a candidate, not a conclusion.

## 17. Next steps

When WatsonX is available again:

1. Rerun the same 9-schema experiment with 3 repeats (`--repeats 3`).
2. Do **not** change prompts, thresholds, retries, model parameters or generation logic before that run.
3. Use a **new output directory**, so the provider-failed run is not mixed with valid results.
4. Check whether:
   - per-schema behavior is stable across repeats
   - grounding remains the main bottleneck
   - `asconfig` keeps succeeding despite its large reference
   - `one-changelog` keeps failing
   - `gematik-test-hcps` keeps failing at reference generation
   - description retries show any consistent benefit
5. Only after repeated evidence, design changes to Create generation/grounding.

Separately, future controlled experiments should test more complex Fix and Modify generation.

Possible training-data comparison:

- original ~2K baseline
- scaled original-style data
- scaled mixed-complexity data

The goal is to separate the effect of **more data** from the effect of **more complex tasks**. A
70/30 original/complex split has been discussed, but it is not yet justified by evidence; treat it
as an experimental condition, not a design decision.

## 18. Reproducibility / experiment implementation

The recent work added a dedicated Create pilot runner and configs so the experiment can be rerun
without changing normal benchmark generation:

- `src/preview_create_pilot.py` (runner), `configs/preview_create_pilot_watsonx.json` and
  `configs/preview_create_pilot_mock.json` (the mock only tests plumbing; its output is not data).
- A one-line change in `BenchmarkBuilder._load_schema_records` passes the existing `files` source
  option, so an explicit schema list can be selected. Behavior without `files` is unchanged.

The pilot:

- runs each selected schema independently, using the builder's own reference/description functions
- records failed schemas instead of skipping them
- records reference and description attempts and the grounding ratio of each attempt
- writes `results.csv`, `results.json` and `results.md`, plus the raw `llm_calls.jsonl`
- writes to an isolated output folder (default `outputs/experiments/create_complexity/<timestamp>`)
- does not change the benchmark dataset format

Example command:

```
uv run python src/preview_create_pilot.py \
  --config configs/preview_create_pilot_watsonx.json \
  --repeats 3 \
  --out-dir outputs/experiments/create_complexity/<new-directory>
```

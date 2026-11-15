"""Publish the Scale Only 7K dataset to a NEW branch of our private Hugging Face repo. Never touches `main`, never the original.

    benchson-research/benchson-data-experiments   branch `scale-only-7k`   (created from the verified `main`, private repo)

Three separate steps; only `upload` writes to Hugging Face, and it asks you to type the destination first.

    python src/publish_scale_only.py build          OFFLINE, no token. Verifies the real 7K JSONL export against the pinned baseline
                                                    and the merged dataset, then writes release/scale-only-7k/ (data, card, PROVENANCE).
    python src/publish_scale_only.py check-release  OFFLINE. Re-verifies a release folder against its own PROVENANCE.json
                                                    (run it after copying the folder to another computer).
    python src/publish_scale_only.py upload         needs HF_TOKEN. Pre-checks, then creates the branch and ONE commit.
    python src/publish_scale_only.py verify-remote  needs HF_TOKEN. Re-downloads the branch commit and compares every file with the
                                                    release folder; checks that `main` and the repo's privacy are unchanged.

`build` can run where the data lives (the BGU cluster); `upload` and `verify-remote` where the token is (the MacBook), after
copying only the release folder. Nothing is normalized, filtered, corrected or rewritten: the data files are byte copies of the
export, and every line is checked against its source. The token is read from HF_TOKEN / `hf auth login`, never accepted as an
argument, never printed, never written to a file.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import merge_scale_dataset as msd                                            # noqa: E402
from data_generation.published import TASKS, encode_record, git_blob_oid, sha256_file   # noqa: E402

FORMAT = "benchson-scale-only-release/1"
ORIGINAL_REPO = "aviv1ron1/Benchson"
DEFAULT_REPO = "benchson-research/benchson-data-experiments"
BRANCH = "scale-only-7k"
MAIN_COMMIT = "1e8aad270c7f0e1f82d139bbe1e32a345188b697"        # our verified baseline copy (docs/scale_only_experiment.md)
ORIGINAL_ROWS = {"train": 1000, "test": 150}
NEW_ROWS = {"create": 1334, "fix": 1333, "modify": 1333}
DATA_FILES = [f"{t}/{s}.jsonl" for t in TASKS for s in ("train", "test")]
AUDIT_FILES = ("quality_report.md", "summary.json", "provenance_report.json")
AUDIT_LOCAL_ONLY = ("metrics_by_task.csv", "validation_failures.csv", "complexity_metrics.csv", "duplicates_and_reuse.csv")
STATIC_FILES = ("README.md", "NOTICE.md", "ORIGINAL_DATASET_CARD.md", "PROVENANCE.json", "row_origin.tsv")
PRIVATE_PATH = re.compile(r"(/Users/|/home/|/sessions/|/scratch|/mnt/|/tmp/|C:\\\\Users)")
KEY_METRICS = (("json_fields", "JSON fields"), ("json_chars", "JSON characters"), ("schema_bytes", "Schema bytes"), ("schema_refs", "Schema `$ref` count"))


class Abort(Exception):
    pass


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _read_lines(path):
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _files(d):
    return sorted(n for n in os.listdir(d) if n.endswith(".json")) if os.path.isdir(d) else []


def _row_for(path):
    return (json.dumps(encode_record(_load(path))) + "\n").encode("utf-8").rstrip(b"\n")


def _diff_counts(want, got):
    miss, extra = want - got, got - want
    return sum(miss.values()), sum(extra.values())


# ------------------------------------------------------------------ build: verification of the export

def verify_export(a, problems):
    """Returns (rows, facts). rows[task][split] = [(line_bytes, origin)] in export order; origin is 'original' or 'new'."""
    expected = {t: {"train": ORIGINAL_ROWS["train"] + a.new_rows[t], "test": ORIGINAL_ROWS["test"]} for t in TASKS}
    try:
        prov, _records = msd.verify_baseline(a.baseline, a.expected_revision, ORIGINAL_ROWS)
    except Exception as e:
        raise Abort(f"the baseline is not the verified pinned dataset: {str(e)[:600]}")
    facts = {"baseline": prov, "files": {}}
    rows = {t: {} for t in TASKS}
    for t in TASKS:
        for s in ("train", "test"):
            rel = f"{t}/{s}.jsonl"
            path = os.path.join(a.export, t, s + ".jsonl")
            if not os.path.isfile(path):
                problems.append(f"{rel}: missing in the export")
                continue
            got = _read_lines(path)
            if len(got) != expected[t][s]:
                problems.append(f"{rel}: {len(got)} rows, expected {expected[t][s]}")
            if any(not ln.strip() for ln in got):
                problems.append(f"{rel}: contains an empty line")
            c = collections.Counter(got)
            if any(n > 1 for n in c.values()):
                problems.append(f"{rel}: {sum(n - 1 for n in c.values() if n > 1)} duplicate line(s) inside the file")
            for ln in got:
                try:
                    json.loads(ln)
                except ValueError:
                    problems.append(f"{rel}: a line is not valid JSON")
                    break
            base = collections.Counter(_read_lines(os.path.join(a.baseline, "original_files", t, s + ".jsonl")))
            new = collections.Counter()
            if s == "train":
                d = os.path.join(a.new, f"benchmark_{t}", "train")
                new = collections.Counter(_row_for(os.path.join(d, n)) for n in _files(d))
                if sum(new.values()) != a.new_rows[t]:
                    problems.append(f"{rel}: the new-run directory holds {sum(new.values())} examples, expected {a.new_rows[t]}")
            lost, extra = _diff_counts(base + new, c)
            if lost or extra:
                problems.append(f"{rel}: {lost} expected line(s) missing and {extra} unexpected line(s) "
                                "(expected = original published rows byte for byte + the new examples encoded from the new-run files)")
            if a.merged:                                                     # independent second source: the merged dataset itself
                d = os.path.join(a.merged, f"benchmark_{t}", s)
                m = collections.Counter(_row_for(os.path.join(d, n)) for n in _files(d))
                lost, extra = _diff_counts(m, c)
                if lost or extra:
                    problems.append(f"{rel}: differs from the merged dataset ({lost} missing, {extra} unexpected)")
            seen_new = collections.Counter(new)
            origin = []
            for ln in got:
                if base[ln] > 0:
                    origin.append("original")
                elif seen_new[ln] > 0:
                    origin.append("new")
                else:
                    origin.append("unknown")
            rows[t][s] = list(zip(got, origin))
            facts["files"][rel] = {"rows": len(got), "original_rows": origin.count("original"), "new_rows": origin.count("new"),
                                   "unexpected_rows": origin.count("unknown"), "sha256": sha256_file(path),
                                   "git_blob_id": git_blob_oid(path), "bytes": os.path.getsize(path)}
    for t in TASKS:                                                           # a row may never be in both splits
        both = {ln for ln, _ in rows[t].get("train", [])} & {ln for ln, _ in rows[t].get("test", [])}
        if both:
            problems.append(f"{t}: {len(both)} identical row(s) in train and test")
    return rows, facts


def verify_audit(a, problems):
    d = a.audit_dir
    out = {"files": {}}
    for name in AUDIT_FILES + AUDIT_LOCAL_ONLY:
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            problems.append(f"audit output missing: {name}")
        else:
            out["files"][name] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p), "published": name in AUDIT_FILES}
    if problems and any(x.startswith("audit output missing") for x in problems):
        return out
    s = _load(os.path.join(d, "summary.json"))
    out["verdicts"] = {k: (s.get(k) or {}).get("verdict") for k in ("data_integrity", "training_readiness", "scientific_comparability")}
    out["validator"] = s.get("validator")
    out["counts"] = s.get("counts")
    if not str(out["verdicts"]["data_integrity"]).startswith("INTACT"):
        problems.append(f"the audit's data integrity verdict is {out['verdicts']['data_integrity']!r}, not INTACT")
    st = {c["id"]: c["status"] for c in s.get("checks", [])}
    for cid in ("A1", "A2", "B1", "B2", "B3", "B6", "C1", "C2", "C4"):          # the claims the dataset card makes
        if st.get(cid) != "PASS":
            problems.append(f"audit check {cid} is {st.get(cid)!r}, not PASS")
    for t in TASKS:
        for cid in (f"D-{t}-new", f"U-{t}-new", f"U-{t}-original"):
            if st.get(cid) != "PASS":
                problems.append(f"audit check {cid} is {st.get(cid)!r}, not PASS")
    pv = _load(os.path.join(d, "provenance_report.json"))
    if ((pv.get("provenance") or {}).get("baseline") or {}).get("revision") != a.expected_revision:
        problems.append("the audit was not run against the pinned baseline revision")
    for name in AUDIT_FILES:                                                  # nothing machine-specific may be published
        text = open(os.path.join(d, name), encoding="utf-8").read()
        n = len(PRIVATE_PATH.findall(text))
        if n:
            problems.append(f"audit file {name} contains {n} machine-specific path(s); it cannot be published as is")
    out["reuse"] = _reuse_stats(s)
    out["complexity"] = _complexity(d)
    out["original_defects"] = _defects(d)
    return out


def _reuse_stats(s):
    per = (s.get("reuse") or {}).get("references_per_schema_id") or {}
    hist = collections.Counter(per.values())
    return {"cap": (s.get("reuse") or {}).get("cap"), "schema_ids_with_new_references": len(per), "max_new_references_on_one_schema": max(per.values(), default=0),
            "histogram_new_references_per_schema": {str(k): hist[k] for k in sorted(hist)}}


def _complexity(d):
    out = {}
    with open(os.path.join(d, "metrics_by_task.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["metric"] in dict(KEY_METRICS) and r.get("relative_delta_mean_pct") not in (None, ""):
                out.setdefault(r["task"], {})[r["metric"]] = {"original_mean": r.get("original_mean"), "new_mean": r.get("new_mean"),
                                                              "relative_delta_pct": r["relative_delta_mean_pct"],
                                                              "ci95_cluster_bootstrap": r.get("delta_mean_ci95_cluster_bootstrap")}
    return out


def _defects(d):
    with open(os.path.join(d, "validation_failures.csv"), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["origin"] == "original"]
    genuine = [{"task": r["task"], "example": r["name"], "field": r["field"], "category": r["category"]}
               for r in rows if r["category"].startswith("genuine_violation") or r["category"] == "expected_invalid_but_valid"]
    is_genuine = lambda r: r["category"].startswith("genuine_violation") or r["category"] == "expected_invalid_but_valid"
    other = collections.Counter(r["category"].split(":")[0] for r in rows if not is_genuine(r))
    return {"genuine_validation_failures": genuine, "other_original_rows_by_category": dict(other)}


# ------------------------------------------------------------------ build: card and provenance

def _card(a, facts, audit, rev, license_id, cfg, chain_len):
    c = lambda t, s: facts["files"][f"{t}/{s}.jsonl"]
    table = "\n".join(f"| {t} | {c(t, 'train')['original_rows']} | {c(t, 'train')['new_rows']} | {c(t, 'train')['rows']} | {c(t, 'test')['rows']} |" for t in TASKS)
    tot = lambda k: sum(c(t, "train")[k] for t in TASKS)
    cfgs = "\n".join(f"- config_name: {t}\n  data_files:\n  - split: train\n    path: {t}/train.jsonl\n  - split: test\n    path: {t}/test.jsonl" for t in TASKS)
    v = audit["verdicts"]
    reuse, rc = audit["reuse"], (cfg.get("reuse") or {})
    mp = (cfg.get("llm_provider") or {}).get("params", {}).get("model_params", {})
    comp = audit["complexity"]
    ctab = "\n".join("| " + lab + " | " + " | ".join(f"{comp.get(t, {}).get(k, {}).get('relative_delta_pct', 'n/a')}%" for t in TASKS) + " |"
                     for k, lab in KEY_METRICS)
    defects = "\n".join(f"- {d['task']}: `{d['example']}`, field `{d['field']}`: {d['category']}" for d in audit["original_defects"]["genuine_validation_failures"]) or "- none found"
    val = audit.get("validator") or {}
    return f"""---
pretty_name: Benchson Scale Only 7K
license: {license_id}
task_categories:
- text-generation
tags:
- json
- json-schema
- structured-output
configs:
{cfgs}
---

# Benchson Scale Only 7K (experiment branch `{BRANCH}`)

Private research derivative of [`{ORIGINAL_REPO}`](https://huggingface.co/datasets/{ORIGINAL_REPO}) (license: {license_id.upper()}).
Branch `main` of this repository is the untouched original baseline. This branch adds newly generated training examples.

## 1. Purpose
Experiment "Scale Only": does training on more examples that follow the original generation logic help, when the number of
training examples grows from 3,000 to 7,000? The test split is the original one, so models trained on the original 3K and on this 7K
can be compared on identical test examples.

## 2. Original baseline
`{ORIGINAL_REPO}` at the pinned revision `{rev}`. Our verified copy is `main` of this repository (commit `{MAIN_COMMIT}`).
The original 3,000 training rows and 450 test rows are contained here **byte for byte** (each line of the original JSONL files
appears unchanged in the files of this branch). Nothing was normalized, filtered, corrected or rewritten.

## 3. Generator and configuration of the new examples
- Generator model: {a.generator_model} (served with vLLM). The exact served model identifier is not recorded in this card.
- Same generation logic, prompts, validation thresholds (grounding 0.9, minimum populated fields 2, at most 3 build attempts), schema
  corpus (`schemas` + `jsb_schemas`, 10 KB limit) and train gates (deterministic, no round-trip) as the original; complex generation off
  (`complex_ratio` {cfg.get('generation', {}).get('complex_ratio')}, standard Modify strategy `{cfg.get('generation', {}).get('standard_modify_strategy')}`).
- Sampling parameters used: temperature {mp.get('temperature')}, top_p {mp.get('top_p')}, max_tokens {mp.get('max_tokens')}.
  These are **our provisional values**; the original generation settings could not be recovered (see the repository's `docs/original_generation_settings.md`).
- Seed {cfg.get('seed')}, concurrency {cfg.get('concurrency')}, max retries {cfg.get('max_retries')}.
- Differences from the original process (both are part of the experiment): a different generator model (original: `openai/gpt-oss-120b`),
  and controlled schema reuse (next section).

## 4. Composition
| Task | Original train | New train | Total train | Test |
|---|---:|---:|---:|---:|
{table}
| Total | {tot('original_rows')} | {tot('new_rows')} | {tot('rows')} | {sum(c(t, 'test')['rows'] for t in TASKS)} |

Rows inside each file are ordered by example id, so original and new rows are interleaved. `row_origin.tsv` lists, for every line,
whether it is an original or a new row (with the SHA256 of the line).

## 5. Controlled schema reuse
The original dataset used one example per task per schema. Here several independent examples may come from the same schema:
every attempt mints a fresh reference with the unchanged standard prompt; a reference equal to a published or earlier new one for that
schema is rejected. At most {rc.get('max_new_references_per_schema')} new references per schema ({reuse['schema_ids_with_new_references']} schemas received new references;
maximum observed on one schema: {reuse['max_new_references_on_one_schema']}; histogram {reuse['histogram_new_references_per_schema']}).
Schemas that occur in the published test split (by id or content) are never used. Examples sharing a schema are not independent.

## 6. Generation and merge provenance
Generation ran as a chain of {chain_len} chunks on the BGU cluster; the chunk chain (state files, seals, ledger) was verified with the audit's
`--chunks-root` option. The 4,000 new examples were merged with the verified baseline by `src/merge_scale_dataset.py`. The JSONL files here are
the merge's export, re-verified before upload against the pinned baseline and against the merged dataset. `PROVENANCE.json` lists the
SHA256 and git blob id of every file, the audit files and their hashes, and the source revisions.

## 7. Quality audit (`src/audit_scale_dataset.py`, jsonschema {val.get('jsonschema_installed')})
- Data integrity: **{v['data_integrity']}**
- Training readiness: **{v['training_readiness']}**
- Scientific comparability: **{v['scientific_comparability']}**
- Every new example passed the automated schema and task checks; no new example is an exact duplicate; no new example leaks into the test split;
  all original test examples are unchanged; remote `$ref` documents were resolved at audit time.
- The full report is in `audit/quality_report.md`, with `audit/summary.json` and `audit/provenance_report.json`. The bulky per-example files
  (`metrics_by_task.csv`, `validation_failures.csv`, `complexity_metrics.csv`, `duplicates_and_reuse.csv`) are not uploaded; their SHA256 and
  sizes are in `PROVENANCE.json`, and they are kept with the audit output of the experiment (`outputs/quality_audit/scale_only_7k_20261114/`).
- Schema validation shows that schema compliance, not semantic correctness, was checked.

## 8. Known defects of the ORIGINAL dataset (kept exactly as published)
{defects}

Other limitation of the original benchmark: some schema contents occur in both its train and test splits under different ids (see the
`main` branch card). These examples are preserved; the status "REVIEW REQUIRED" of the audit refers to them.

## 9. Lower measured structural complexity of the new examples
Relative difference of the mean, new vs original training examples:

| Metric | Create | Fix | Modify |
|---|---:|---:|---:|
{ctab}

This is a descriptive comparison, not evidence of a mechanism. Its causes have not been investigated yet.

## 10. Scientific limitations and intended comparison
- Intended use: train the same model with the same recipe on (a) the original 3K (`main`) and (b) this 7K, and evaluate both on the identical original test split.
- The 7K differs from the 3K in three ways at once: more examples, a different generator model, and controlled schema reuse. The new examples are
  also structurally simpler on average. A difference in results cannot be attributed to any single one of these.
- Semantic correctness of descriptions, instructions and fixes was not assessed. Lexical grounding is a proxy only.
- Original generation settings are not recoverable; sampling values above are ours.

## Attribution and license
Data: [`{ORIGINAL_REPO}`](https://huggingface.co/datasets/{ORIGINAL_REPO}) @ `{rev}`, license {license_id.upper()}; the original card is in `ORIGINAL_DATASET_CARD.md`, see `NOTICE.md`.
Schemas: JSONSchemaStore (MIT) and JSONSchemaBench (MIT, arXiv:2501.10868), as stated by the original dataset.
"""


# ------------------------------------------------------------------ build

def cmd_build(a):
    out = os.path.abspath(a.out_dir)
    inputs = [a.export, a.baseline, a.new, a.audit_dir] + ([a.merged] if a.merged else [])
    for p in inputs:
        if not os.path.isdir(p):
            sys.exit(f"input directory not found: {p}")
        if msd._overlap(out, p):
            sys.exit("--out-dir must be outside every input directory and must not contain one.")
    if os.path.lexists(out):
        sys.exit(f"{out} already exists. Refusing to overwrite or reuse it; choose a new --out-dir.")
    problems = []
    try:
        rows, facts = verify_export(a, problems)
    except Abort as e:
        sys.exit(f"BUILD FAILED: {e}")
    audit = verify_audit(a, problems)
    pub = _load(os.path.join(a.baseline, "published_manifest.json"))
    card = pub.get("original_card") or {}
    card_src = os.path.join(a.baseline, "original_card", "README.md")
    if not card.get("license") or not os.path.isfile(card_src) or sha256_file(card_src) != card.get("sha256"):
        problems.append("the baseline has no verified original dataset card / license")
    if problems:
        print("BUILD FAILED. Nothing was written.")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    cfg = _load(a.config)
    try:
        chain_len = len(_load(os.path.join(a.new, "merged_chunks_manifest.json")).get("chain", []))
    except (OSError, ValueError):
        chain_len = "an unknown number of"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".merge_staging_", dir=os.path.dirname(out))
    created = {staging: os.path.realpath(staging)}
    try:
        files = {}
        for rel in DATA_FILES:
            os.makedirs(os.path.dirname(os.path.join(staging, rel)), exist_ok=True)
            shutil.copyfile(os.path.join(a.export, rel), os.path.join(staging, rel))
            if sha256_file(os.path.join(staging, rel)) != facts["files"][rel]["sha256"]:
                raise Abort(f"{rel}: copy differs from the export")
        os.makedirs(os.path.join(staging, "audit"))
        for name in AUDIT_FILES:
            shutil.copyfile(os.path.join(a.audit_dir, name), os.path.join(staging, "audit", name))
        shutil.copyfile(card_src, os.path.join(staging, "ORIGINAL_DATASET_CARD.md"))
        import prepare_hf_release as phr
        with open(os.path.join(staging, "NOTICE.md"), "w", encoding="utf-8") as f:
            f.write(phr._notice(a.repo_id, pub["revision"], card["license"], card))
        with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
            f.write(_card(a, facts, audit, pub["revision"], card["license"], cfg, chain_len))
        with open(os.path.join(staging, "row_origin.tsv"), "w", encoding="utf-8") as f:
            f.write("task\tsplit\tline\torigin\tline_sha256\n")
            for t in TASKS:
                for s in ("train", "test"):
                    for i, (ln, origin) in enumerate(rows[t][s], 1):
                        f.write(f"{t}\t{s}\t{i}\t{origin}\t{hashlib.sha256(ln).hexdigest()}\n")
        for name in STATIC_FILES + tuple(f"audit/{n}" for n in AUDIT_FILES):
            if name != "PROVENANCE.json" and os.path.isfile(os.path.join(staging, name)):
                files[name] = {"sha256": sha256_file(os.path.join(staging, name)), "bytes": os.path.getsize(os.path.join(staging, name))}
        for rel in DATA_FILES:
            files[rel] = {k: facts["files"][rel][k] for k in ("sha256", "git_blob_id", "bytes", "rows", "original_rows", "new_rows")}
        prov = {"format": FORMAT, "created": _now(), "repo": a.repo_id, "branch": BRANCH, "base_branch": "main", "expected_main_commit": a.expect_main,
                "original_dataset": ORIGINAL_REPO, "original_revision": pub["revision"], "license": card["license"],
                "counts": {t: {s: facts["files"][f"{t}/{s}.jsonl"]["rows"] for s in ("train", "test")} for t in TASKS},
                "originals_byte_identical": True, "new_rows_checked_against": ["new-run directory", "merged dataset" if a.merged else "(merged dataset not given)"],
                "files": files, "audit": {"verdicts": audit["verdicts"], "validator": audit["validator"], "files": audit["files"],
                                          "complexity": audit["complexity"], "reuse": audit["reuse"], "original_defects": audit["original_defects"]},
                "generator_model": a.generator_model, "config_sha256": sha256_file(a.config)}
        with open(os.path.join(staging, "PROVENANCE.json"), "x", encoding="utf-8") as f:
            json.dump(prov, f, indent=1, sort_keys=True)
        problems = check_release(staging)
        if problems:
            raise Abort("the release folder failed its own check: " + "; ".join(problems))
        msd._publish(staging, out)
    except BaseException:
        msd._remove_own_staging(staging, created, [os.path.abspath(p) for p in inputs])
        raise
    print(f"Release folder written: {out}\n  {len(files)} files; PROVENANCE.json lists every SHA256. Nothing was uploaded.")
    for rel in DATA_FILES:
        print(f"  {rel:22s} {files[rel]['rows']:5d} rows  sha256 {files[rel]['sha256']}")
    print("\nNext: python src/publish_scale_only.py check-release --release-dir", out)


# ------------------------------------------------------------------ check-release

def check_release(rel_dir):
    problems = []
    try:
        prov = _load(os.path.join(rel_dir, "PROVENANCE.json"))
    except (OSError, ValueError) as e:
        return [f"PROVENANCE.json unreadable: {type(e).__name__}"]
    if prov.get("format") != FORMAT or prov.get("branch") != BRANCH:
        problems.append("PROVENANCE.json is not a scale-only-7k release manifest")
    want = set(prov.get("files", {})) | {"PROVENANCE.json"}
    have = {os.path.relpath(os.path.join(d, f), rel_dir).replace(os.sep, "/") for d, _, fs in os.walk(rel_dir) for f in fs}
    if want != have:
        problems.append(f"file set differs: missing {sorted(want - have)}, extra {sorted(have - want)}")
    for name, info in prov.get("files", {}).items():
        p = os.path.join(rel_dir, name)
        if not os.path.isfile(p):
            continue
        if sha256_file(p) != info["sha256"] or os.path.getsize(p) != info["bytes"]:
            problems.append(f"{name}: differs from PROVENANCE.json")
        if name in DATA_FILES:
            n = len(_read_lines(p))
            t, s = name.split("/")[0], name.split("/")[1][:-6]
            if n != info["rows"] or n != prov["counts"][t][s]:
                problems.append(f"{name}: {n} rows, manifest says {info['rows']}")
            if git_blob_oid(p) != info["git_blob_id"]:
                problems.append(f"{name}: git blob id differs")
    for t in TASKS:
        if prov.get("counts", {}).get(t, {}).get("train") != ORIGINAL_ROWS["train"] + NEW_ROWS[t] or prov["counts"][t].get("test") != ORIGINAL_ROWS["test"]:
            problems.append(f"{t}: counts differ from the expected experiment composition")
    for name in ("README.md", "audit/quality_report.md", "audit/summary.json", "audit/provenance_report.json", "PROVENANCE.json"):
        p = os.path.join(rel_dir, name)
        if os.path.isfile(p):
            text = open(p, encoding="utf-8").read()
            if PRIVATE_PATH.search(text) or re.search(r"hf_[A-Za-z0-9]{30,}", text):
                problems.append(f"{name}: contains a machine-specific path or a token-like string")
    return problems


def cmd_check_release(a):
    problems = check_release(a.release_dir)
    if problems:
        print("check-release: FAILED")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("check-release: PASSED (every file matches PROVENANCE.json; counts, git blob ids and sizes agree)")


# ------------------------------------------------------------------ Hugging Face steps

def _hub():
    import hf_baseline
    return hf_baseline._hub()


def _hfb():
    import hf_baseline
    return hf_baseline


def _head(api, repo, revision):
    return api.dataset_info(repo, revision=revision).sha


def cmd_upload(a):
    hfb = _hfb()
    hfb._refuse_original(a.repo_id)
    problems = check_release(a.release_dir)
    if problems:
        sys.exit("release folder check failed: " + "; ".join(problems))
    prov = _load(os.path.join(a.release_dir, "PROVENANCE.json"))
    hub = _hub()
    api = hub.HfApi(token=hfb._token())
    problems = hfb._auth_problems(api, a.repo_id, a.allow_broad_token)
    info = api.dataset_info(a.repo_id)
    if not info.private:
        problems.append("the repo is PUBLIC; make it private first")
    main_sha = _head(api, a.repo_id, "main")
    if main_sha != a.expect_main:
        problems.append(f"main is at {main_sha}, expected {a.expect_main}; refusing to branch from an unexpected state")
    branches = [b.name for b in api.list_repo_refs(a.repo_id, repo_type="dataset").branches]
    if BRANCH in branches:
        problems.append(f"branch {BRANCH} already exists; nothing is overwritten")
    main_files = sorted(t.path for t in api.list_repo_tree(a.repo_id, repo_type="dataset", revision=main_sha, recursive=True) if getattr(t, "size", None) is not None)
    hfb._report(problems, "pre-upload checks")
    plan = sorted(prov["files"]) + ["PROVENANCE.json"]
    print(f"\nAbout to create branch {BRANCH} from main ({main_sha}) in {a.repo_id} (private) and add ONE commit with {len(plan)} files:")
    for f in plan:
        print(f"   {f}  ({os.path.getsize(os.path.join(a.release_dir, f))} bytes)")
    print(f"Files on main that this commit replaces on the branch: {sorted(set(main_files) & set(plan))}. `main` itself is never written.")
    if not sys.stdin.isatty():
        sys.exit("Interactive confirmation required (run this in a terminal).")
    want = f"{a.repo_id}:{BRANCH}"
    if input(f"Type '{want}' to upload: ").strip() != want:
        sys.exit("Not confirmed. Nothing was uploaded.")
    api.create_branch(a.repo_id, branch=BRANCH, repo_type="dataset", revision=main_sha, exist_ok=False)
    try:
        commit = api.upload_folder(repo_id=a.repo_id, repo_type="dataset", folder_path=a.release_dir, revision=BRANCH,
                                   ignore_patterns=[".DS_Store", "**/.DS_Store"],
                                   commit_message="Scale Only 7K: original 3K train + 4K generated (Granite, controlled schema reuse); test unchanged",
                                   commit_description=f"Original revision {prov['original_revision']}. See README.md and PROVENANCE.json.")
    except BaseException as e:
        print(f"\nUPLOAD FAILED after the branch was created ({hfb._err(e)}). The branch {BRANCH} exists and is a copy of main; it was NOT deleted. "
              "Decide with the team whether to retry the upload into it or to remove it.", file=sys.stderr)
        raise
    sha = getattr(commit, "oid", None) or _head(api, a.repo_id, BRANCH)
    after = _head(api, a.repo_id, "main")
    receipt = {"repo": a.repo_id, "branch": BRANCH, "commit": sha, "main_before": main_sha, "main_after": after, "uploaded": _now(),
               "files": plan}
    with open(os.path.join(os.path.dirname(os.path.abspath(a.release_dir)), "scale-only-7k.upload_receipt.json"), "x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1)
    print(f"\nUploaded. Branch commit: {sha}\nmain before/after: {main_sha} / {after}" + ("" if after == main_sha else "   <-- MAIN CHANGED, investigate"))
    print("Next: python src/publish_scale_only.py verify-remote --release-dir", a.release_dir, "--commit", sha)


def cmd_verify_remote(a):
    hfb = _hfb()
    hfb._refuse_original(a.repo_id)
    local = check_release(a.release_dir)
    prov = _load(os.path.join(a.release_dir, "PROVENANCE.json"))
    hub = _hub()
    tok = hfb._token()
    api = hub.HfApi(token=tok)
    sha = a.commit or _head(api, a.repo_id, BRANCH)
    print(f"Verifying {a.repo_id} @ {sha}")
    problems = [f"local release folder: {p}" for p in local]
    info = api.dataset_info(a.repo_id)
    if not info.private:
        problems.append("the repository is PUBLIC")
    if _head(api, a.repo_id, BRANCH) != sha:
        problems.append(f"branch {BRANCH} is not at {sha} any more")
    main_sha = _head(api, a.repo_id, "main")
    if main_sha != a.expect_main:
        problems.append(f"main is at {main_sha}, expected {a.expect_main} (main changed)")
    main_tree = {t.path: t for t in api.list_repo_tree(a.repo_id, repo_type="dataset", revision=main_sha, recursive=True) if getattr(t, "size", None) is not None}
    known = hfb._known()["files"]
    for name in DATA_FILES:                                              # the original baseline on main is still byte-identical
        oid, size = known[name]
        t = main_tree.get(name)
        if t is None or t.size != size or (not getattr(t, "lfs", None) and t.blob_id != oid):
            problems.append(f"main/{name}: differs from the pinned original")
    tree = {t.path: t for t in api.list_repo_tree(a.repo_id, repo_type="dataset", revision=sha, recursive=True) if getattr(t, "size", None) is not None}
    names = sorted(p for p in tree if p != ".gitattributes")
    if names != sorted(set(prov["files"]) | {"PROVENANCE.json"}):
        problems.append(f"unexpected file set on the branch: {names}")
    cache = tempfile.mkdtemp(prefix="benchson_verify_")
    try:
        for name in sorted(set(prov["files"]) | {"PROVENANCE.json"}):
            if name not in tree:
                continue
            got = hub.hf_hub_download(a.repo_id, name, repo_type="dataset", revision=sha, cache_dir=cache, token=tok)
            h = sha256_file(got)
            want = prov["files"][name]["sha256"] if name != "PROVENANCE.json" else sha256_file(os.path.join(a.release_dir, name))
            if h != want or h != sha256_file(os.path.join(a.release_dir, name)):
                problems.append(f"{name}: sha256 differs from the release folder")
            elif name in DATA_FILES:
                n = len(_read_lines(got))
                t_, s_ = name.split("/")[0], name.split("/")[1][:-6]
                if n != prov["counts"][t_][s_]:
                    problems.append(f"{name}: {n} rows, expected {prov['counts'][t_][s_]}")
                else:
                    print(f"  ok  {name}  {n} rows  sha256 {h[:16]}")
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    out = {"repo": a.repo_id, "branch": BRANCH, "commit": sha, "main": main_sha, "private": bool(info.private), "checked": _now(),
           "result": "FAIL" if problems else "PASS", "problems": problems}
    with open(os.path.join(os.path.dirname(os.path.abspath(a.release_dir)), "scale-only-7k.verify_receipt.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    hfb._report(problems, "verify-remote")
    print(f"Record: {a.repo_id} branch {BRANCH} = commit {sha}; main = {main_sha} (unchanged); private = {info.private}")


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-id", default=DEFAULT_REPO)
    common.add_argument("--expect-main", default=MAIN_COMMIT, help="commit that main must be at (our verified baseline copy)")
    b = sub.add_parser("build", parents=[common])
    b.add_argument("--export", required=True, help="hf_export_7k_<date>/ (the six JSONL files)")
    b.add_argument("--baseline", required=True, help="data_published/<sha12>")
    b.add_argument("--new", required=True, help="merged new run, e.g. data_scale_only/merged_4k_<date>")
    b.add_argument("--merged", help="data_scale_7k_<date>/ (second, independent check of every row)")
    b.add_argument("--audit-dir", required=True, help="outputs/quality_audit/scale_only_7k_<date>")
    b.add_argument("--config", default=os.path.join(os.path.dirname(SRC), "configs", "scale_only_vllm_granite.json"))
    b.add_argument("--out-dir", default="release/scale-only-7k")
    b.add_argument("--expected-revision", default=msd.PINNED_REVISION)
    b.add_argument("--generator-model", default="IBM Granite 4.0 H Small")
    b.set_defaults(fn=cmd_build, new_rows=NEW_ROWS)
    c = sub.add_parser("check-release", parents=[common])
    c.add_argument("--release-dir", default="release/scale-only-7k")
    c.set_defaults(fn=cmd_check_release)
    for name, fn in (("upload", cmd_upload), ("verify-remote", cmd_verify_remote)):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--release-dir", default="release/scale-only-7k")
        p.add_argument("--allow-broad-token", action="store_true")
        if name == "verify-remote":
            p.add_argument("--commit", help="verify this commit (printed by upload) instead of the head of the branch")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

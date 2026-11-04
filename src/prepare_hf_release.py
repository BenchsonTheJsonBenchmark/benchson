"""Prepare LOCAL upload folders for a new Hugging Face dataset repo. Never touches the network.

Nothing is created, pushed or uploaded by this script. It builds `release/<variant>/` (data files,
dataset card, PROVENANCE.json) and writes `release/PUBLISH_COMMANDS.md` with the commands to run
later, by hand, after explicit approval. It refuses to target the original repo.

Branches of the NEW repo:
    main            the pinned original baseline, byte-identical to aviv1ron1/Benchson @ <sha>
    scale-only-7k   original train (verbatim) + newly generated 100% standard examples; test identical
    mixed-80-20     original train (verbatim) + newly generated 80/20 standard/complex examples; test identical

    python src/prepare_hf_release.py --variant main --baseline data_published/<sha12> --new-repo-id <owner>/<name>
    python src/prepare_hf_release.py --variant scale-only-7k --baseline data_published/<sha12> \
        --new-dir data_scale_only/<run> --new-repo-id <owner>/<name>
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generation.published import (TASKS, encode_record, git_blob_oid, overlap_report, reference_hash, schema_hash,
                                       sha256_file, task_keys)

ORIGINAL_REPO = "aviv1ron1/Benchson"
VARIANTS = {"main", "scale-only-7k", "mixed-80-20"}


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              capture_output=True, text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()
    except OSError:
        return None


def _new_rows(new_dir, task, reserved):
    d = os.path.join(new_dir, f"benchmark_{task}", "train")
    if not os.path.isdir(d):
        sys.exit(f"missing {d}")
    if os.path.isdir(os.path.join(new_dir, f"benchmark_{task}", "test")) and \
            [f for f in os.listdir(os.path.join(new_dir, f"benchmark_{task}", "test")) if f.endswith(".json")]:
        sys.exit("the new run contains test examples; the test split must stay the original one")
    rows = []
    same_task_ids = set(reserved["ids_by_task"][task]["train"]) | set(reserved["ids_by_task"][task]["test"])
    test_hashes = set(reserved["schema_hashes"]["test"])
    for name in sorted(f for f in os.listdir(d) if f.endswith(".json")):
        rec = json.load(open(os.path.join(d, name), encoding="utf-8"))
        if name[:-5] in same_task_ids:
            sys.exit(f"{task}/{name}: same task+schema id already in the original dataset")
        if schema_hash(rec["schema"]) in test_hashes:
            sys.exit(f"{task}/{name}: schema content is in the original TEST split")
        exact, _ = task_keys(task, rec)
        if exact in set(reserved.get("task_keys", {}).get(task, {}).get("exact", [])):
            sys.exit(f"{task}/{name}: exact duplicate of a published {task} example")
        if reference_hash(task, rec) in set(reserved.get("reference_hashes_by_schema", {}).get(schema_hash(rec["schema"]), [])):
            sys.exit(f"{task}/{name}: same reference as a published example of this schema")
        rows.append(json.dumps(encode_record(rec)))
    return rows


def _limitation_text(ov):
    """Known limitation of the ORIGINAL benchmark (kept as published; nothing is removed or changed)."""
    n = ov["shared_schema_contents"]
    if not n:
        return ""
    ident = max(max(d["identical_examples_incl_schema"], d["identical_examples_ignoring_schema"],
                    d["same_schema_and_same_reference"], d["identical_rows"]) for d in ov["per_task"].values())
    return (f"{n} schema contents (identical JSON Schema content, compared by canonical hash) occur in both the train and the test "
            "split of the original dataset, under different example ids. The original split was made per schema file, so a "
            "schema stored twice under different ids could land on both sides. "
            + (f"No example is identical across train and test under any of four comparisons (same task fields with the schema, "
               f"same task fields ignoring the schema, same reference object on the same schema, identical rows); "
               if ident == 0 else
               f"Example-level overlap exists: up to {ident} identical examples in one task (see PROVENANCE.json); ")
            + "only the schema repeats. This is a property of the original published benchmark. It is kept unchanged here: "
              "no example was removed or modified. Results on the test examples of these schemas should be read with this in mind. "
              "New training examples added on other branches use only schemas whose content is not in the test split.")


def _card(variant, repo_id, rev, counts, new_info, files, license_id, limitation=""):
    base = (f"This repository holds **local experiment derivatives** of [`{ORIGINAL_REPO}`](https://huggingface.co/datasets/{ORIGINAL_REPO}) "
            f"(revision `{rev}`, license: {license_id.upper()}). The original dataset is unchanged and is not modified by this repository.")
    desc = {"main": "Branch `main` is the original baseline: the six data files (`create|fix|modify` x `train|test`) are byte-identical "
                    "to the pinned revision (3,000 train / 450 test). Only the dataset card and the added notices differ.",
            "scale-only-7k": "Branch `scale-only-7k`: the original train examples (verbatim, first) plus newly generated examples "
                             "using the original, standard generation logic. The test split is byte-identical to the original.",
            "mixed-80-20": "Branch `mixed-80-20`: the original train examples (verbatim, first) plus newly generated examples with an "
                           "80% standard / 20% complex mix. The test split is byte-identical to the original."}[variant]
    lim = ""
    if variant != "main":
        lim = ("\n## Provenance and limitations\n"
               f"- Original examples were generated with `openai/gpt-oss-120b` (round-trip validator `Qwen/Qwen3.5-397B-A17B-FP8`).\n"
               f"- New examples: model `{new_info.get('model')}`, seed `{new_info.get('seed')}`, code commit `{new_info.get('commit')}`. "
               "A different generator model means these additions are not an exact reproduction of the original process, "
               "even though the generation logic and quality thresholds are the same.\n"
               "- Schemas of the new examples are disjoint from every schema of the original train and test splits.\n"
               "- See `PROVENANCE.json` for file hashes and the generation manifest.\n")
    cfgs = "\n".join(f"- config_name: {t}\n  data_files:\n  - split: train\n    path: {t}/train.jsonl\n  - split: test\n    path: {t}/test.jsonl"
                     for t in TASKS)
    table = "\n".join(f"| `{t}` | {counts[t]['train']} | {counts[t]['test']} |" for t in TASKS)
    return (f"---\npretty_name: Benchson derivative ({variant})\nlicense: {license_id}\ntask_categories:\n- text-generation\n"
            f"tags:\n- json\n- json-schema\n- structured-output\nconfigs:\n{cfgs}\n---\n\n# Benchson derivative - `{variant}`\n\n"
            f"{base}\n\n{desc}\n\n| config | train | test |\n|---|---|---|\n{table}\n{lim}\n"
            "Fields are as in the original dataset (JSON-valued fields are stored as strings).\n\n"
            + (f"\n## Known limitation of the original benchmark\n{limitation}\n\n" if limitation else "\n") +
            "## Attribution and license\n"
            f"- Data: [`{ORIGINAL_REPO}`](https://huggingface.co/datasets/{ORIGINAL_REPO}) @ `{rev}`, license {license_id.upper()}. "
            "The original card is kept verbatim in `ORIGINAL_DATASET_CARD.md`; see `NOTICE.md`.\n"
            "- Schemas: JSONSchemaStore (MIT) and JSONSchemaBench "
            "([`epfl-dlab/JSONSchemaBench`](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench), MIT, arXiv:2501.10868), "
            "as stated by the original dataset.\n"
            "- `PROVENANCE.json` lists the sha256 and git blob id of every data file and the revision it was copied from.\n")


def _notice(repo_id, rev, license_id, card, limitation=""):
    limit_block = f"\n## Known limitation of the original benchmark\n\n{limitation}\n" if limitation else ""
    return f"""# Notice and attribution

This repository (`{repo_id}`) is a derivative workspace for experiments on the Benchson dataset.

## Original dataset

- Dataset: `{ORIGINAL_REPO}` (https://huggingface.co/datasets/{ORIGINAL_REPO})
- Revision copied: `{rev}`
- License stated by the original dataset card: **{license_id.upper()}** ("This dataset is licensed MIT").
  The original card is reproduced unchanged in `ORIGINAL_DATASET_CARD.md` (sha256 `{card['sha256']}`).
- The original card names no copyright holder or year, so none is stated here. If the authors provide a
  copyright line, add it to this notice.

## Upstream sources named by the original dataset

- JSONSchemaStore schemas (MIT).
- JSONSchemaBench, `epfl-dlab/JSONSchemaBench` (MIT), arXiv:2501.10868.
  The `source` and `subset` fields of each row record the origin.

## Generation models named by the original dataset

- Ground truth generated by `openai/gpt-oss-120b`; round-trip verified by `Qwen/Qwen3.5-397B-A17B-FP8`.

## What this repository changes

- Branch `main`: nothing in the data. The six data files are byte-identical to the original revision.
- Other branches (added later) append new training examples after the original ones and keep the test split unchanged.
  They are described in their own `README.md` and `PROVENANCE.json`.
- The original dataset `{ORIGINAL_REPO}` is not modified by this repository.
{limit_block}"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--baseline", required=True, help="data_published/<sha12> made by import_published_benchmark.py")
    ap.add_argument("--new-dir", help="new run directory (benchmark_*/train); required for non-main variants")
    ap.add_argument("--new-repo-id", required=True, help="<owner>/<name> of YOUR new dataset repo")
    ap.add_argument("--release-root", default="release")
    args = ap.parse_args(argv)

    if args.new_repo_id.strip().lower() == ORIGINAL_REPO.lower():
        sys.exit(f"Refusing: {ORIGINAL_REPO} is the original dataset and must never be a publish target.")
    if "/" not in args.new_repo_id:
        sys.exit("--new-repo-id must look like <owner>/<name>")
    if args.variant != "main" and not args.new_dir:
        sys.exit("--new-dir is required for this variant")
    out = os.path.join(args.release_root, args.variant)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} is not empty. Refusing to overwrite.")

    pub = json.load(open(os.path.join(args.baseline, "published_manifest.json")))
    reserved = json.load(open(os.path.join(args.baseline, "reserved_schemas.json")))
    rev = reserved["revision"]
    card = pub.get("original_card")
    if not card or not card.get("license"):
        sys.exit("The baseline has no original dataset card / license (re-run the import so README.md is downloaded). "
                 "A release without a verified license is not prepared.")
    license_id = card["license"]
    card_src = os.path.join(args.baseline, "original_card", "README.md")
    if not os.path.isfile(card_src) or sha256_file(card_src) != card["sha256"]:
        sys.exit("original_card/README.md is missing or does not match the manifest.")
    orig = os.path.join(args.baseline, "original_files")
    new_info, new_manifest = {}, None
    if args.new_dir:
        mp = os.path.join(args.new_dir, "generation_manifest.json")
        if os.path.isfile(mp):
            new_manifest = json.load(open(mp))
            new_info = {"model": new_manifest.get("model"), "seed": new_manifest.get("seed"),
                        "commit": (new_manifest.get("git") or {}).get("commit")}

    ov = overlap_report(orig)
    limitation = _limitation_text(ov)
    counts, files = {}, {}
    for t in TASKS:
        os.makedirs(os.path.join(out, t), exist_ok=True)
        tr_src, te_src = (os.path.join(orig, t, f"{s}.jsonl") for s in ("train", "test"))
        shutil.copyfile(te_src, os.path.join(out, t, "test.jsonl"))
        train_bytes = open(tr_src, "rb").read()
        added = [] if args.variant == "main" else _new_rows(args.new_dir, t, reserved)
        if train_bytes and not train_bytes.endswith(b"\n"):
            train_bytes += b"\n"
        with open(os.path.join(out, t, "train.jsonl"), "wb") as f:
            f.write(train_bytes + "".join(r + "\n" for r in added).encode("utf-8"))
        n_orig = train_bytes.count(b"\n")
        counts[t] = {"train": n_orig + len(added), "test": open(te_src, "rb").read().count(b"\n"), "added": len(added)}
        if sha256_file(te_src) != sha256_file(os.path.join(out, t, "test.jsonl")):
            sys.exit(f"{t}: test split changed")
        if open(os.path.join(out, t, "train.jsonl"), "rb").read()[:len(train_bytes)] != train_bytes:
            sys.exit(f"{t}: original train rows are not preserved verbatim")
        for s in ("train", "test"):
            p = os.path.join(out, t, f"{s}.jsonl")
            files[f"{t}/{s}.jsonl"] = {"sha256": sha256_file(p), "git_blob_oid": git_blob_oid(p), "bytes": os.path.getsize(p)}
        if args.variant == "main":
            for s in ("train", "test"):
                if files[f"{t}/{s}.jsonl"]["sha256"] != pub["source_file_sha256"][f"{t}/{s}.jsonl"]:
                    sys.exit(f"{t}/{s}: baseline copy differs from the imported original")

    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(_card(args.variant, args.new_repo_id, rev, counts, new_info, files, license_id, limitation))
    shutil.copyfile(card_src, os.path.join(out, "ORIGINAL_DATASET_CARD.md"))
    open(os.path.join(out, "NOTICE.md"), "w", encoding="utf-8").write(_notice(args.new_repo_id, rev, license_id, card, limitation))
    extra = {}
    for name in ("README.md", "NOTICE.md", "ORIGINAL_DATASET_CARD.md"):
        p = os.path.join(out, name)
        extra[name] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    prov = {"variant": args.variant, "prepared": datetime.datetime.now().isoformat(timespec="seconds"),
            "target_repo": args.new_repo_id, "original_repo": ORIGINAL_REPO, "original_revision": rev, "original_published_manifest": pub,
            "counts": counts, "files": files, "other_files": extra, "license": license_id,
            "original_card": card, "known_limitations": {"train_test_overlap": ov, "text": limitation}, "test_split_identical_to_original": True,
            "original_train_rows_preserved_verbatim": True, "generation_manifest": new_manifest,
            "code": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                     "dirty_files": (_git("status", "--porcelain") or "").splitlines()}}
    json.dump(prov, open(os.path.join(out, "PROVENANCE.json"), "w"), indent=2, default=str)

    cmds = os.path.join(args.release_root, "PUBLISH_COMMANDS.md")
    _write_commands(cmds, args.new_repo_id, rev)
    print(json.dumps(counts, indent=2))
    print(f"Prepared {out} (local only). Commands to run later, after approval: {cmds}")


def _write_commands(path, repo, rev):
    text = f"""# Publishing commands - DO NOT RUN without explicit approval

Nothing below has been executed. The original dataset `{ORIGINAL_REPO}` (revision `{rev}`) is never a target.
Target repo (already created, private): `{repo}`.

## `main` (pinned baseline): use the guided tool, it checks everything before and after the upload

```bash
python src/hf_baseline.py auth-check            # token scope, repo is private, what main contains today
python src/hf_baseline.py prepare               # download the original @ the pinned SHA, build release/main, verify bytes
python src/hf_baseline.py verify-local
python src/hf_baseline.py upload                # asks you to type the repo id; the only step that writes
python src/hf_baseline.py verify-remote
python src/hf_baseline.py cleanup
```
See docs/publishing_workflow.md for details.

## Later branches (not implemented in hf_baseline.py yet, do not run before `main` is verified)

```python
from huggingface_hub import HfApi
api = HfApi()
REPO = "{repo}"
assert REPO != "{ORIGINAL_REPO}"
for branch, folder in [("scale-only-7k", "release/scale-only-7k"), ("mixed-80-20", "release/mixed-80-20")]:
    api.create_branch(REPO, branch=branch, repo_type="dataset")          # branches off main
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=folder, revision=branch,
                      commit_message=f"{{branch}}: see PROVENANCE.json")
```
Make the repo public only when you decide to.
"""
    open(path, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()

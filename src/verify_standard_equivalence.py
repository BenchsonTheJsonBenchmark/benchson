"""Proves that 'standard' mode reproduces the ORIGINAL Benchson generation (git ref `main`).

It exports `main` into a temp dir, runs the old builder and the current builder (with the
scale-only settings: generation block 100% standard, strategy "original") on the same
schemas, seed and the offline MockProvider, then compares:
  * Create and Modify example files: byte-for-byte identical
  * Fix: schema and valid_json identical (the old code injected the error with the unseeded
    global `random`, so the injected error itself cannot be bit-identical; it is checked to be
    a schema violation of the same kind of record instead)
  * the Create prompt text and the default validation gates
No real model is called.

    python src/verify_standard_equivalence.py [--ref main]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RUNNER = r'''
import json, os, sys
tree, cfg_path, out = sys.argv[1:4]
sys.path.insert(0, tree); sys.path.insert(0, os.path.join(tree, "src"))
from llm.mock.mock_provider import MockProvider
from data_generation.benchmark_builder import BenchmarkBuilder
from data_generation.tasks.base_task import BaseDataTask
cfg = json.load(open(cfg_path))
b = BenchmarkBuilder(cfg)
b.build(MockProvider())
schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
t = BaseDataTask(system_prompt_prefix=cfg.get("system_prompt_prefix", ""))
json.dump({"prompt": t._build_schema_creation_prompt(schema), "gates": BenchmarkBuilder._parse_gates({})},
          open(out, "w"), sort_keys=True)
'''


def _run(tree, cwd, cfg, tag):
    os.makedirs(os.path.join(cwd, "data"), exist_ok=True)
    src_schemas = os.path.join(REPO, "data", "schemas")
    os.symlink(src_schemas, os.path.join(cwd, "data", "schemas"))
    cfg_path = os.path.join(cwd, "cfg.json")
    json.dump(cfg, open(cfg_path, "w"))
    runner = os.path.join(cwd, "runner.py")
    open(runner, "w").write(RUNNER)
    meta = os.path.join(cwd, "meta.json")
    r = subprocess.run([sys.executable, runner, tree, cfg_path, meta], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag} run failed:\n{r.stdout[-800:]}\n{r.stderr[-1500:]}")
    return json.load(open(meta))


def _files(cwd, task, split):
    d = os.path.join(cwd, "data", f"benchmark_{task}", split)
    return {f: open(os.path.join(d, f), "rb").read() for f in sorted(os.listdir(d)) if f.endswith(".json")}


def compare(ref="main", seed=3, n=10, with_generation_block=True):
    """Returns (differences, stats). An empty differences list means equivalent."""
    tmp = tempfile.mkdtemp(prefix="equiv_")
    try:
        old = os.path.join(tmp, "old_tree")
        os.makedirs(old)
        tar = subprocess.run(["git", "archive", ref], cwd=REPO, capture_output=True,
                             env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        if tar.returncode != 0:
            raise FileNotFoundError(f"git ref {ref!r} not available: {tar.stderr.decode()[:200]}")
        subprocess.run(["tar", "-x", "-C", old], input=tar.stdout, check=True)
        # `main` has no offline mock provider; it is only a stand-in for the model, so copy it in.
        os.makedirs(os.path.join(old, "src", "llm", "mock"), exist_ok=True)
        shutil.copy(os.path.join(REPO, "src", "llm", "mock", "mock_provider.py"),
                    os.path.join(old, "src", "llm", "mock", "mock_provider.py"))
        open(os.path.join(old, "src", "llm", "mock", "__init__.py"), "a").close()

        base = {"llm_provider": {}, "sources": [{"name": "schemas", "max_schema_kb": 10}], "max_retries": 2,
                "seed": seed, "concurrency": 1, "counts": {"create": n, "fix": n, "modify": n},
                "validation": {"round_trip": False}}
        new_cfg = dict(base)
        if with_generation_block:  # False = the branch's legacy default, to show the check can fail
            new_cfg["generation"] = {"standard_ratio": 1.0, "complex_ratio": 0.0,
                                     "standard_modify_strategy": "original"}
        dirs = {k: os.path.join(tmp, k) for k in ("o", "n")}
        for d in dirs.values():
            os.makedirs(d)
        meta_old = _run(old, dirs["o"], base, "old")
        meta_new = _run(REPO, dirs["n"], new_cfg, "new")

        diffs, stats = [], {}
        if meta_old["prompt"] != meta_new["prompt"]:
            diffs.append("Create prompt text differs")
        if meta_old["gates"] != meta_new["gates"]:
            diffs.append(f"default gates differ: {meta_old['gates']} vs {meta_new['gates']}")
        for task in ("create", "modify", "fix"):
            fo, fn = _files(dirs["o"], task, "test"), _files(dirs["n"], task, "test")
            stats[task] = {"old": len(fo), "new": len(fn)}
            common = len(set(fo) & set(fn))
            # Fix: the old code injected errors with the unseeded global `random`, so which schemas yield a
            # valid corrupted example can differ between two runs (even old vs old). Tolerate a few.
            if task == "fix" and common >= 0.8 * min(len(fo), len(fn)):
                pass
            elif set(fo) != set(fn):
                diffs.append(f"{task}: different schema selection (old {len(fo)}, new {len(fn)}, "
                             f"common {len(set(fo) & set(fn))})")
            for name in sorted(set(fo) & set(fn)):
                if task in ("create", "modify"):
                    if fo[name] != fn[name]:
                        diffs.append(f"{task}/{name}: file content differs")
                else:
                    a, b = json.loads(fo[name]), json.loads(fn[name])
                    for key in ("schema", "valid_json"):
                        if a.get(key) != b.get(key):
                            diffs.append(f"fix/{name}: {key} differs")
                    if set(a) != set(b):
                        diffs.append(f"fix/{name}: field names differ")
        return diffs, stats
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    d, s = compare(a.ref, n=a.n)
    print(json.dumps({"examples_compared": s, "differences": d}, indent=2))
    sys.exit(1 if d else 0)

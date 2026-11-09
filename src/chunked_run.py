"""Validate, plan and merge the chunks of the full Scale Only run (see data_generation/chunking.py).

    python src/chunked_run.py increments --totals create=1334,fix=1333,modify=1333 --chunks 8
    python src/chunked_run.py status   --root data_scale_only/full/chunks
    python src/chunked_run.py plan     --root ROOT --chunks 8 --totals ... [--chain c01_x c02_y ...]
    python src/chunked_run.py validate CHUNK_DIR --anchor-file ANCHOR [--calls-log LOG --build-exit-code N]
    python src/chunked_run.py seal     CHUNK_DIR --anchor-file ANCHOR --calls-log LOG --build-exit-code 0
    python src/chunked_run.py resolve  --root ROOT --index K --anchor-file ANCHOR --reserved-file RESERVED --chunks 8 --totals ... [--pin NAME]
    python src/chunked_run.py merge    --root ROOT --chain c01_x ... c08_z --anchor-file ANCHOR --out-dir NEW_DIR

Rules: nothing is chosen, repaired or merged automatically. The chain to continue from or to merge is always given
explicitly, every link is re-validated (hashes of every file, one pool, one config, global per-schema cap, no duplicates,
no train/test leakage, exact per-task totals), outputs are written to new directories only, and sources are only read.
Exit codes: 0 ok, 2 shortfall, 3 technical, 4 invalid (same as the smoke/pilot checker).
`resolve` (used by the overnight chain, slurm/full_chunk_chain_job.sbatch): exit 0 exactly one fully valid, sealed chunk with that
index (its directory name is printed), 6 none, 7 ambiguous (more than one) -- never a guess.
"""
import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_smoke_result as csr
from data_generation import chunking
from data_generation.published import TASKS, load_reference_index, reference_hash, schema_hash, task_keys


def parse_totals(text):
    try:
        totals = {k: int(v) for k, v in (item.split("=") for item in text.split(","))}
    except ValueError:
        raise SystemExit("--totals must look like create=1334,fix=1333,modify=1333")
    if set(totals) != set(TASKS):
        raise SystemExit("--totals needs create, fix and modify")
    return totals


def _manifest(chunk_dir):
    with open(os.path.join(chunk_dir, "generation_manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def validate_chunk(chunk_dir, anchor_file, reserved_file=None, calls_log=None, build_exit_code=None,
                   require_context_guard=None, seal="all"):
    """Full validation of a chunk AND of every chunk before it. Returns the checker-style result dict.

    seal="all" (default): every chunk of the chain, this one included, must carry a valid success marker.
    seal="ancestors": only the chunks before this one (used by `seal`, before this chunk's own marker exists)."""
    technical, invalid, shortfall, warnings = [], [], [], []
    if build_exit_code == 2:   # build_benchmark.py's own code for "quota not met": a shortfall, with no checkpoint written
        counts = {t: len(f) for t, f in chunking.collect_files(chunk_dir).items()}
        return csr._result(counts, [], [], [f"chunk quota not met (build exit 2): produced {counts}; no checkpoint was written, "
                                           "the chunk output is kept and can never be a parent"], [])
    try:
        chain = chunking.load_chain(chunk_dir, seal)
    except (chunking.ChunkError, OSError) as e:
        return csr._result({t: 0 for t in TASKS}, [f"chunk chain: {e}"], [], [], [])
    last_dir, last = chain[-1]
    base = csr.check(last_dir, per_task=last["quota"], calls_log=calls_log, build_exit_code=build_exit_code,
                     anchor_file=anchor_file)
    technical += base["technical"]; invalid += base["invalid"]; shortfall += base["shortfall"]; warnings += base["warnings"]
    counts = dict(base["counts"])

    manifest = _manifest(last_dir)
    if require_context_guard is not None:   # the run must have been protected by the window guard it was planned with
        guard = manifest.get("context_guard") or {}
        if not guard.get("enabled") or guard.get("max_model_len") != require_context_guard:
            technical.append(f"context guard was not enabled for a {require_context_guard}-token window (manifest: {guard})")
        elif not guard.get("checked"):
            technical.append("context guard counted no request: it did not protect this run")
        elif guard.get("overflow"):
            warnings.append(f"context guard refused {guard['overflow']} request(s) that could not fit the window (expected "
                            "capacity exclusion; see capacity_exclusions in generation_summary.json)")
    cap = ((manifest.get("config") or {}).get("reuse") or {}).get("max_new_references_per_schema")
    if not isinstance(cap, int) or cap < 1:
        technical.append("manifest has no valid reuse.max_new_references_per_schema")
        cap = 0
    if reserved_file is None:
        reserved_file = (manifest.get("published_baseline") or {}).get("file")
    from data_generation.anchor import load_anchor
    try:
        anchor = load_anchor(anchor_file)
    except Exception as e:
        return csr._result(counts, technical + [f"anchor unusable: {e}"], invalid, shortfall, warnings)
    train, test = {e["schema_hash"] for e in anchor["train"]}, {e["schema_hash"] for e in anchor["test"]}
    pub_refs, pub_keys = ({}, {t: {"exact": set()} for t in TASKS})
    if reserved_file and os.path.isfile(reserved_file):
        pub_refs, pub_keys = load_reference_index(reserved_file)
    else:
        technical.append(f"reserved_schemas.json not found ({reserved_file}); duplicate control against the published baseline not possible")

    cum = {t: 0 for t in TASKS}
    refs_per_schema, per_task_per_schema, seen_keys, seen_names = {}, {}, {t: {} for t in TASKS}, set()
    ref_owner = {}   # (schema id, reference hash) -> attempt that minted it
    for cdir, state in chain:
        for t in TASKS:
            cum[t] += state["counts"][t]
        for task in TASKS:
            folder = os.path.join(cdir, f"benchmark_{task}", "train")
            for name in state["files"][task]:
                if (task, name) in seen_names:
                    invalid.append(f"{task}/{name}: the same example file name appears in two chunks")
                seen_names.add((task, name))
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    rec = json.load(f)
                sid, _, att = name[:-5].rpartition("__a")
                refs_per_schema.setdefault(sid, set()).add(att)
                per_task_per_schema[(task, sid)] = per_task_per_schema.get((task, sid), 0) + 1
                h = schema_hash(rec["schema"])
                if h not in train or h in test:
                    invalid.append(f"{cdir}/{task}/{name}: schema is not a published train schema or overlaps the test set")
                if cdir != last_dir:   # the last chunk was already checked example by example by csr.check
                    problems, _ = csr.example_problems(task, rec, 0.9)
                    invalid.extend(f"{cdir}/{task}/{name}: {p}" for p in problems)
                rh = reference_hash(task, rec)
                if rh in pub_refs.get(h, ()):
                    invalid.append(f"{task}/{name}: its reference equals a published reference of the same schema")
                if ref_owner.setdefault((sid, rh), att) != att:
                    invalid.append(f"{task}/{name}: same reference as attempt a{ref_owner[(sid, rh)]} of the same schema")
                exact, _c = task_keys(task, rec)
                if exact in pub_keys[task]["exact"]:
                    invalid.append(f"{task}/{name}: identical to a published example")
                if exact in seen_keys[task]:
                    invalid.append(f"{task}/{name}: duplicate of {seen_keys[task][exact]}")
                seen_keys[task][exact] = name
    for sid, attempts in sorted(refs_per_schema.items()):
        if cap and len(attempts) > cap:
            invalid.append(f"schema {sid}: {len(attempts)} new references across chunks, global cap is {cap}")
    for (task, sid), n in sorted(per_task_per_schema.items()):
        if cap and n > cap:
            invalid.append(f"{task}/{sid}: {n} new examples across chunks, global cap is {cap}")
    saved = last["ledger"]
    for sid, row in saved.items():
        if row["accepted_references"] != len(refs_per_schema.get(sid, ())):
            invalid.append(f"schema {sid}: ledger says {row['accepted_references']} references, files show {len(refs_per_schema.get(sid, ()))}")
    k, n, totals = last["chunk_index"], last["n_chunks"], last["totals"]
    for t in TASKS:
        if cum[t] != totals[t] * k // n:
            invalid.append(f"{t}: cumulative {cum[t]} after chunk {k}, expected {totals[t] * k // n}")
    res = csr._result(counts, technical, invalid, shortfall, warnings)
    res["chain"] = [os.path.basename(d) for d, _ in chain]
    res["cumulative"] = cum
    res["chunk_index"], res["n_chunks"] = k, n
    return res


def calls_log_summary(calls_log):
    """What the success marker records about the calls log (it lives outside the chunk directory)."""
    n_calls = n_failed = 0
    with open(calls_log, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                n_calls += 1
                n_failed += 1 if row.get("error") else 0
    return {"checked": True, "file": os.path.basename(calls_log), "sha256": chunking.file_sha256(calls_log),
            "n_calls": n_calls, "n_failed": n_failed}


def seal_chunk(chunk_dir, anchor_file, reserved_file=None, calls_log=None, build_exit_code=None, require_context_guard=None):
    """Runs every final validation (chain, checker, calls log) and ONLY if all pass writes chunk_success.json.
    Fails closed: no calls log, a build exit code other than 0 or an existing marker means no marker."""
    chunk_dir = os.path.abspath(chunk_dir)
    refuse = []
    if os.path.lexists(os.path.join(chunk_dir, chunking.SEAL_FILE)):
        refuse.append("a success marker already exists; it is never replaced")
    if build_exit_code != 0:   # shortfall (2) or technical failure: reported exactly as `validate` does, never sealed
        res = validate_chunk(chunk_dir, anchor_file, reserved_file, calls_log, build_exit_code, require_context_guard,
                             seal="ancestors")
        if res["code"] == 0:
            res = csr._result(res["counts"], [f"not sealed: build exit code is {build_exit_code!r}, not 0"], [], [], [])
        return res
    if not calls_log or not os.path.isfile(calls_log):
        refuse.append("the calls log is required to seal a chunk and was not found")
    if refuse:
        return csr._result({t: 0 for t in TASKS}, ["not sealed: " + "; ".join(refuse)], [], [], [])
    res = validate_chunk(chunk_dir, anchor_file, reserved_file, calls_log, build_exit_code, require_context_guard,
                         seal="ancestors")
    if res["code"] != 0:
        return res
    try:
        state = chunking.load_state(chunk_dir)
        checker = {"code": res["code"], "verdict": res["verdict"], "counts": res["counts"],
                   "warnings": len(res["warnings"])}
        chunking.write_seal(chunk_dir, state, checker, calls_log_summary(calls_log), build_exit_code, require_context_guard)
        chunking.load_seal(chunk_dir, state)   # read back what was written
    except (chunking.ChunkError, OSError, ValueError) as e:
        return csr._result(res["counts"], [f"could not write a valid success marker: {e}"], [], [], res["warnings"])
    res["sealed"] = True
    return res


RESOLVE_NONE, RESOLVE_AMBIGUOUS = 6, 7
CHUNK_DIR_RE = re.compile(r"^c([0-9]{2})_([0-9]+)-r([0-9]+)$")


def find_sealed(root, index, anchor_file, reserved_file, n_chunks, totals):
    """Every directory of chunk `index` under `root` is fully validated (hash chain, success marker, ancestors, global cap,
    duplicates, leakage, exact cumulative totals, this exact plan). Returns (valid names, [(name, reason)] ignored).
    Unsealed, partial, preempted, failed or tampered attempts are ignored (and left untouched); nothing is written."""
    root = os.path.abspath(root)
    valid, ignored = [], []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = CHUNK_DIR_RE.match(name)
        if not m or int(m.group(1)) != index:
            continue
        path = os.path.join(root, name)
        if os.path.islink(path) or not os.path.isdir(path):
            ignored.append((name, "not a real directory"))
            continue
        try:
            res = validate_chunk(path, anchor_file, reserved_file)
            if res["code"] != 0:
                ignored.append((name, "; ".join((res["technical"] + res["invalid"] + res["shortfall"])[:2])[:240]))
                continue
            state = chunking.load_state(path)
        except Exception as e:   # anything unexpected means "not usable", never "usable"
            ignored.append((name, f"{type(e).__name__}: {str(e)[:200]}"))
            continue
        if (state["chunk_index"], state["n_chunks"], state["totals"]) != (index, n_chunks, totals) or \
                res.get("chain", [None])[-1] != name:
            ignored.append((name, "belongs to a different plan or chain"))
            continue
        valid.append(name)
    return valid, ignored


def resolve_chunk(root, index, anchor_file, reserved_file, n_chunks, totals, pin=None):
    """(code, name, valid, ignored): exactly one valid sealed chunk -> 0; none -> 6; several -> 7 unless `pin` names one of them."""
    valid, ignored = find_sealed(root, index, anchor_file, reserved_file, n_chunks, totals)
    if pin is not None:
        if pin not in valid:
            return RESOLVE_NONE, None, valid, ignored
        return 0, pin, valid, ignored
    if not valid:
        return RESOLVE_NONE, None, valid, ignored
    if len(valid) > 1:
        return RESOLVE_AMBIGUOUS, None, valid, ignored
    return 0, valid[0], valid, ignored


def merge_chain(root, names, anchor_file, out_dir, reserved_file=None):
    """Copies the explicitly chosen, fully validated chain (1..N) into a NEW directory. Sources are only read."""
    dirs = [os.path.join(os.path.abspath(root), n) for n in names]
    if os.path.lexists(out_dir):
        raise SystemExit(f"merge output already exists, refusing to touch it: {out_dir}")
    res = validate_chunk(dirs[-1], anchor_file, reserved_file)
    if res["code"] != 0:
        raise SystemExit(f"final chunk does not validate ({res['verdict']}): " + "; ".join((res['technical'] + res['invalid'] + res['shortfall'])[:5]))
    chain = chunking.load_chain(dirs[-1])
    if [os.path.basename(d) for d, _ in chain] != list(names):
        raise SystemExit("the given chain is not the parent chain of its last chunk (wrong, missing or extra chunks)")
    last = chain[-1][1]
    if last["chunk_index"] != last["n_chunks"]:
        raise SystemExit(f"chain ends at chunk {last['chunk_index']} of {last['n_chunks']}: not complete")
    if res["cumulative"] != last["totals"]:
        raise SystemExit(f"totals {res['cumulative']} differ from the plan {last['totals']}")
    os.makedirs(out_dir)   # fails if it exists: never overwrites
    manifest = {"format": "benchson-merged-chunks/1", "chain": [], "totals": last["totals"], "files": {t: {} for t in TASKS}}
    for cdir, state in chain:
        manifest["chain"].append({"dir": os.path.basename(cdir), "state_sha256": state["self_sha256"],
                                  "success_marker_sha256": chunking.load_seal(cdir, state)["self_sha256"]})
        for task in TASKS:
            dst = os.path.join(out_dir, f"benchmark_{task}", "train")
            os.makedirs(dst, exist_ok=True)
            for name, digest in sorted(state["files"][task].items()):
                target = os.path.join(dst, name)
                if os.path.exists(target):
                    raise SystemExit(f"file name collision while merging: {task}/{name}")
                shutil.copyfile(os.path.join(cdir, f"benchmark_{task}", "train", name), target)
                if chunking.file_sha256(target) != digest:
                    raise SystemExit(f"copy of {task}/{name} differs from the recorded hash")
                manifest["files"][task][name] = digest
                meta = os.path.join(cdir, f"benchmark_{task}", "_metadata", "train", name)
                if os.path.isfile(meta):
                    mdst = os.path.join(out_dir, f"benchmark_{task}", "_metadata", "train")
                    os.makedirs(mdst, exist_ok=True)
                    shutil.copyfile(meta, os.path.join(mdst, name))
    for task in TASKS:   # no test examples are produced; keep the layout the downstream merge tool expects
        os.makedirs(os.path.join(out_dir, f"benchmark_{task}", "test"), exist_ok=True)
    with open(os.path.join(out_dir, "merged_chunks_manifest.json"), "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


def status(root):
    rows = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        row = {"dir": name}
        try:
            st = chunking.load_state(d)
            chunking.verify_files(d, st)
            row.update(chunk_index=st["chunk_index"], counts=st["counts"], self_sha256=st["self_sha256"])
            try:
                chunking.load_seal(d, st)
                row["state"] = "sealed"
            except chunking.ChunkError as e:
                row.update(state="checkpointed_unsealed (not a valid parent, never merged)", problem=str(e)[:200])
        except chunking.ChunkError as e:
            row.update(state="incomplete_or_invalid (kept, never merged)", problem=str(e)[:200])
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("increments"); p.add_argument("--totals", required=True); p.add_argument("--chunks", type=int, required=True)
    p = sub.add_parser("status"); p.add_argument("--root", required=True)
    p = sub.add_parser("plan"); p.add_argument("--root", required=True); p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--totals", required=True); p.add_argument("--chain", nargs="*", default=[]); p.add_argument("--anchor-file")
    p = sub.add_parser("validate"); p.add_argument("chunk_dir"); p.add_argument("--anchor-file", required=True)
    p.add_argument("--reserved-file"); p.add_argument("--calls-log"); p.add_argument("--build-exit-code", type=int)
    p.add_argument("--require-context-guard", type=int, default=None, metavar="MAX_MODEL_LEN")
    p = sub.add_parser("seal"); p.add_argument("chunk_dir"); p.add_argument("--anchor-file", required=True)
    p.add_argument("--reserved-file"); p.add_argument("--calls-log", required=True)
    p.add_argument("--build-exit-code", type=int, required=True)
    p.add_argument("--require-context-guard", type=int, default=None, metavar="MAX_MODEL_LEN")
    p = sub.add_parser("resolve"); p.add_argument("--root", required=True); p.add_argument("--index", type=int, required=True)
    p.add_argument("--anchor-file", required=True); p.add_argument("--reserved-file", required=True)
    p.add_argument("--chunks", type=int, required=True); p.add_argument("--totals", required=True)
    p.add_argument("--pin", default=None, help="explicit operator choice among several valid, sealed chunks")
    p = sub.add_parser("merge"); p.add_argument("--root", required=True); p.add_argument("--chain", nargs="+", required=True)
    p.add_argument("--anchor-file", required=True); p.add_argument("--reserved-file"); p.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "increments":
        totals = parse_totals(a.totals)
        rows = [chunking.increments(totals, a.chunks, k) for k in range(1, a.chunks + 1)]
        print(json.dumps({"chunks": rows, "sum": {t: sum(r[t] for r in rows) for t in TASKS}}, indent=1))
        return 0
    if a.cmd == "status":
        print(json.dumps(status(a.root), indent=1))
        return 0
    if a.cmd == "plan":
        totals = parse_totals(a.totals)
        out = {"status": status(a.root)}
        if a.chain:
            if not a.anchor_file:
                raise SystemExit("--anchor-file is needed to validate the chain")
            res = validate_chunk(os.path.join(a.root, a.chain[-1]), a.anchor_file)
            out["chain_validation"] = {"verdict": res["verdict"], "chain": res.get("chain"), "problems": (res["technical"] + res["invalid"] + res["shortfall"])[:10]}
            if res["code"] != 0 or res["chain"] != list(a.chain):
                print(json.dumps(out, indent=1)); return res["code"] or 4
            nxt = res["chunk_index"] + 1
            parent = a.chain[-1]
        else:
            nxt, parent = 1, None
        if nxt > a.chunks:
            out["next"] = "complete: run `merge`"
        else:
            out["next"] = {"CHUNK_INDEX": nxt, "N_CHUNKS": a.chunks, "PARENT_CHUNK": parent, "quota": chunking.increments(totals, a.chunks, nxt)}
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "validate":
        res = validate_chunk(a.chunk_dir, a.anchor_file, a.reserved_file, a.calls_log, a.build_exit_code, a.require_context_guard)
        print(f"CHUNK RESULT: {res['verdict']}  chain={res.get('chain')} cumulative={res.get('cumulative')} this chunk={res['counts']}")
        for label in ("technical", "invalid", "shortfall", "warnings"):
            for item in res[label][:20]:
                print(f"  [{label}] {item}")
        return res["code"]
    if a.cmd == "seal":
        res = seal_chunk(a.chunk_dir, a.anchor_file, a.reserved_file, a.calls_log, a.build_exit_code, a.require_context_guard)
        print(f"CHUNK RESULT: {res['verdict']}  chain={res.get('chain')} cumulative={res.get('cumulative')} "
              f"this chunk={res['counts']} sealed={bool(res.get('sealed'))}")
        for label in ("technical", "invalid", "shortfall", "warnings"):
            for item in res[label][:20]:
                print(f"  [{label}] {item}")
        return res["code"]
    if a.cmd == "resolve":
        code, name, valid, ignored = resolve_chunk(a.root, a.index, a.anchor_file, a.reserved_file, a.chunks,
                                                   parse_totals(a.totals), a.pin)
        print(json.dumps({"index": a.index, "valid_sealed": valid, "ignored": [{"dir": n, "reason": r} for n, r in ignored],
                          "result": name or ("none" if code == RESOLVE_NONE else "ambiguous")}, indent=1), file=sys.stderr)
        if code == 0:
            print(name)
        return code
    if a.cmd == "merge":
        manifest = merge_chain(a.root, a.chain, a.anchor_file, a.out_dir, a.reserved_file)
        print(f"merged {len(a.chain)} chunks into {a.out_dir}: " + json.dumps({t: len(v) for t, v in manifest['files'].items()}))
        return 0


if __name__ == "__main__":
    sys.exit(main())

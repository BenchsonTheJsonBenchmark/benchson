"""Chunked generation with checkpoints for the full Scale Only run.

A chunk is a bounded generation job that continues the SAME run from a verified parent state, so that every limit of
Controlled Schema Reuse (3 new references per schema, failure counts, suspensions, retirement, the attempt budget, the
duplicate index) is global and is never reset between chunks.

What is carried (`chunk_state.json`, written last, only when the chunk met its quota exactly):
  * the per-schema ledger (attempts, accepted references, failures, suspensions, status),
  * the SHA-256 of every example file the chunk produced, and a link to the parent state (hash chain),
  * fingerprints of the anchor pool and the config, which must stay identical for the whole chain.
The duplicate index is rebuilt from the example files of the whole chain (every file hash is re-verified first).

Nothing here changes a prompt, a gate, a sampling parameter or an acceptance rule, and nothing is ever overwritten or
merged implicitly: a chunk that was interrupted has no state file and can only be inspected.
"""
import datetime
import hashlib
import json
import os

STATE_FILE = "chunk_state.json"
STATE_FORMAT = "benchson-chunk-state/1"
SEAL_FILE = "chunk_success.json"
SEAL_FORMAT = "benchson-chunk-success/1"
TASKS = ("create", "fix", "modify")
LEDGER_FIELDS = ("attempts", "accepted_references", "failed_attempts", "consecutive_failures", "suspensions",
                 "suspended_until_pass", "status", "reason", "examples", "published_references")


class ChunkError(ValueError):
    pass


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def increments(totals, n_chunks, index):
    """Per-task quota of chunk `index` (1-based); the quotas of all chunks sum to `totals` exactly."""
    if not (isinstance(n_chunks, int) and n_chunks >= 1 and isinstance(index, int) and 1 <= index <= n_chunks):
        raise ChunkError(f"chunk index {index!r} is outside 1..{n_chunks!r}")
    return {t: totals[t] * index // n_chunks - totals[t] * (index - 1) // n_chunks for t in totals}


def pool_fingerprint(by_id):
    """by_id: {schema_id: (..., schema_hash)} -> one hash of the exact pool (ids and content hashes)."""
    return canonical_sha256(sorted((sid, v[4]) for sid, v in by_id.items()))


def collect_files(out_dir):
    """{task: {filename: sha256}} of the new training examples in a chunk directory."""
    files = {}
    for task in TASKS:
        folder = os.path.join(out_dir, f"benchmark_{task}", "train")
        files[task] = {n: file_sha256(os.path.join(folder, n)) for n in sorted(os.listdir(folder)) if n.endswith(".json")} \
            if os.path.isdir(folder) else {}
    return files


def write_state(out_dir, state):
    """Writes chunk_state.json once (never overwrites) with its own hash inside."""
    state = {**state, "format": STATE_FORMAT}
    state["self_sha256"] = canonical_sha256(state)
    path = os.path.join(out_dir, STATE_FILE)
    with open(path, "x", encoding="utf-8") as f:   # "x": an existing state is never replaced
        json.dump(state, f, indent=1, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    return state


def load_state(chunk_dir):
    path = os.path.join(chunk_dir, STATE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        raise ChunkError(f"{chunk_dir}: no valid {STATE_FILE} (the chunk is incomplete or was interrupted): {e}")
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT:
        raise ChunkError(f"{path}: unknown state format")
    body = {k: v for k, v in state.items() if k != "self_sha256"}
    if state.get("self_sha256") != canonical_sha256(body):
        raise ChunkError(f"{path}: the state file was modified or is corrupt (hash mismatch)")
    return state


def verify_files(chunk_dir, state):
    """Every example of the chunk must exist, be unchanged, and nothing else may be in the train folders."""
    on_disk = collect_files(chunk_dir)
    for task in TASKS:
        want, have = state["files"][task], on_disk[task]
        if want != have:
            missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
            changed = sorted(n for n in set(want) & set(have) if want[n] != have[n])
            raise ChunkError(f"{chunk_dir}/{task}: files differ from the recorded state "
                             f"(missing {missing[:3]}, unexpected {extra[:3]}, changed {changed[:3]})")


def write_seal(chunk_dir, state, checker, calls_log, build_exit_code, context_guard=None, recovery=None):
    """The final success marker: written once (never overwrites), only after every final validation passed.
    It binds the run identity, the hash of the validated state (and so of every example file), the exact counts,
    the final checker result and the calls-log validation result."""
    body = {"format": SEAL_FORMAT, "run_id": os.path.basename(os.path.abspath(chunk_dir)),
            "chunk_index": state["chunk_index"], "n_chunks": state["n_chunks"],
            "state_sha256": state["self_sha256"], "files_digest": canonical_sha256(state["files"]),
            "counts": dict(state["counts"]), "checker": checker, "calls_log": calls_log,
            "build_exit_code": build_exit_code, "context_guard_required": context_guard,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    if recovery is not None:   # a late, explicit recovery says so in the marker; the marker is verified exactly like any other
        body["recovery"] = recovery
    body["self_sha256"] = canonical_sha256(body)
    with open(os.path.join(chunk_dir, SEAL_FILE), "x", encoding="utf-8") as f:   # "x": an existing marker is never replaced
        json.dump(body, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    return body


def load_seal(chunk_dir, state):
    """Fails closed: the marker must exist, be intact and agree with this exact (already verified) state."""
    path = os.path.join(chunk_dir, SEAL_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            seal = json.load(f)
    except (OSError, ValueError) as e:
        raise ChunkError(f"{chunk_dir}: no valid {SEAL_FILE}: the chunk was never fully validated ({e})")
    if not isinstance(seal, dict) or seal.get("format") != SEAL_FORMAT:
        raise ChunkError(f"{path}: unknown success-marker format")
    if seal.get("self_sha256") != canonical_sha256({k: v for k, v in seal.items() if k != "self_sha256"}):
        raise ChunkError(f"{path}: the success marker was modified or is corrupt (hash mismatch)")
    checker, calls = seal.get("checker"), seal.get("calls_log")
    problems = []
    if seal.get("run_id") != os.path.basename(os.path.abspath(chunk_dir)):
        problems.append("run identity differs from the directory name")
    if seal.get("chunk_index") != state["chunk_index"] or seal.get("n_chunks") != state["n_chunks"]:
        problems.append("chunk index differs from the state")
    if seal.get("state_sha256") != state["self_sha256"]:
        problems.append("the marker belongs to a different state file")
    if seal.get("files_digest") != canonical_sha256(state["files"]):
        problems.append("the example-file hashes differ from the validated ones")
    if seal.get("counts") != state["counts"]:
        problems.append("counts differ from the state")
    if not (isinstance(checker, dict) and checker.get("code") == 0 and checker.get("verdict") == "SUCCESS"
            and checker.get("counts") == state["counts"]):
        problems.append("the final checker result is not SUCCESS with the state's counts")
    if not (isinstance(calls, dict) and calls.get("checked") is True and isinstance(calls.get("n_calls"), int)
            and calls["n_calls"] >= 1 and calls.get("n_failed") == 0):
        problems.append("the calls log was not validated as complete and failure-free")
    if seal.get("build_exit_code") != 0:
        problems.append("the build did not exit with 0")
    if problems:
        raise ChunkError(f"{path}: inconsistent success marker: " + "; ".join(problems))
    return seal


def load_chain(chunk_dir, seal="all"):
    """[(dir, state)] from chunk 1 to `chunk_dir`, every link and every file hash verified.

    seal: "all" (default, fail closed) every chunk needs a valid success marker; "ancestors" the starting chunk may still
    be unsealed (used only by the job that is about to seal it); "none" inspection only, never use for parents/merges."""
    if seal not in ("all", "ancestors", "none"):
        raise ChunkError(f"unknown seal mode {seal!r}")
    chunk_dir = os.path.abspath(chunk_dir)
    chain, seen = [], set()
    cur = chunk_dir
    while True:
        if cur in seen:
            raise ChunkError("loop in the chunk chain")
        seen.add(cur)
        state = load_state(cur)
        verify_files(cur, state)
        if seal == "all" or (seal == "ancestors" and cur != chunk_dir):
            load_seal(cur, state)
        chain.append((cur, state))
        parent = state.get("parent")
        if parent is None:
            if state["chunk_index"] != 1:
                raise ChunkError(f"{cur}: chunk {state['chunk_index']} has no parent")
            break
        pdir = os.path.join(os.path.dirname(cur), parent["dir"])
        pstate = load_state(pdir)
        if pstate["self_sha256"] != parent["state_sha256"] or pstate["chunk_index"] != state["chunk_index"] - 1:
            raise ChunkError(f"{cur}: parent {parent['dir']} does not match the recorded link")
        cur = pdir
    chain.reverse()
    first = chain[0][1]
    for d, st in chain:
        for key in ("anchor_sha256", "pool_fingerprint", "config_sha256", "totals", "n_chunks", "seed"):
            if st[key] != first[key]:
                raise ChunkError(f"{d}: {key} differs from chunk 1 (the chain must use one pool, one config and one plan)")
    return chain


def restore_ledger(ledger, chain):
    """Loads the last state's ledger into a fresh SchemaLedger (every schema must be there, with the same content)."""
    last_dir, last = chain[-1]
    saved = last["ledger"]
    if set(saved) != set(ledger.entries):
        raise ChunkError("the saved ledger and the current pool contain different schemas")
    for sid, row in saved.items():
        entry = ledger.entries[sid]
        if entry["schema_hash"] != row["schema_hash"]:
            raise ChunkError(f"schema {sid} changed content since the parent chunk")
        for key in LEDGER_FIELDS:
            entry[key] = json.loads(json.dumps(row[key]))
        # pass numbers restart in every chunk: a suspension keeps only the passes it still has to wait
        entry["suspended_until_pass"] = max(0, row["suspended_until_pass"] - last["passes_end"])
    ledger.events.extend(json.loads(json.dumps(last.get("events", []))))
    return ledger

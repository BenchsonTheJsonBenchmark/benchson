"""Prepares (and verifies) data/jsb_schemas, the JSONSchemaBench part of the original schema pool.

data/jsb_schemas is git-ignored, so a fresh clone or a cluster checkout does not have it, and the
builder stops with "Dataset folder not found: .../data/jsb_schemas/test". The authoritative copy is
the one the published Benchson v2 benchmark was built from: git commit e7f8b22 ("v2 benchmark and
training data"), 1308 files, which reproduces the original pool exactly (1669 schemas = 250 test +
1419 train). This script recreates it from that commit - nothing is downloaded from anywhere else
and no other schema set is ever substituted.

    python src/prepare_jsb_schemas.py --check-only   # verify what is on disk (no git needed)
    python src/prepare_jsb_schemas.py                # create it from the pinned git commit, then verify
    python src/prepare_jsb_schemas.py --fetch        # same, after `git fetch origin` (commit not in clone yet)

Verification is exact: the git *tree id* of the directory (computed here from the file contents, so no
git objects are needed to check) must equal the pinned value. An existing directory that does not match
is reported and never touched.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED_COMMIT = "e7f8b22e23097ebbedfd972faeeeb0c38c8d58ff"
PINNED_TREE_OID = "32e06bd10699e6e9b82f9b2aa0b8a1857e603523"   # git rev-parse <commit>:data/jsb_schemas
PINNED_FILE_COUNT = 1308
GIT_PATH = "data/jsb_schemas"
DEFAULT_TARGET = os.path.join(REPO, "data", "jsb_schemas")


class PrepareError(Exception):
    pass


def _hash_object(kind, payload):
    return hashlib.sha1(kind.encode() + b" %d\x00" % len(payload) + payload).hexdigest()


def tree_oid(path):
    """The git tree id of a directory: identical to `git rev-parse <commit>:<dir>` for the same content."""
    entries = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.islink(full):
            raise PrepareError(f"symbolic link not allowed in {path}: {name}")
        if os.path.isdir(full):
            entries.append((name + "/", b"40000 " + name.encode() + b"\x00" + bytes.fromhex(tree_oid(full))))
        elif os.path.isfile(full):
            with open(full, "rb") as f:
                data = f.read()
            entries.append((name, b"100644 " + name.encode() + b"\x00" + bytes.fromhex(_hash_object("blob", data))))
    entries.sort(key=lambda e: e[0].encode())   # git orders directories as if their name ended with "/"
    return _hash_object("tree", b"".join(e[1] for e in entries))


def count_files(path):
    return sum(len(files) for _, _, files in os.walk(path))


def verify(target, expected_tree=PINNED_TREE_OID, expected_files=PINNED_FILE_COUNT):
    """Returns (ok, message). Never modifies anything."""
    if not os.path.isdir(target):
        return False, f"not found: {target}"
    n = count_files(target)
    if n != expected_files:
        return False, f"{target} has {n} files, expected {expected_files}"
    try:
        got = tree_oid(target)
    except PrepareError as e:
        return False, str(e)
    if got != expected_tree:
        return False, (f"{target} content differs from the pinned corpus (tree id {got}, expected "
                       f"{expected_tree}); compare with `git ls-tree -r {PINNED_COMMIT[:7]} {GIT_PATH}`")
    return True, f"{target}: {n} files, tree id {got} matches commit {PINNED_COMMIT[:7]}"


def _git(args, repo, **kw):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, **kw)


def extract(commit, staging, repo=REPO, git_path=GIT_PATH):
    """Writes git_path of `commit` into `staging` (whose layout is <staging>/<subdir>/<file>)."""
    r = _git(["cat-file", "-e", commit + "^{commit}"], repo)
    if r.returncode != 0:
        raise PrepareError(f"commit {commit} is not in this clone. Run: git fetch origin   "
                           "(it is on origin/main), or re-run with --fetch")
    r = _git(["ls-tree", "-r", "-z", commit, git_path], repo)
    if r.returncode != 0:
        raise PrepareError("git ls-tree failed: " + r.stderr.decode(errors="replace")[:300])
    prefix = git_path.rstrip("/") + "/"
    entries = []
    for rec in r.stdout.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, kind, oid = meta.decode().split()
        path = path.decode()
        if kind != "blob" or mode not in ("100644", "100755") or not path.startswith(prefix):
            raise PrepareError(f"unexpected tree entry: {meta.decode()} {path}")
        rel = path[len(prefix):]
        if rel.startswith("/") or ".." in rel.split("/"):
            raise PrepareError(f"unsafe path in commit: {path}")
        entries.append((oid, rel))
    batch = _git(["cat-file", "--batch"], repo, input="".join(o + "\n" for o, _ in entries).encode())
    if batch.returncode != 0:
        raise PrepareError("git cat-file failed: " + batch.stderr.decode(errors="replace")[:300])
    buf, pos = batch.stdout, 0
    for oid, rel in entries:
        eol = buf.index(b"\n", pos)
        head = buf[pos:eol].split()
        if len(head) != 3 or head[0].decode() != oid or head[1] != b"blob":
            raise PrepareError(f"unexpected cat-file answer for {oid}: {buf[pos:eol][:80]!r}")
        size = int(head[2])
        data = buf[eol + 1: eol + 1 + size]
        pos = eol + 1 + size + 1
        dest = os.path.join(staging, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
    return len(entries)


def prepare(target=DEFAULT_TARGET, commit=PINNED_COMMIT, expected_tree=PINNED_TREE_OID,
            expected_files=PINNED_FILE_COUNT, fetch=False, repo=REPO):
    """Idempotent. Returns (status, message); status is 'present' or 'created'. Raises PrepareError."""
    if os.path.lexists(target):
        ok, msg = verify(target, expected_tree, expected_files)
        if ok:
            return "present", msg
        raise PrepareError(msg + ". Refusing to overwrite it; move it away and re-run if it is not needed.")
    if fetch:
        r = _git(["fetch", "origin"], repo)
        if r.returncode != 0:
            raise PrepareError("git fetch origin failed: " + r.stderr.decode(errors="replace")[:300])
    staging = target.rstrip("/") + f".partial-{os.getpid()}"
    try:
        os.makedirs(staging)
        extract(commit, staging, repo)
        ok, msg = verify(staging, expected_tree, expected_files)
        if not ok:
            raise PrepareError("extracted copy failed verification: " + msg)
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    ok, msg = verify(target, expected_tree, expected_files)
    if not ok:
        raise PrepareError(msg)
    return "created", msg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--check-only", action="store_true", help="only verify the existing directory")
    ap.add_argument("--fetch", action="store_true", help="run `git fetch origin` before extracting")
    args = ap.parse_args(argv)
    try:
        if args.check_only:
            ok, msg = verify(args.target)
            print(("OK: " if ok else "FAILED: ") + msg)
            if not ok:
                print("Prepare it with: python src/prepare_jsb_schemas.py")
            return 0 if ok else 1
        status, msg = prepare(args.target, fetch=args.fetch)
        print(f"{status.upper()}: {msg}")
        return 0
    except PrepareError as e:
        print("FAILED: " + str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())

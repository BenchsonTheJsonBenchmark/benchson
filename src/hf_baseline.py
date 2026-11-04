"""Guided copy of the pinned original Benchson dataset into OUR private Hugging Face repo (branch `main`).

    aviv1ron1/Benchson @ 7962b0947f7d96eb3a0e6b9b928d939c7f3061a6   (public, read only, never modified)
        -> temporary folder on this computer
        -> benchson-research/benchson-data-experiments, branch main         (private, our repo)

Steps (each one is a separate command, run them in this order):

    python src/hf_baseline.py auth-check      token scope, repo is private, what `main` holds today (no writes)
    python src/hf_baseline.py prepare         download the original @ the pinned SHA (no token is sent), build
                                              release/main, verify the six data files byte for byte (no writes remote)
    python src/hf_baseline.py verify-local    re-run the local checks (also run automatically by `upload`)
    python src/hf_baseline.py diagnose        read only: explain a credential-check result / train-test schema overlap
                                              (prints counts, kinds, line numbers and ids; never matched text)
    python src/hf_baseline.py classify-credentials   read only: group credential-like strings by value and field
                                              (sanitized: file, line, field path, classification; never values)
    python src/hf_baseline.py inspect-credential --group <8hex>   read only: structure of ONE flagged private-key value
                                              (format, lengths, validity; never the value or its decoded bytes)
    python src/hf_baseline.py record-credential-review --body-sha256 <64hex> --expect-occurrences N
                                              write a hash-pinned, place-pinned review for one exact body (only for the
                                              "public part of a truncated RSA container" verdict)
    python src/hf_baseline.py upload          the ONLY command that writes to Hugging Face; asks you to type the repo id
    python src/hf_baseline.py verify-remote   re-download from the new repo and compare with the original
    python src/hf_baseline.py cleanup         delete the temporary folder (after a passing verify-remote)

The token is read from the HF_TOKEN environment variable or from `hf auth login`. It is never accepted as a
command-line argument, never printed and never written to a file. Branches `scale-only-7k` and `mixed-80-20`
are NOT handled here and are never created by this script.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from data_generation.published import (KNOWN_REVISIONS, TASKS, git_blob_oid, overlap_report,  # noqa: E402
                                       parse_card_license, sha256_file)

ORIGINAL_REPO = "aviv1ron1/Benchson"
PINNED = "7962b0947f7d96eb3a0e6b9b928d939c7f3061a6"
DEFAULT_REPO = "benchson-research/benchson-data-experiments"
MARKER = ".benchson_hf_transfer"
# Human-reviewed exceptions: exact value (full SHA-256 of the encoded body) + exact context. See docs/publishing_workflow.md.
REVIEWS_PATH = os.path.join(ROOT, "docs", "credential_reviews.json")
REVIEWED_VERDICT = "truncated_rsa_key_prefix_public_part_only"
EXPECTED_ROWS = {"train": 1000, "test": 150}
EXTRA_FILES = ("README.md", "NOTICE.md", "ORIGINAL_DATASET_CARD.md", "PROVENANCE.json")
DATA_FILES = [f"{t}/{s}.jsonl" for t in TASKS for s in ("train", "test")]
# Credential detector. Each pattern needs a left boundary (a match that starts inside a longer word, such as the
# `sk-` in an ordinary hyphenated name, is not a key) and the generic ones also need a digit in the secret part
# (random keys have digits; ordinary words do not). Matched text is NEVER printed, only kind, file and line number.
_B = r"(?<![A-Za-z0-9_\-])"
CREDENTIAL_PATTERNS = [  # (kind, regex, needs_digit)
    ("huggingface_token", re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}"), False),
    ("openai_style_key", re.compile(_B + r"sk-[A-Za-z0-9_\-]{20,}"), True),
    ("aws_access_key_id", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"), False),
    ("github_token", re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})"), False),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9_])xox[baprs]-[A-Za-z0-9\-]{10,}"), False),
    ("google_api_key", re.compile(r"(?<![A-Za-z0-9_])AIza[0-9A-Za-z_\-]{35}"), False),
    ("private_key_header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), False),
    ("bearer_token", re.compile(r"(?<![A-Za-z0-9_])Bearer\s+[A-Za-z0-9._\-]{20,}"), True),
]
# The first (looser) version of the check, kept only so `diagnose` can say why it over-fired.
_LEGACY = {"huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"), "openai_style_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
           "aws_access_key_id": re.compile(r"AKIA[0-9A-Z]{16}"), "private_key_header": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
           "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}")}


def scan_credentials(text):
    """[(kind, offset)] of credential-like strings in `text`. Never returns the matched text."""
    hits = []
    for kind, rx, needs_digit in CREDENTIAL_PATTERNS:
        for m in rx.finditer(text):
            if needs_digit and not any(c.isdigit() for c in m.group()):
                continue
            hits.append((kind, m.start()))
    return hits


_ALNUM_ORDER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PLACEHOLDER_WORDS = re.compile(r"x{4,}|example|your|sample|dummy|fake|placeholder|redacted", re.I)
_TOP_FIELD_CLASS = {"schema": "schema", "description": "description", "instructions": "instruction", "reference_json": "example data",
                    "valid_json": "example data", "erroneous_json": "example data", "data": "example data",
                    "ground_truth": "example data", "modification": "example data"}


def _entropy(text):
    n = len(text)
    counts = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1
    return -sum(v / n * math.log2(v / n) for v in counts.values()) if n else 0.0


def _sequential(text):
    idx = [_ALNUM_ORDER.find(c) for c in text.upper()]
    if len(idx) < 8 or -1 in idx:
        return False
    return sum(1 for a, b in zip(idx, idx[1:]) if b == a + 1) >= 0.7 * (len(idx) - 1)


def assess_credential(kind, match, full, start):
    """('placeholder' | 'needs_manual_review', reason). Narrow rules; anything not clearly a placeholder stays blocked.

    placeholder only when the secret part is (a) the AWS-documented example suffix EXAMPLE, (b) a private-key header with
    no key body, (c) made of at most 4 distinct characters, (d) a low-entropy string (< 3.0 bits/char), (e) an alphabet
    or digit sequence, or (f) contains an obvious placeholder word. Random-looking strings are never accepted here.
    """
    if kind == "private_key_header":
        rest = full[start + len(match):]
        end = rest.find("-----END")
        body = re.sub(r"\\[nr]|\s", "", rest if end < 0 else rest[:end])
        if len(body) < 40:
            return "placeholder", "private-key header without a key body"
        body_kind = "key-like body"
    elif kind == "aws_access_key_id":
        body = match[4:]
        if body.endswith("EXAMPLE"):
            return "placeholder", "AWS documentation example key (suffix EXAMPLE)"
        body_kind = "key id"
    elif kind == "github_token":
        body = match.split("_", 1)[1]
        body_kind = "token"
    else:
        return "needs_manual_review", f"{kind}: no placeholder rule for this kind"
    if len(set(body)) <= 4:
        return "placeholder", f"{body_kind} made of <= 4 distinct characters"
    if _entropy(body) < 3.0:
        return "placeholder", f"{body_kind} with very low entropy ({_entropy(body):.1f} bits/char)"
    if _sequential(body):
        return "placeholder", f"{body_kind} is an alphabet/digit sequence"
    if _PLACEHOLDER_WORDS.search(body):
        return "placeholder", f"{body_kind} contains a placeholder word"
    return "needs_manual_review", f"{body_kind} looks random (length {len(body)}, entropy {_entropy(body):.1f} bits/char)"


def _walk(node, path):
    """Yield (path, string) for every string in a decoded row; stringified JSON values are decoded and walked."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield path + ["<key>"], str(k)
            yield from _walk(v, path + [str(k)])
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, path + ["[]"])
    elif isinstance(node, str):
        if node[:1] in "{[":
            try:
                yield from _walk(json.loads(node), path)
                return
            except ValueError:
                pass
        yield path, node


def _iter_credential_matches(root):
    """Yield (file, line, field_path_list, text, kind, match) for every credential-like string. Internal: carries the value."""
    for d, _, fs in os.walk(root):
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(d, f), root)
            with open(os.path.join(d, f), encoding="utf-8", errors="ignore") as fh:
                for no, line in enumerate(fh, 1):
                    if f.endswith(".jsonl") and line.strip():
                        try:
                            items = list(_walk(json.loads(line), []))
                        except ValueError:
                            items = [([], line)]
                    else:
                        items = [([], line)]
                    for path, text in items:
                        for kind, rx, needs_digit in CREDENTIAL_PATTERNS:
                            for m in rx.finditer(text):
                                if needs_digit and not any(c.isdigit() for c in m.group()):
                                    continue
                                yield rel, no, path, text, kind, m


def load_reviews(path=None):
    """{body_sha256: review} from the reviews file (empty if there is none)."""
    path = path or REVIEWS_PATH
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for r in data.get("reviews", []):
        if not re.fullmatch(r"[0-9a-f]{64}", r.get("body_sha256", "")) or r.get("verdict") != REVIEWED_VERDICT:
            sys.exit(f"{path}: malformed review entry (needs a full 64-hex body_sha256 and the reviewed verdict)")
        out[r["body_sha256"]] = r
    return out


def credential_findings(root, reviews=None):
    """All credential-like strings under `root` with a sanitized description (never the matched text)."""
    out = []
    reviews = load_reviews() if reviews is None else reviews
    for rel, no, path, text, kind, m in _iter_credential_matches(root):
        cls, why = assess_credential(kind, m.group(), text, m.start())
        if cls != "placeholder" and kind == "private_key_header" and reviews:
            from key_structure import body_sha256, inspect_pem_like
            digest = body_sha256(m.group(), text, m.start())
            entry = reviews.get(digest)
            ctx = {(c["file"], c["line"], c["path"]) for c in (entry or {}).get("contexts", [])}
            # exact value AND exact place AND the same structural verdict today; anything else stays blocked
            if entry and (rel, no, ".".join(path) or "(whole line)") in ctx and \
                    inspect_pem_like(m.group(), text, m.start())["verdict"] == entry["verdict"]:
                cls, why = "reviewed_exception", f"human-reviewed exact value (sha256 {digest[:16]}...), see docs/credential_reviews.json"
        out.append({"file": rel, "line": no, "path": ".".join(path) or "(whole line)",
                    "field_class": _TOP_FIELD_CLASS.get(path[0] if path else "", "other"), "kind": kind,
                    "group": hashlib.sha256(m.group().encode("utf-8")).hexdigest()[:8],
                    "classification": cls, "reason": why})
    return out


def _credential_problems(root, verbose=True):
    found = credential_findings(root)
    placeholders = [x for x in found if x["classification"] == "placeholder"]
    reviewed = [x for x in found if x["classification"] == "reviewed_exception"]
    manual = [x for x in found if x["classification"] not in ("placeholder", "reviewed_exception")]
    if verbose:
        by_reason = {}
        for x in placeholders:
            by_reason[x["reason"]] = by_reason.get(x["reason"], 0) + 1
        print(f"  credential scan: {len(found)} credential-like string(s): {len(placeholders)} classified as placeholders, "
              f"{len(reviewed)} covered by a human-reviewed exception, {len(manual)} need manual review")
        for r, n in sorted(by_reason.items()):
            print(f"      placeholder x{n}: {r}")
    groups = {}
    for x in manual:
        groups.setdefault((x["file"], x["kind"], x["group"], x["path"]), []).append(x["line"])
    return [f"{f}: credential-like {k} (value group {g}) at {p}, line(s) {ls[:5]}{' ...' if len(ls) > 5 else ''}: needs manual review "
            f"(content not shown; run `classify-credentials`)" for (f, k, g, p), ls in sorted(groups.items())]


# ------------------------------------------------------------------ helpers

def _paths(work):
    work = os.path.abspath(work)
    return {"work": work, "cache": os.path.join(work, "hf_cache"), "verify_cache": os.path.join(work, "verify_cache"),
            "pub": os.path.join(work, "data_published", PINNED[:12]), "release_root": os.path.join(work, "release"),
            "main": os.path.join(work, "release", "main"), "marker": os.path.join(work, MARKER)}


def _hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run:  python3 -m pip install --upgrade huggingface_hub")


def _token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        tok = _hub().get_token()
    if not tok:
        sys.exit("No Hugging Face token found. Set it for this terminal session only (it is not echoed):\n"
                 "    read -s HF_TOKEN && export HF_TOKEN\n"
                 "or run `hf auth login`.")
    return tok


def _refuse_original(repo):
    if repo.strip().lower() == ORIGINAL_REPO.lower() or "/" not in repo:
        sys.exit(f"Refusing: the target must be your own repo '<owner>/<name>', never {ORIGINAL_REPO}.")


def _err(e):
    """Error text without anything that could carry a credential."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    return f"{type(e).__name__}" + (f" (HTTP {code})" if code else "")


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _known():
    return KNOWN_REVISIONS[PINNED]


# ------------------------------------------------------------------ auth-check

def _auth_problems(api, repo, allow_broad):
    problems = []
    try:
        who = api.whoami()
    except Exception as e:
        sys.exit(f"Authentication failed: {_err(e)}. Check the token (fine-grained, write access to {repo}).")
    acc = (who.get("auth") or {}).get("accessToken") or {}
    role = acc.get("role")
    print(f"  account: {who.get('name')}   token name: {acc.get('displayName')}   token type: {role}")
    if role == "fineGrained":
        fg = acc.get("fineGrained") or {}
        scoped, glob = fg.get("scoped") or [], fg.get("global") or []
        for item in scoped:
            ent = item.get("entity") or {}
            print(f"  scope: {ent.get('type')} {ent.get('name')}: {', '.join(sorted(item.get('permissions') or []))}")
        if glob:
            print(f"  global permissions: {', '.join(sorted(glob))}")
        # The Hub lists one entry per entity the token mentions. Repos have type model/dataset/space; the account itself
        # (and any org) appears as type user/org, often with an EMPTY permission list when nothing is granted there.
        repo_types = ("model", "dataset", "space")
        repo_scopes, account_perms = [], []
        for x in scoped:
            ent, perms = x.get("entity") or {}, sorted(x.get("permissions") or [])
            if not perms:
                continue  # an entity without any permission grants nothing
            if ent.get("type") in repo_types:
                repo_scopes.append((ent.get("type"), (ent.get("name") or "").lower(), set(perms)))
            else:
                account_perms.append(f"{ent.get('type')} {ent.get('name')}: {', '.join(perms)}")
        if [(t, n) for t, n, _ in repo_scopes] != [("dataset", repo.lower())]:
            problems.append(f"token must grant access to exactly one repo (dataset {repo}); found "
                            f"{[f'{t} {n}' for t, n, _ in repo_scopes] or 'none'}")
        else:
            perms = repo_scopes[0][2]
            for need in ("repo.content.read", "repo.write"):
                if need not in perms:
                    problems.append(f"token lacks '{need}' on {repo}")
        if account_perms:
            problems.append(f"token has user/org-level permissions ({'; '.join(account_perms)}); remove them")
        if glob and not allow_broad:
            problems.append(f"token has global permissions ({', '.join(sorted(glob))}); remove them or use --allow-broad-token")
    else:
        problems.append(f"token type is '{role}', not fine-grained; create a fine-grained token limited to {repo}")
    if allow_broad:
        problems = [p for p in problems if "exactly one repo" not in p and "not fine-grained" not in p
                    and "global permissions" not in p and "user/org-level" not in p]
    return problems


def cmd_auth_check(a):
    _refuse_original(a.repo_id)
    hub = _hub()
    api = hub.HfApi(token=_token())
    print(f"Checking token and repo {a.repo_id} (read only):")
    problems = _auth_problems(api, a.repo_id, a.allow_broad_token)
    try:
        info = api.dataset_info(a.repo_id)
    except Exception as e:
        sys.exit(f"Cannot read {a.repo_id}: {_err(e)}")
    print(f"  repo private: {info.private}   main commit: {info.sha}")
    if not info.private:
        problems.append("the repo is PUBLIC; make it private before uploading")
    files = api.list_repo_files(a.repo_id, repo_type="dataset")
    print(f"  files on main today: {files}")
    extra = [f for f in files if f not in (".gitattributes", "README.md")]
    if extra:
        problems.append(f"main already contains {extra}; this tool only writes into an empty main")
    refs = api.list_repo_refs(a.repo_id, repo_type="dataset")
    print(f"  branches: {[b.name for b in refs.branches]}   tags: {[t.name for t in refs.tags]}")
    _report(problems, "auth-check")


# ------------------------------------------------------------------ prepare

def _run(cmd, env=None):
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd[:3])} ...")


def cmd_prepare(a):
    _refuse_original(a.repo_id)
    P = _paths(a.work_dir)
    if os.path.isdir(P["work"]) and os.listdir(P["work"]) and not os.path.exists(P["marker"]):
        sys.exit(f"{P['work']} exists, is not empty and was not created by this tool. Choose another --work-dir.")
    if a.rebuild_release:  # keep the verified download, rebuild only release/ (README, NOTICE, PROVENANCE)
        if not os.path.exists(os.path.join(P["pub"], "published_manifest.json")):
            sys.exit("--rebuild-release needs an existing import in this work dir.")
        shutil.rmtree(P["release_root"], ignore_errors=True)
        print("rebuilding release/main from the existing verified download (no network)")
        _run([sys.executable, os.path.join(SRC, "prepare_hf_release.py"), "--variant", "main", "--baseline", P["pub"],
              "--new-repo-id", a.repo_id, "--release-root", P["release_root"]])
        _report(_verify_local(P, a.repo_id), "prepare / verify-local")
        return
    if os.path.exists(P["pub"]) or os.path.exists(P["main"]):
        sys.exit("Already prepared in this work dir. Use --rebuild-release to rebuild only release/, or run `cleanup`.")
    os.makedirs(P["work"], exist_ok=True)
    open(P["marker"], "w").write("temporary files of src/hf_baseline.py; safe to delete with `cleanup`\n")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # the original repo is public: no credential is sent to it
    print(f"1/2 downloading {ORIGINAL_REPO} @ {PINNED[:12]} into {P['work']} (no token sent)")
    _run([sys.executable, os.path.join(SRC, "import_published_benchmark.py"), "--revision", PINNED,
          "--out-dir", P["pub"], "--cache-dir", P["cache"]], env=env)
    print("2/2 building release/main")
    _run([sys.executable, os.path.join(SRC, "prepare_hf_release.py"), "--variant", "main", "--baseline", P["pub"],
          "--new-repo-id", a.repo_id, "--release-root", P["release_root"]], env=env)
    _report(_verify_local(P, a.repo_id), "prepare / verify-local")
    print(f"\nNext: python src/hf_baseline.py upload   (work dir: {P['work']})")


# ------------------------------------------------------------------ verify-local

def _count_rows(path):
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                n += 1
    return n


def _verify_local(P, repo_id=None):
    problems, rel = [], P["main"]
    if not os.path.isdir(rel):
        return [f"{rel} does not exist; run `prepare` first"]
    present = sorted(os.path.relpath(os.path.join(d, f), rel) for d, _, fs in os.walk(rel) for f in fs)
    want = sorted(DATA_FILES + list(EXTRA_FILES))
    if present != want:
        problems.append(f"release/main holds the wrong files: unexpected {sorted(set(present) - set(want))}, "
                        f"missing {sorted(set(want) - set(present))}")
    totals = {"train": 0, "test": 0}
    for name in DATA_FILES:
        p = os.path.join(rel, name)
        if not os.path.isfile(p):
            continue
        oid, size = _known()["files"][name]
        if git_blob_oid(p) != oid or os.path.getsize(p) != size:
            problems.append(f"{name}: not byte-identical to the original (git blob id / size differ)")
        orig = os.path.join(P["pub"], "original_files", name)
        if os.path.isfile(orig) and sha256_file(orig) != sha256_file(p):
            problems.append(f"{name}: differs from the downloaded original")
        split = name.split("/")[1][:-6]
        n = _count_rows(p)
        totals[split] += n
        before = len(problems)
        if n != EXPECTED_ROWS[split]:
            problems.append(f"{name}: {n} rows, expected {EXPECTED_ROWS[split]}")
        if len(problems) == before and git_blob_oid(p) == oid:
            print(f"  ok  {name}  {size} bytes  {n} rows  blob {oid[:10]}")
    want_tot = {s: len(TASKS) * n for s, n in EXPECTED_ROWS.items()}
    print(f"  totals: train {totals['train']} (expected {want_tot['train']}), test {totals['test']} (expected {want_tot['test']})")
    if totals != want_tot:
        problems.append(f"totals {totals} differ from {want_tot}")
    card = os.path.join(rel, "ORIGINAL_DATASET_CARD.md")
    if os.path.isfile(card):
        coid, csize = _known()["other_files"]["README.md"]
        if git_blob_oid(card) != coid or os.path.getsize(card) != csize:
            problems.append("ORIGINAL_DATASET_CARD.md is not the original README.md")
    readme = os.path.join(rel, "README.md")
    if os.path.isfile(readme) and parse_card_license(open(readme, encoding="utf-8").read()) != _known()["license"]:
        problems.append("README.md license differs from the original dataset's license")
    prov = os.path.join(rel, "PROVENANCE.json")
    if os.path.isfile(prov):
        pv = json.load(open(prov))
        if pv.get("original_revision") != PINNED or pv.get("original_repo") != ORIGINAL_REPO:
            problems.append("PROVENANCE.json does not name the pinned original revision")
        if repo_id and pv.get("target_repo") != repo_id:
            problems.append(f"release/main was built for repo '{pv.get('target_repo')}', not '{repo_id}'. "
                            "Rebuild it: python src/hf_baseline.py prepare --rebuild-release")
        if repo_id and os.path.isfile(os.path.join(rel, "NOTICE.md")):
            with open(os.path.join(rel, "NOTICE.md"), encoding="utf-8") as fh:
                if repo_id not in fh.read():
                    problems.append(f"NOTICE.md does not mention the target repo '{repo_id}'; rebuild with --rebuild-release")
    problems.extend(_credential_problems(rel))  # nothing credential-like may be uploaded
    return problems


def cmd_verify_local(a):
    _report(_verify_local(_paths(a.work_dir), a.repo_id), "verify-local")


# ------------------------------------------------------------------ diagnose (read only, sanitized output)

def _classify_legacy_hit(line, m, needs_digit_kind):
    pre = line[m.start() - 1] if m.start() > 0 else " "
    body = m.group()
    reasons = []
    if pre.isalnum() or pre in "_-":
        reasons.append("starts inside a longer word")
    if needs_digit_kind and not any(c.isdigit() for c in body):
        reasons.append("no digits (ordinary word)")
    return reasons or ["still flagged by the precise rules"]


def cmd_diagnose(a):
    """Explain the credential-check result and the train/test schema overlap. Prints counts and ids only, never content."""
    P = _paths(a.work_dir)
    rel = P["main"]
    if not os.path.isdir(rel):
        sys.exit("run `prepare` first")
    print("== 1. Credential check (matched text is never printed)")
    for name in DATA_FILES + list(EXTRA_FILES):
        path = os.path.join(rel, name)
        legacy, precise, why = {}, {}, {}
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for kind, rx in _LEGACY.items():
                    for m in rx.finditer(line):
                        legacy[kind] = legacy.get(kind, 0) + 1
                        for r in _classify_legacy_hit(line, m, kind in ("openai_style_key", "bearer_token")):
                            why[(kind, r)] = why.get((kind, r), 0) + 1
                for kind, _ in scan_credentials(line):
                    precise[kind] = precise.get(kind, 0) + 1
        print(f"  {name}: loose-rule hits {legacy or 0}; precise-rule hits {precise or 0}")
        for (kind, r), n in sorted(why.items()):
            print(f"      loose hit kind={kind}: {n} x {r}")
    print("\n== 2. Train/test schema overlap in the published files (content hash of the schema, key order ignored)")
    rep = overlap_report(os.path.join(P["pub"], "original_files"))
    print(f"  distinct schema contents present in BOTH splits (any task): {rep['shared_schema_contents']}")
    for task, d in rep["per_task"].items():
        print(f"  {task}: {d}")
    for k, u in enumerate(rep["schemas"], 1):
        print(f"  [{k}] source/subset={u['sources']} tasks={u['tasks']} train_id={u['train_ids']} "
              f"test_id={u['test_ids']} same_id_on_both_sides={u['same_id_on_both_sides']}")
    print("\nNothing was modified. The original files are untouched.")


def cmd_classify_credentials(a):
    """Group credential-like strings by value and field. Prints sanitized descriptions only, never the values."""
    P = _paths(a.work_dir)
    if not os.path.isdir(P["main"]):
        sys.exit("run `prepare` first")
    found = credential_findings(P["main"])
    groups = {}
    for x in found:
        groups.setdefault((x["kind"], x["group"], x["classification"], x["reason"]), []).append(x)
    print(f"{len(found)} credential-like string(s) in {len(groups)} distinct value group(s) (group = first 8 hex of sha256 of the value)\n")
    for (kind, g, cls, why), xs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0], -len(kv[1]))):
        paths = sorted({re.sub(r"\.\[\]", "[]", x["path"]) for x in xs})
        print(f"- group {g}  kind={kind}  classification={cls}  occurrences={len(xs)}")
        print(f"    why: {why}")
        print(f"    files: {sorted({x['file'] for x in xs})}")
        print(f"    where: {sorted({x['field_class'] for x in xs})}")
        print(f"    field paths ({len(paths)} distinct): {paths[:6]}{' ...' if len(paths) > 6 else ''}")
        print(f"    first locations: {[x['file'] + ':' + str(x['line']) for x in xs[:4]]}")
    manual = [x for x in found if x["classification"] not in ("placeholder", "reviewed_exception")]
    print(f"\nplaceholders: {sum(x['classification'] == 'placeholder' for x in found)}   "
          f"reviewed exceptions: {sum(x['classification'] == 'reviewed_exception' for x in found)}   need manual review: {len(manual)}")
    print("Upload stays blocked while any string needs manual review. The original files were not modified.")


def cmd_inspect_credential(a):
    """Read-only structural inspection of ONE value group. Prints safe metadata only; never the value or decoded bytes."""
    if not re.fullmatch(r"[0-9a-f]{8}", a.group or ""):
        sys.exit("--group must be 8 hex characters (as printed by classify-credentials)")
    P = _paths(a.work_dir)
    if not os.path.isdir(P["main"]):
        sys.exit("run `prepare` first")
    from key_structure import inspect_pem_like
    from key_structure import body_sha256
    want_body = (getattr(a, "body_sha256", None) or "").strip() or None
    if want_body and not re.fullmatch(r"[0-9a-f]{64}", want_body):
        sys.exit("--body-sha256 must be a full 64-character lowercase hex digest")
    n, verdicts, bodies, table = 0, {}, set(), {}
    for rel, no, path, text, kind, m in _iter_credential_matches(P["main"]):
        if hashlib.sha256(m.group().encode("utf-8")).hexdigest()[:8] != a.group:
            continue
        if want_body and (kind != "private_key_header" or body_sha256(m.group(), text, m.start()) != want_body):
            continue
        n += 1
        top = path[0] if path else ""
        print(f"\n[{n}] {rel}:{no}  field path: {'.'.join(path) or '(whole line)'}  "
              f"field class: {_TOP_FIELD_CLASS.get(top, 'other')}  kind: {kind}")
        if kind != "private_key_header":
            print("    no structural inspector for this kind; review it manually in the file")
            verdicts["unsupported_kind"] = verdicts.get("unsupported_kind", 0) + 1
            continue
        info = inspect_pem_like(m.group(), text, m.start())
        verdicts[info["verdict"]] = verdicts.get(info["verdict"], 0) + 1
        bodies.add(info["body_fingerprint"])
        cnt = table.get(info["body_sha256"], (0, info["verdict"]))[0]
        table[info["body_sha256"]] = (cnt + 1, info["verdict"])
        der = info.get("der") or {}
        print(f"    header label: '{info['label']}'   footer present: {info['footer_present']}   "
              f"footer well-formed: {info['footer_well_formed']}   footer label matches header: {info['footer_label_matches_header']}")
        print(f"    layout: body chars {info['body_chars']}, lines {info['body_lines']}, line lengths {info['line_lengths']}, "
              f"real newlines {info['real_newlines']}, escaped newlines {info['escaped_newlines']}, PEM header lines {info['pem_header_lines']}")
        print(f"    encoding: base64: {info['base64']}; characters outside the base64 alphabets: {info['chars_outside_base64_alphabets']}")
        if info.get("decoded_bytes") is not None:
            print(f"    decoded: {info['decoded_bytes']} bytes, entropy {info['decoded_entropy_bits_per_byte']} bits/byte, "
                  f"printable-ASCII fraction {info['decoded_printable_ascii_fraction']}, repeating pattern length {info['decoded_periodic_pattern_length']}, "
                  f"OpenSSH container magic: {info['openssh_magic']}")
            print(f"    ASN.1/DER: top-level SEQUENCE: {der.get('top_level_asn1_sequence')}   complete: {der.get('complete')}   "
                  f"declared/actual bytes: {der.get('declared_total_bytes')}/{der.get('actual_bytes')}   outline: {der.get('outline')}")
        if info.get("rsa_prefix"):
            r = info["rsa_prefix"]
            print(f"    RSA container walk: {r['layout']}; declared total {r['declared_total_bytes']} bytes, present {r['actual_bytes']}; "
                  f"modulus declared {r['modulus_declared_bytes']} bytes, present {r['modulus_bytes_present']}; "
                  f"bytes after the modulus (private components): {r['bytes_after_modulus']}")
        print(f"    recognized format: {info['format']}")
        print(f"    optional library check: {info['cryptography']}")
        print(f"    body fingerprint (sha256 prefix of the encoded body): {info['body_fingerprint']}   full sha256: {info['body_sha256']}")
        print(f"    classification: {info['verdict']} - {info['reason']}")
    if not n:
        sys.exit(f"no credential-like string with value group {a.group} found")
    print(f"\n{n} occurrence(s); distinct bodies: {len(bodies)}; classifications: {verdicts}")
    if table:
        print("Distinct bodies under this header (full sha256 of the encoded body; use one with --body-sha256):\n" + _body_table(table))
    print("UPLOAD REMAINS BLOCKED. This command only describes the value; it does not change the scan, whitelist anything, "
          "or modify any file. Nothing was printed, logged or sent that contains the value.")


def _body_table(rows):
    """Printable summary of the distinct bodies seen under a header: digest, occurrences, verdict (never the value)."""
    lines = []
    for digest, (n, verdict) in sorted(rows.items(), key=lambda kv: (-kv[1][0], kv[0])):
        lines.append(f"    sha256 {digest}  occurrences {n}  structural verdict {verdict}")
    return "\n".join(lines) or "    (none)"


def cmd_record_credential_review(a):
    """Record a human review for ONE exact body, selected by its full SHA-256 (not by the header group).

    Only the verdict 'truncated RSA container, public part only' qualifies. Every other body that shares the same
    header, and any occurrence beyond the expected ones, stays subject to the normal checks.
    """
    target = (a.body_sha256 or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", target):
        sys.exit("--body-sha256 must be the full 64-character lowercase hex digest printed by inspect-credential.")
    P = _paths(a.work_dir)
    if not os.path.isdir(P["main"]):
        sys.exit("run `prepare` first")
    from key_structure import body_sha256, inspect_pem_like
    path = getattr(a, "reviews_path", None) or REVIEWS_PATH
    seen, occ, info = {}, [], None
    for rel, no, fpath, text, kind, m in _iter_credential_matches(P["main"]):
        if kind != "private_key_header":
            continue
        digest = body_sha256(m.group(), text, m.start())
        if digest == target:
            info = inspect_pem_like(m.group(), text, m.start())
            occ.append({"file": rel, "line": no, "path": ".".join(fpath) or "(whole line)"})
            verdict = info["verdict"]
        else:
            n, verdict = seen.get(digest, (0, None))
            seen[digest] = (n + 1, verdict or inspect_pem_like(m.group(), text, m.start())["verdict"])
    if not occ:
        sys.exit("No private-key block has that body digest. Bodies present in the data (digests only):\n"
                 + _body_table(seen) + "\nCopy the digest from the output of `inspect-credential`. Nothing was written.")
    if len(occ) != a.expect_occurrences:
        sys.exit(f"Found {len(occ)} occurrence(s) of that body but --expect-occurrences is {a.expect_occurrences}. "
                 "Not recording a review for an unexpected number of places. Nothing was written.")
    if info["verdict"] != REVIEWED_VERDICT:
        sys.exit(f"Not eligible: the structural verdict for this body is '{info['verdict']}', not '{REVIEWED_VERDICT}'. "
                 "Upload stays blocked.")
    r = info["rsa_prefix"]
    print(f"Body {target}\n  {len(occ)} occurrence(s). Other bodies under private-key headers ({len(seen)} distinct) are not affected "
          f"and keep their existing classification:\n{_body_table(seen)}")
    print(f"  structure: {r['layout']} container, declared {r['declared_total_bytes']} bytes, present {r['actual_bytes']}; "
          f"modulus {r['modulus_bytes_present']} of {r['modulus_declared_bytes']} bytes present; "
          f"bytes after the modulus (private components): {r['bytes_after_modulus']}")
    print(f"  library check: {info['cryptography']}")
    for o in occ:
        print(f"  context: {o['file']}:{o['line']}  {o['path']}")
    print("Meaning: the value is the beginning of an RSA key container cut off inside the PUBLIC modulus. It contains no private "
          "exponent, primes or CRT values and cannot be completed into a key. Residual risk: it may be the first bytes of a real "
          "key's public modulus.\nThe exception applies only to this exact digest at exactly these places.")
    try:
        typed = input(f"Type the first 16 characters of the digest ({target[:16]}) to record this review: ").strip()
        who_ = input("Your name for the record: ").strip()
    except EOFError:
        sys.exit("Not confirmed. Nothing was written.")
    if typed != target[:16] or not who_:
        sys.exit("Not confirmed. Nothing was written.")
    data = {"version": 1, "reviews": []}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    if any(r_.get("body_sha256") == target for r_ in data["reviews"]):
        sys.exit("This value is already recorded.")
    data["reviews"].append({
        "id": target[:8], "kind": "private_key_header", "label": info["label"], "body_sha256": target, "verdict": REVIEWED_VERDICT,
        "decision": "allowed in these exact contexts only; the value is not a usable key",
        "evidence": {"decoded_bytes": info["decoded_bytes"], "base64": info["base64"], "rsa_container": r,
                     "library_check": info["cryptography"], "footer_present": info["footer_present"],
                     "body_chars": info["body_chars"]},
        "residual_risk": "may be the first bytes of the public modulus of a real key; no private components are present",
        "contexts": occ, "reviewed_by": who_, "reviewed_on": _now()})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Recorded in {path}. Keep this file with the code (it contains digests and places, not the value). Not committed.")


# ------------------------------------------------------------------ upload

def cmd_upload(a):
    _refuse_original(a.repo_id)
    P = _paths(a.work_dir)
    print("Local checks:")
    local = _verify_local(P, a.repo_id)
    if local:
        _report(local, "verify-local")
    hub = _hub()
    api = hub.HfApi(token=_token())
    print("Remote checks:")
    problems = _auth_problems(api, a.repo_id, a.allow_broad_token)
    info = api.dataset_info(a.repo_id)
    if not info.private:
        problems.append("the repo is PUBLIC; make it private first")
    files = api.list_repo_files(a.repo_id, repo_type="dataset")
    extra = [f for f in files if f not in (".gitattributes", "README.md")]
    if extra:
        problems.append(f"main already contains {extra}; this tool only writes into an empty main")
    _report(problems, "pre-upload checks")
    plan = sorted(os.path.relpath(os.path.join(d, f), P["main"]) for d, _, fs in os.walk(P["main"]) for f in fs)
    print(f"\nAbout to create ONE commit on branch main of {a.repo_id} (private) with {len(plan)} files:")
    for f in plan:
        print(f"   {f}  ({os.path.getsize(os.path.join(P['main'], f))} bytes)")
    print(f"Source: {ORIGINAL_REPO} @ {PINNED}. No other branch is created. The original repo is not touched.")
    if not sys.stdin.isatty():
        sys.exit("Interactive confirmation required (run this in a terminal).")
    if input(f"Type the repo id '{a.repo_id}' to upload: ").strip() != a.repo_id:
        sys.exit("Not confirmed. Nothing was uploaded.")
    commit = api.upload_folder(
        repo_id=a.repo_id, repo_type="dataset", folder_path=P["main"], revision="main",
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
        commit_message=f"Baseline: copy of {ORIGINAL_REPO} @ {PINNED[:12]} (data files byte-identical)",
        commit_description=f"Source revision {PINNED}. See PROVENANCE.json and NOTICE.md.")
    sha = getattr(commit, "oid", None)
    json.dump({"repo": a.repo_id, "branch": "main", "commit": sha, "source": ORIGINAL_REPO, "source_revision": PINNED,
               "uploaded": _now(), "files": plan,
               "credential_reviews_applied": sorted({x["reason"] for x in credential_findings(P["main"])
                                                      if x["classification"] == "reviewed_exception"})}, open(os.path.join(P["work"], "upload_receipt.json"), "w"), indent=2)
    print(f"\nUploaded. Commit: {sha}\nNext: python src/hf_baseline.py verify-remote")


# ------------------------------------------------------------------ verify-remote

def cmd_verify_remote(a):
    _refuse_original(a.repo_id)
    P = _paths(a.work_dir)
    hub = _hub()
    tok = _token()
    api = hub.HfApi(token=tok)
    sha = a.commit or api.dataset_info(a.repo_id, revision="main").sha
    print(f"Verifying {a.repo_id} @ {sha}")
    problems = []
    tree = {t.path: t for t in api.list_repo_tree(a.repo_id, repo_type="dataset", revision=sha, recursive=True)
            if getattr(t, "size", None) is not None and not str(getattr(t, "path", "")).endswith("/")}
    names = sorted(p for p in tree if p != ".gitattributes")
    if names != sorted(DATA_FILES + list(EXTRA_FILES)):
        problems.append(f"unexpected file set on the remote: {names}")
    for name in DATA_FILES:  # 1) metadata reported by the Hub
        oid, size = _known()["files"][name]
        t = tree.get(name)
        if t is None:
            problems.append(f"{name}: missing on the remote")
            continue
        lfs = getattr(t, "lfs", None)
        if t.size != size:
            problems.append(f"{name}: remote size {t.size} != {size}")
        if not lfs and t.blob_id != oid:
            problems.append(f"{name}: remote git blob id {t.blob_id} != original {oid}")
    shutil.rmtree(P["verify_cache"], ignore_errors=True)
    local_main, pub = P["main"], os.path.join(P["pub"], "original_files")
    for name in DATA_FILES + list(EXTRA_FILES):  # 2) real content: download and hash
        if name not in tree:
            continue
        got = hub.hf_hub_download(a.repo_id, name, repo_type="dataset", revision=sha, cache_dir=P["verify_cache"], token=tok)
        if name in DATA_FILES:
            oid, size = _known()["files"][name]
            if git_blob_oid(got) != oid or os.path.getsize(got) != size:
                problems.append(f"{name}: downloaded content differs from the original (git blob id / size)")
            if os.path.isfile(os.path.join(pub, name)) and sha256_file(got) != sha256_file(os.path.join(pub, name)):
                problems.append(f"{name}: sha256 differs from the original download")
            n = _count_rows(got)
            if n != EXPECTED_ROWS[name.split("/")[1][:-6]]:
                problems.append(f"{name}: {n} rows")
            else:
                print(f"  ok  {name}  {n} rows  sha256 {sha256_file(got)[:16]}")
        elif os.path.isfile(os.path.join(local_main, name)) and sha256_file(got) != sha256_file(os.path.join(local_main, name)):
            problems.append(f"{name}: differs from the file that was uploaded")
    try:  # 3) information only
        refs = api.list_repo_refs(a.repo_id, repo_type="dataset")
        print(f"  branches on the new repo: {[b.name for b in refs.branches]}")
        orig = hub.HfApi().dataset_info(ORIGINAL_REPO, token=False)
        print(f"  original repo main is at {orig.sha}" + ("" if orig.sha == PINNED else
              " (it has moved on since; the pinned copy is unaffected)"))
    except Exception as e:
        print(f"  (informational lookup skipped: {_err(e)})")
    json.dump({"repo": a.repo_id, "commit": sha, "checked": _now(), "result": "FAIL" if problems else "PASS",
               "problems": problems}, open(os.path.join(P["work"], "verify_receipt.json"), "w"), indent=2)
    _report(problems, "verify-remote")
    if not problems:
        print(f"Record this in the experiment log: {a.repo_id} main = {sha}, copied from {ORIGINAL_REPO} @ {PINNED}")


# ------------------------------------------------------------------ cleanup

def cmd_cleanup(a):
    P = _paths(a.work_dir)
    work = P["work"]
    if os.path.islink(work) or not os.path.isdir(work):
        sys.exit(f"{work} is not a plain directory; nothing to do.")
    if not os.path.exists(P["marker"]):
        sys.exit(f"{work} was not created by this tool (no {MARKER} marker); refusing to delete it.")
    home = os.path.expanduser("~")
    if work in ("/", home, ROOT) or ROOT.startswith(work + os.sep) or home.startswith(work + os.sep):
        sys.exit("Refusing to delete this folder.")
    rec = os.path.join(work, "verify_receipt.json")
    if not (os.path.isfile(rec) and json.load(open(rec)).get("result") == "PASS") and not a.force:
        sys.exit("verify-remote has not passed in this work dir. Run it first (or use --force).")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(work) for f in fs)
    print(f"Will delete {work} ({size / 1e6:.1f} MB): downloaded original, release/main, caches.")
    if not sys.stdin.isatty() or input("Type 'delete' to confirm: ").strip() != "delete":
        sys.exit("Not confirmed. Nothing was deleted.")
    shutil.rmtree(work)
    print("Deleted. Also remove the token from this terminal: `unset HF_TOKEN`, and revoke it at "
          "https://huggingface.co/settings/tokens when you no longer need it.")


# ------------------------------------------------------------------ main

def _report(problems, what):
    if problems:
        print(f"\n{what}: FAILED")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"\n{what}: PASSED")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-id", default=DEFAULT_REPO, help=f"our repo (default {DEFAULT_REPO}); the original is refused")
    common.add_argument("--work-dir", default=os.path.join(tempfile.gettempdir(), "benchson_hf_transfer"),
                        help="temporary folder for downloads and the release folder")
    common.add_argument("--allow-broad-token", action="store_true", help="accept a token that is not scoped to this repo only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("auth-check", cmd_auth_check), ("prepare", cmd_prepare), ("verify-local", cmd_verify_local),
                     ("diagnose", cmd_diagnose), ("classify-credentials", cmd_classify_credentials),
                     ("inspect-credential", cmd_inspect_credential),
                     ("record-credential-review", cmd_record_credential_review),
                     ("upload", cmd_upload), ("verify-remote", cmd_verify_remote), ("cleanup", cmd_cleanup)):
        p = sub.add_parser(name, parents=[common])
        p.set_defaults(fn=fn)
        if name == "verify-remote":
            p.add_argument("--commit", help="check this commit instead of the current head of main")
        if name == "inspect-credential":
            p.add_argument("--group", required=True, help="8-hex value group (header) from classify-credentials")
            p.add_argument("--body-sha256", help="only the body with this full sha256 (64 hex)")
        if name == "record-credential-review":
            p.add_argument("--body-sha256", required=True, help="full sha256 (64 hex) of the exact body to review")
            p.add_argument("--expect-occurrences", type=int, required=True, help="how many places must contain this body")
        if name == "prepare":
            p.add_argument("--rebuild-release", action="store_true", help="reuse the verified download, rebuild release/ only")
        if name == "cleanup":
            p.add_argument("--force", action="store_true", help="delete even if verify-remote has not passed")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

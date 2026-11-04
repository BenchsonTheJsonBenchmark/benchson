import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import hf_baseline as hb
from data_generation.published import parse_card_license
from test_import_published import _snapshot

CARD = "---\nlicense: mit\n---\n# Original card\n"


def run(script, *a, cwd, stdin=subprocess.DEVNULL):
    return subprocess.run([sys.executable, os.path.join(ROOT, "src", script), *a], capture_output=True, text=True,
                          cwd=cwd, stdin=stdin)


class FakeApi:
    def __init__(self, who):
        self._who = who

    def whoami(self):
        return self._who


def who(role, scoped=None, glob=None):
    return {"name": "example-user", "auth": {"accessToken": {"displayName": "t", "role": role, "fineGrained": {
        "scoped": scoped or [], "global": glob or []}}}}


def scope(name, perms):
    return {"entity": {"type": "dataset", "name": name}, "permissions": perms}


class CardAndNotice(unittest.TestCase):
    def test_license_parse(self):
        self.assertEqual(parse_card_license(CARD), "mit")
        self.assertIsNone(parse_card_license("# no front matter\nlicense: mit\n"))

    def test_release_has_notice_card_and_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, base = os.path.join(tmp, "src"), os.path.join(tmp, "pub")
            _snapshot(src)
            open(os.path.join(src, "README.md"), "w").write(CARD)
            r = run("import_published_benchmark.py", "--revision", "b" * 40, "--source-dir", src, "--out-dir", base,
                    "--allow-count-mismatch", cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.load(open(os.path.join(base, "published_manifest.json")))["original_card"]["license"], "mit")
            r = run("prepare_hf_release.py", "--variant", "main", "--baseline", base, "--new-repo-id", "x/y", cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            rel = os.path.join(tmp, "release", "main")
            self.assertEqual(sorted(os.listdir(rel)), ["NOTICE.md", "ORIGINAL_DATASET_CARD.md", "PROVENANCE.json", "README.md",
                                                       "create", "fix", "modify"])
            self.assertEqual(open(os.path.join(rel, "ORIGINAL_DATASET_CARD.md")).read(), CARD)
            self.assertIn("license: mit", open(os.path.join(rel, "README.md")).read())
            self.assertIn("arXiv:2501.10868", open(os.path.join(rel, "NOTICE.md")).read())
            prov = json.load(open(os.path.join(rel, "PROVENANCE.json")))
            self.assertEqual(prov["license"], "mit"); self.assertIn("NOTICE.md", prov["other_files"])

    def test_release_refused_without_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, base = os.path.join(tmp, "src"), os.path.join(tmp, "pub")
            _snapshot(src)
            run("import_published_benchmark.py", "--revision", "b" * 40, "--source-dir", src, "--out-dir", base,
                "--allow-count-mismatch", cwd=tmp)
            r = run("prepare_hf_release.py", "--variant", "main", "--baseline", base, "--new-repo-id", "x/y", cwd=tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("license", r.stderr + r.stdout)


class LocalChecksPositive(unittest.TestCase):
    def test_verify_local_passes_for_a_faithful_copy(self):
        from data_generation.published import git_blob_oid
        fake = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            src, base = os.path.join(tmp, "src"), os.path.join(tmp, "work", "data_published", fake[:12])
            _snapshot(src)
            open(os.path.join(src, "README.md"), "w").write(CARD)
            self.assertEqual(run("import_published_benchmark.py", "--revision", fake, "--source-dir", src, "--out-dir", base,
                                 "--allow-count-mismatch", cwd=tmp).returncode, 0)
            rel_root = os.path.join(tmp, "work", "release")
            r = run("prepare_hf_release.py", "--variant", "main", "--baseline", base, "--new-repo-id", "x/y",
                    "--release-root", rel_root, cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            files = {n: (git_blob_oid(os.path.join(src, n)), os.path.getsize(os.path.join(src, n))) for n in hb.DATA_FILES}
            rows = {s: sum(1 for _ in open(os.path.join(src, "create", f"{s}.jsonl"))) for s in ("train", "test")}
            saved = (hb.PINNED, dict(hb.KNOWN_REVISIONS), dict(hb.EXPECTED_ROWS))
            try:
                hb.PINNED = fake
                hb.KNOWN_REVISIONS[fake] = {"files": files, "license": "mit", "other_files": {
                    "README.md": (git_blob_oid(os.path.join(src, "README.md")), os.path.getsize(os.path.join(src, "README.md")))}}
                hb.EXPECTED_ROWS.update(rows)
                P = hb._paths(os.path.join(tmp, "work"))
                self.assertEqual(hb._verify_local(P), [])
                self.assertEqual(hb._verify_local(P, "x/y"), [])                      # built for this repo
                wrong = hb._verify_local(P, hb.DEFAULT_REPO)                           # built for another repo: must be rebuilt
                self.assertTrue(any("--rebuild-release" in p for p in wrong), wrong)
                open(os.path.join(P["main"], "fix", "test.jsonl"), "ab").write(b"\n")  # one stray byte
                self.assertTrue(any("not byte-identical" in p for p in hb._verify_local(P)))
            finally:
                hb.PINNED = saved[0]; hb.KNOWN_REVISIONS.clear(); hb.KNOWN_REVISIONS.update(saved[1])
                hb.EXPECTED_ROWS.update(saved[2])


class CredentialDetector(unittest.TestCase):
    def kinds(self, text):
        return sorted({k for k, _ in hb.scan_credentials(text)})

    def test_real_looking_credentials_are_found(self):
        alnum = "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3z"
        cases = {"huggingface_token": "hf_" + alnum,
                 "openai_style_key": 'key: "sk-' + "A1b2C3d4" * 3 + '"',
                 "aws_access_key_id": "id AKIA" + "ABCDEFGH12345678",
                 "private_key_header": "-----BEGIN " + "RSA PRIVATE KEY-----",
                 "bearer_token": "Authorization: Bearer " + "abc123." * 4,
                 "github_token": "ghp_" + alnum + "0123"}
        for kind, text in cases.items():
            self.assertIn(kind, self.kinds(text), kind)

    def test_false_positives_seen_in_the_published_data_are_ignored(self):
        # Ordinary hyphenated names in schemas: letters only, and the "sk-" starts inside a longer word.
        word = "tta" + "sk-" + "Abcdefghijklmnopqrstuvwx"
        self.assertEqual(self.kinds(f'{{"x": "{word}"}}'), [])
        self.assertEqual(self.kinds("disk-" + "a1" * 12), [])                 # starts inside a word, even with digits
        self.assertEqual(self.kinds("sk-" + "abcdefghij" * 3), [])            # no digit: an ordinary word
        self.assertEqual(self.kinds("chf_" + "abcdefghijklmnopqrstuvwxyz0123456789"), [])  # hf_ inside a longer word
        self.assertEqual(self.kinds("AKIA[0-9A-Z]{16}"), [])                  # a regex that describes keys
        self.assertEqual(self.kinds("XAKIA" + "ABCDEFGH12345678"), [])        # inside a longer uppercase run
        self.assertEqual(self.kinds("Bearer token"), [])
        self.assertEqual(self.kinds("Bearer " + "abcdefghijklmnopqrstuvwxyz"), [])  # no digit

    def test_problem_messages_never_contain_the_matched_text(self):
        secret = "hf_" + "Zq8" * 12
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.jsonl"), "w").write('{"k": 1}\n{"k": "' + secret + '"}\n')
            msgs = hb._credential_problems(tmp)
            self.assertEqual(len(msgs), 1)
            self.assertIn("line(s) [2]", msgs[0]); self.assertNotIn(secret, msgs[0]); self.assertNotIn("Zq8", msgs[0])


def _rand(n, seed=7):
    import random
    r = random.Random(seed)
    return "".join(r.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(n))


class CredentialTriage(unittest.TestCase):
    def verdict(self, kind, value, prefix="", suffix=""):
        full = prefix + value + suffix
        return hb.assess_credential(kind, value, full, len(prefix))[0]

    def test_documented_and_obvious_placeholders(self):
        self.assertEqual(self.verdict("aws_access_key_id", "AKIA" + "IOSFODNN7EXAMPLE"), "placeholder")
        self.assertEqual(self.verdict("aws_access_key_id", "AKIA" + "1234567890ABCDEF"), "placeholder")
        self.assertEqual(self.verdict("aws_access_key_id", "AKIA" + "A" * 16), "placeholder")
        self.assertEqual(self.verdict("github_token", "ghp_" + "x" * 36), "placeholder")
        self.assertEqual(self.verdict("github_token", "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"), "placeholder")
        self.assertEqual(self.verdict("github_token", "ghp_" + "YOUR" + _rand(32)), "placeholder")

    def test_random_looking_values_stay_blocked(self):
        self.assertEqual(self.verdict("aws_access_key_id", "AKIA" + _rand(16).upper()), "needs_manual_review")
        self.assertEqual(self.verdict("github_token", "ghp_" + _rand(36)), "needs_manual_review")
        self.assertEqual(self.verdict("huggingface_token", "hf_" + _rand(34)), "needs_manual_review")  # no rule for this kind
        self.assertEqual(self.verdict("openai_style_key", "sk-" + _rand(40)), "needs_manual_review")

    def test_private_key_header_with_and_without_body(self):
        head = "-----BEGIN " + "PRIVATE KEY-----"
        for text in (head, head + "\\n...\\n-----END PRIVATE KEY-----", head + "\n<your key here>\n-----END PRIVATE KEY-----"):
            self.assertEqual(self.verdict("private_key_header", head, suffix=text[len(head):]), "placeholder", text)
        body = _rand(300)
        self.assertEqual(self.verdict("private_key_header", head, suffix="\n" + body + "\n-----END PRIVATE KEY-----"),
                         "needs_manual_review")
        self.assertEqual(self.verdict("private_key_header", head, suffix="\n" + "A" * 300), "placeholder")

    def test_findings_have_field_paths_and_never_values(self):
        aws_ex = "AKIA" + "IOSFODNN7EXAMPLE"
        real_like = "AKIA" + _rand(16, 3).upper()
        row = {"name": "n", "schema": json.dumps({"type": "object", "properties": {"k": {"type": "string", "default": real_like}}}),
               "description": f"The key is {aws_ex} in the config.",
               "reference_json": json.dumps({"creds": [{"id": real_like}], "ok": 1})}
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "create.jsonl"), "w").write(json.dumps(row) + "\n")
            found = hb.credential_findings(tmp)
            by = {(x["field_class"], x["classification"]) for x in found}
            self.assertEqual(by, {("schema", "needs_manual_review"), ("description", "placeholder"),
                                  ("example data", "needs_manual_review")})
            paths = {x["path"] for x in found}
            self.assertIn("reference_json.creds.[].id", paths); self.assertIn("schema.properties.k.default", paths)
            self.assertEqual(len({x["group"] for x in found if x["classification"] == "needs_manual_review"}), 1)  # same value
            for text in (repr(found), "\n".join(hb._credential_problems(tmp, verbose=False))):
                self.assertNotIn(real_like, text); self.assertNotIn(aws_ex, text)

    def test_classify_command_prints_no_values(self):
        import contextlib, io
        real_like = "ghp_" + _rand(36, 5)
        with tempfile.TemporaryDirectory() as tmp:
            P = hb._paths(tmp)
            os.makedirs(P["main"])
            open(os.path.join(P["main"], "x.jsonl"), "w").write(json.dumps({"description": f"token {real_like}"}) + "\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                hb.cmd_classify_credentials(type("A", (), {"work_dir": tmp})())
            out = buf.getvalue()
            self.assertIn("need manual review: 1", out); self.assertNotIn(real_like, out); self.assertNotIn(real_like[4:12], out)


class OverlapAndLimitation(unittest.TestCase):
    def _dup_snapshot(self, root, identical=False):
        from data_generation.published import encode_record
        _snapshot(root)
        for task in ("create", "fix", "modify"):
            rows = [json.loads(l) for l in open(os.path.join(root, task, "train.jsonl"))]
            dup = dict(rows[0]); dup["name"] = "copy.json"
            if not identical:
                if task == "create":
                    dup["reference_json"] = json.dumps({"a": 99})
                elif task == "fix":
                    dup["valid_json"] = json.dumps({"a": "99"})
                else:
                    dup["data"] = json.dumps({"a": 99})
            with open(os.path.join(root, task, "test.jsonl"), "a") as f:
                f.write(json.dumps(dup) + "\n")

    def test_report_counts_shared_schemas_and_identical_examples(self):
        from data_generation.published import overlap_report
        with tempfile.TemporaryDirectory() as tmp:
            self._dup_snapshot(tmp)
            rep = overlap_report(tmp)
            self.assertEqual(rep["shared_schema_contents"], 1)
            self.assertFalse(rep["schemas"][0]["same_id_on_both_sides"])
            for d in rep["per_task"].values():
                self.assertEqual((d["shared_schemas"], d["identical_rows"], d["identical_examples_incl_schema"]), (1, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            self._dup_snapshot(tmp, identical=True)
            self.assertGreater(overlap_report(tmp)["per_task"]["create"]["identical_examples_ignoring_schema"], 0)

    def test_release_documents_limitation_without_touching_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, base = os.path.join(tmp, "src"), os.path.join(tmp, "pub")
            self._dup_snapshot(src)
            open(os.path.join(src, "README.md"), "w").write(CARD)
            self.assertEqual(run("import_published_benchmark.py", "--revision", "b" * 40, "--source-dir", src, "--out-dir", base,
                                 "--allow-count-mismatch", cwd=tmp).returncode, 0)
            r = run("prepare_hf_release.py", "--variant", "main", "--baseline", base, "--new-repo-id", "x/y", cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            rel = os.path.join(tmp, "release", "main")
            for name in ("README.md", "NOTICE.md"):
                text = open(os.path.join(rel, name)).read()
                self.assertIn("Known limitation of the original benchmark", text)
                self.assertIn("1 schema contents", text)
            prov = json.load(open(os.path.join(rel, "PROVENANCE.json")))
            self.assertEqual(prov["known_limitations"]["train_test_overlap"]["shared_schema_contents"], 1)
            for t in ("create", "fix", "modify"):  # data files stay byte-identical, including the duplicate-schema rows
                for sp in ("train", "test"):
                    self.assertEqual(open(os.path.join(src, t, f"{sp}.jsonl"), "rb").read(),
                                     open(os.path.join(rel, t, f"{sp}.jsonl"), "rb").read())


class Diagnose(unittest.TestCase):
    def test_diagnose_reports_without_printing_content(self):
        import contextlib, io
        from data_generation.published import git_blob_oid
        fake = "b" * 40
        secret = "sk-" + "Q7" * 14
        with tempfile.TemporaryDirectory() as tmp:
            src, base = os.path.join(tmp, "src"), os.path.join(tmp, "work", "data_published", fake[:12])
            _snapshot(src)
            open(os.path.join(src, "README.md"), "w").write(CARD)
            self.assertEqual(run("import_published_benchmark.py", "--revision", fake, "--source-dir", src, "--out-dir", base,
                                 "--allow-count-mismatch", cwd=tmp).returncode, 0)
            self.assertEqual(run("prepare_hf_release.py", "--variant", "main", "--baseline", base, "--new-repo-id", "x/y",
                                 "--release-root", os.path.join(tmp, "work", "release"), cwd=tmp).returncode, 0)
            P = hb._paths(os.path.join(tmp, "work"))
            open(os.path.join(P["main"], "NOTICE.md"), "a").write("\nleaked " + secret + "\n")
            saved = hb.PINNED
            buf = io.StringIO()
            try:
                hb.PINNED = fake
                with contextlib.redirect_stdout(buf):
                    hb.cmd_diagnose(type("A", (), {"work_dir": os.path.join(tmp, "work")})())
            finally:
                hb.PINNED = saved
            out = buf.getvalue()
            self.assertIn("schema contents present in BOTH splits", out)
            self.assertIn("NOTICE.md", out)
            self.assertNotIn(secret, out); self.assertNotIn("Q7Q7", out)


class TokenScope(unittest.TestCase):
    REPO = "benchson-research/benchson-data-experiments"

    def check(self, w, allow=False):
        return hb._auth_problems(FakeApi(w), self.REPO, allow)

    def test_default_repo_is_the_research_org(self):
        self.assertEqual(hb.DEFAULT_REPO, "benchson-research/benchson-data-experiments")
        self.assertEqual(self.REPO, hb.DEFAULT_REPO)

    def test_token_scoped_to_the_previous_owner_is_rejected(self):
        old = "example-user/benchson-data-experiments"
        self.assertTrue(any("exactly one repo" in p for p in
                            self.check(who("fineGrained", [scope(old, ["repo.content.read", "repo.write"])]))))

    def test_org_scope_entry_without_permissions_is_ignored_but_with_permissions_rejected(self):
        org_empty = {"entity": {"type": "org", "name": "benchson-research"}, "permissions": []}
        user_empty = {"entity": {"type": "user", "name": "example-user"}, "permissions": []}
        good = scope(self.REPO, ["repo.access.read", "repo.content.read", "repo.write"])
        self.assertEqual(self.check(who("fineGrained", [good, org_empty, user_empty])), [])
        org_write = {"entity": {"type": "org", "name": "benchson-research"}, "permissions": ["repo.write"]}
        self.assertTrue(any("user/org-level" in p for p in self.check(who("fineGrained", [good, org_write]))))

    def test_good_token(self):
        self.assertEqual(self.check(who("fineGrained", [scope(self.REPO, ["repo.content.read", "repo.write"])])), [])

    def test_classic_write_token_rejected(self):
        self.assertTrue(self.check(who("write")))

    def test_exact_response_seen_on_the_hub(self):
        # Real structure from `auth-check`: the repo scope plus an EMPTY user-level scope entry.
        w = who("fineGrained", [scope(self.REPO, ["repo.access.read", "repo.content.read", "repo.write"]),
                                {"entity": {"type": "user", "name": "example-user"}, "permissions": []}])
        self.assertEqual(self.check(w), [])

    def test_empty_scopes_never_grant_anything(self):
        only_empty = who("fineGrained", [{"entity": {"type": "user", "name": "example-user"}, "permissions": []}])
        self.assertTrue(any("exactly one repo" in p for p in self.check(only_empty)))
        empty_repo = who("fineGrained", [scope(self.REPO, ["repo.write", "repo.content.read"]),
                                         scope("benchson-research/other", [])])
        self.assertEqual(self.check(empty_repo), [])  # an empty second repo entry authorizes nothing

    def test_user_or_org_permissions_rejected(self):
        for ent in ({"type": "user", "name": "example-user"}, {"type": "org", "name": "some-org"}):
            w = who("fineGrained", [scope(self.REPO, ["repo.content.read", "repo.write"]),
                                    {"entity": ent, "permissions": ["repo.write"]}])
            self.assertTrue(any("user/org-level" in p for p in self.check(w)), ent)

    def test_second_authorized_repo_or_other_type_rejected(self):
        for second in (scope("benchson-research/other", ["repo.content.read"]),
                       {"entity": {"type": "model", "name": "benchson-research/m"}, "permissions": ["repo.write"]}):
            w = who("fineGrained", [scope(self.REPO, ["repo.content.read", "repo.write"]), second])
            self.assertTrue(any("exactly one repo" in p for p in self.check(w)))

    def test_missing_read_or_write_rejected(self):
        self.assertTrue(self.check(who("fineGrained", [scope(self.REPO, ["repo.write"])])))
        self.assertTrue(self.check(who("fineGrained", [scope(self.REPO, ["repo.content.read"])])))

    def test_other_repo_or_read_only_rejected(self):
        self.assertTrue(self.check(who("fineGrained", [scope("someone/else", ["repo.write"])])))
        self.assertTrue(self.check(who("fineGrained", [scope(self.REPO, ["repo.content.read"])])))

    def test_two_repos_and_global_rejected(self):
        two = [scope(self.REPO, ["repo.write"]), scope("benchson-research/other", ["repo.write"])]
        self.assertTrue(self.check(who("fineGrained", two)))
        self.assertTrue(self.check(who("fineGrained", [scope(self.REPO, ["repo.write"])], ["post.write"])))


class LocalChecksAndSafety(unittest.TestCase):
    def test_verify_local_flags_non_identical_files_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            P = hb._paths(tmp)
            os.makedirs(os.path.join(P["main"], "create"))
            open(os.path.join(P["main"], "create", "train.jsonl"), "w").write('{"a": 1}\n')
            open(os.path.join(P["main"], "README.md"), "w").write("token hf_" + "A" * 30)
            problems = hb._verify_local(P)
            text = "\n".join(problems)
            self.assertIn("not byte-identical", text)
            self.assertIn("wrong files", text)
            self.assertIn("credential", text)

    def test_original_repo_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cmd in ("prepare", "upload", "verify-remote", "auth-check"):
                r = run("hf_baseline.py", cmd, "--repo-id", "aviv1ron1/Benchson", "--work-dir", os.path.join(tmp, "w"), cwd=tmp)
                self.assertNotEqual(r.returncode, 0); self.assertIn("Refusing", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "w")))

    def test_cleanup_refuses_unmarked_and_unverified_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = os.path.join(tmp, "w"); os.makedirs(w)
            open(os.path.join(w, "keep.txt"), "w").write("x")
            r = run("hf_baseline.py", "cleanup", "--work-dir", w, cwd=tmp)
            self.assertNotEqual(r.returncode, 0); self.assertIn("marker", r.stderr)
            open(os.path.join(w, hb.MARKER), "w").write("x")
            r = run("hf_baseline.py", "cleanup", "--work-dir", w, cwd=tmp)
            self.assertNotEqual(r.returncode, 0); self.assertIn("verify-remote", r.stderr)
            self.assertTrue(os.path.exists(os.path.join(w, "keep.txt")))

    def test_token_is_not_a_cli_argument(self):
        r = run("hf_baseline.py", "auth-check", "--token", "x", cwd=ROOT)
        self.assertNotEqual(r.returncode, 0)

    def test_pinned_revision_known(self):
        self.assertIn(hb.PINNED, hb.KNOWN_REVISIONS)
        self.assertEqual(len(hb.DATA_FILES), 6)


if __name__ == "__main__":
    unittest.main()

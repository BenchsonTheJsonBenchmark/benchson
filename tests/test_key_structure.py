import base64, contextlib, hashlib, io, json, os, random, re, sys, tempfile, textwrap, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import hf_baseline as hb
from key_structure import inspect_pem_like

HEAD = "-----BEGIN " + "PRIVATE KEY-----"
FOOT = "-----END " + "PRIVATE KEY-----"


def rbytes(n, seed):
    r = random.Random(seed)
    return bytes(r.randrange(256) for _ in range(n))


def ed25519_pkcs8(seed=1):
    return bytes([0x30, 0x2E, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2B, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20]) + rbytes(32, seed)


def pem(der_or_text, wrap=None, escaped=False, footer=FOOT):
    body = base64.b64encode(der_or_text).decode() if isinstance(der_or_text, bytes) else der_or_text
    nl = "\\n" if escaped else "\n"
    if wrap:
        body = nl.join(textwrap.wrap(body, wrap))
    return f"{HEAD}{nl}{body}{nl}{footer}"


def inspect(text):
    return inspect_pem_like(HEAD, text, 0)


class KeyStructure(unittest.TestCase):
    def test_complete_key_is_recognised_as_a_real_key(self):
        info = inspect(pem(ed25519_pkcs8(), wrap=64))
        self.assertEqual(info["verdict"], "complete_parseable_private_key")
        self.assertIn("Ed25519", info["format"]); self.assertTrue(info["der"]["complete"])
        self.assertIn("Ed25519", info["der"]["outline"])

    def test_truncated_key_stays_in_a_blocking_class(self):
        info = inspect(pem(ed25519_pkcs8()[:-10]))
        self.assertIn(info["verdict"], ("partial_private_key_structure", "unrecognized_der_structure"))
        self.assertFalse(info["der"]["complete"]); self.assertEqual(info["der"]["missing_bytes"], 10)

    def test_random_96_bytes_is_not_a_key_structure(self):
        raw = rbytes(96, 11)
        self.assertNotEqual(raw[0], 0x30)
        info = inspect(pem(raw))
        self.assertEqual((info["body_chars"], info["decoded_bytes"]), (128, 96))
        self.assertEqual(info["verdict"], "synthetic_non_key_body")
        self.assertFalse(info["der"]["top_level_asn1_sequence"]); self.assertFalse(info["openssh_magic"])

    def test_readable_text_and_non_base64(self):
        text = base64.b64encode(b"this is only a sentence pretending to be a private key body, nothing more").decode()
        info = inspect(pem(text))
        self.assertEqual(info["verdict"], "synthetic_non_key_body"); self.assertGreater(info["decoded_printable_ascii_fraction"], 0.95)
        info = inspect(pem("<your private key goes here, replace this text with the real one>"))
        self.assertEqual(info["verdict"], "synthetic_non_key_body"); self.assertEqual(info["base64"], "invalid or not base64")

    def test_openssh_container_is_not_called_synthetic(self):
        info = inspect(pem(b"openssh-key-v1\x00" + rbytes(200, 3)))
        self.assertEqual(info["verdict"], "partial_private_key_structure")

    def test_layout_reporting(self):
        der = ed25519_pkcs8()
        a = inspect(pem(der, wrap=64)); b = inspect(pem(der, wrap=64, escaped=True))
        self.assertEqual((a["real_newlines"] > 0, a["escaped_newlines"]), (True, 0))
        self.assertGreater(b["escaped_newlines"], 0); self.assertEqual(a["body_fingerprint"], b["body_fingerprint"])
        self.assertTrue(a["footer_label_matches_header"])
        c = inspect(pem(der, footer="-----END " + "RSA PRIVATE KEY-----"))
        self.assertTrue(c["footer_well_formed"]); self.assertFalse(c["footer_label_matches_header"])
        d = inspect(HEAD + "\n" + base64.b64encode(der).decode())
        self.assertFalse(d["footer_present"])

    def test_pem_headers_are_not_part_of_the_body(self):
        text = HEAD + "\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,00\n\n" + base64.b64encode(rbytes(64, 5)).decode() + "\n" + FOOT
        self.assertEqual(inspect(text)["pem_header_lines"], 2)


class InspectCommand(unittest.TestCase):
    def run_cmd(self, tmp, group):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            hb.cmd_inspect_credential(type("A", (), {"work_dir": tmp, "group": group})())
        return out.getvalue()

    def make(self, tmp, text):
        P = hb._paths(tmp)
        os.makedirs(P["main"])
        row = {"description": "see config", "reference_json": json.dumps({"tls": {"private_key": text}})}
        with open(os.path.join(P["main"], "create.jsonl"), "w") as f:
            f.write(json.dumps(row) + "\n")
        return hashlib.sha256(HEAD.encode()).hexdigest()[:8], P

    def test_output_has_metadata_but_no_value_and_scan_stays_blocking(self):
        raw = rbytes(96, 21)
        body = base64.b64encode(raw).decode()
        with tempfile.TemporaryDirectory() as tmp:
            group, P = self.make(tmp, pem(raw))
            out = self.run_cmd(tmp, group)
            self.assertIn("reference_json.tls.private_key", out); self.assertIn("synthetic_non_key_body", out)
            self.assertIn("UPLOAD REMAINS BLOCKED", out); self.assertIn("create.jsonl:1", out)
            for i in range(0, len(body) - 8, 4):  # no 9-character window of the encoded body appears in the output
                self.assertNotIn(body[i:i + 9], out)
            for i in range(0, len(raw.hex()) - 8, 4):
                self.assertNotIn(raw.hex()[i:i + 9], out)
            problems = hb._credential_problems(P["main"], verbose=False)  # the scan itself is unchanged: still blocked
            self.assertEqual(len(problems), 1); self.assertIn("needs manual review", problems[0])
            for i in range(0, len(body) - 8, 4):
                self.assertNotIn(body[i:i + 9], problems[0])

    def test_argument_validation_and_unknown_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            group, _ = self.make(tmp, pem(rbytes(96, 2)))
            for bad in ("", "xyz", "3021D90E", "3021d90"):
                with self.assertRaises(SystemExit):
                    self.run_cmd(tmp, bad)
            with self.assertRaises(SystemExit):
                self.run_cmd(tmp, "00000000")


def rsa_container():
    """A 1217-byte PKCS#8-shaped RSA-2048 container filled with synthetic random numbers (not a real key)."""
    mod = b"\x00" + rbytes(256, 31)
    inner_content = bytes([2, 1, 0]) + bytes([2, 0x82, 0x01, 0x01]) + mod + rbytes(923, 32)
    assert len(inner_content) == 1187
    inner = bytes([0x30, 0x82, 0x04, 0xA3]) + inner_content
    octet = bytes([0x04, 0x82, 0x04, 0xA7]) + inner
    alg = bytes([0x30, 0x0D, 0x06, 0x09, 0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x01, 0x05, 0x00])
    out = bytes([0x30, 0x82, 0x04, 0xBD]) + bytes([2, 1, 0]) + alg + octet
    assert len(out) == 1217
    return out


class RsaPrefix(unittest.TestCase):
    EL = "truncated_rsa_key_prefix_public_part_only"

    def test_prefix_cut_inside_the_modulus_has_no_private_components(self):
        info = inspect(pem(rsa_container()[:96]))
        self.assertEqual(info["verdict"], self.EL)
        r = info["rsa_prefix"]
        self.assertEqual((r["declared_total_bytes"], r["actual_bytes"], r["modulus_declared_bytes"]), (1217, 96, 257))
        self.assertEqual((r["modulus_bytes_present"], r["bytes_after_modulus"]), (59, 0))
        self.assertIn("PKCS#8", r["layout"])

    def test_going_past_the_modulus_or_the_whole_key_is_not_eligible(self):
        for cut, expect in ((400, "not"), (1217, "complete")):
            info = inspect(pem(rsa_container()[:cut]))
            self.assertNotEqual(info["verdict"], self.EL, cut)
            if expect == "complete":
                self.assertEqual(info["verdict"], "complete_parseable_private_key")
        self.assertGreater(inspect(pem(rsa_container()[:400]))["rsa_prefix"]["bytes_after_modulus"], 0)

    def test_other_shapes_are_not_eligible(self):
        self.assertNotEqual(inspect(pem(rsa_container()[:20]))["verdict"], self.EL)          # header itself cut
        self.assertNotEqual(inspect(pem(ed25519_pkcs8()[:30]))["verdict"], self.EL)          # not RSA
        self.assertNotEqual(inspect(pem(rbytes(96, 11)))["verdict"], self.EL)                # random bytes


class ReviewException(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.P = hb._paths(self.tmp)
        os.makedirs(self.P["main"])
        self.reviews = os.path.join(self.tmp, "credential_reviews.json")
        self.der = rsa_container()[:96]
        self.block = pem(self.der)
        self.body = base64.b64encode(self.der).decode()
        self.digest = hashlib.sha256(self.body.encode()).hexdigest()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rows):
        with open(os.path.join(self.P["main"], "create.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def row(self, block=None, field="private_key"):
        return {"description": "d", "reference_json": json.dumps({"tls": {field: block or self.block}})}

    def record(self, answers=None, digest=None, expect=1):
        from unittest import mock
        digest = digest or self.digest
        answers = answers if answers is not None else [digest[:16], "Tester"]
        args = type("A", (), {"work_dir": self.tmp, "body_sha256": digest, "expect_occurrences": expect,
                              "reviews_path": self.reviews})()
        with mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(io.StringIO()) as out:
            hb.cmd_record_credential_review(args)
        return out.getvalue()

    def problems(self):
        from unittest import mock
        with mock.patch.object(hb, "REVIEWS_PATH", self.reviews):
            return hb._credential_problems(self.P["main"], verbose=False)

    def manual(self):
        found = hb.credential_findings(self.P["main"], reviews=hb.load_reviews(self.reviews))
        return [x for x in found if x["classification"] not in ("placeholder", "reviewed_exception")]

    def no_value(self, text):
        for i in range(0, len(self.body) - 8, 4):
            self.assertNotIn(self.body[i:i + 9], text)

    def test_blocked_until_reviewed_then_allowed_only_in_exact_contexts(self):
        self.write([self.row()])
        self.assertEqual(len(self.problems()), 1)                    # blocked before the review
        out = self.record()
        self.assertIn(self.digest, out); self.no_value(out); self.no_value(open(self.reviews).read())
        saved = json.load(open(self.reviews))["reviews"][0]
        self.assertEqual((saved["body_sha256"], saved["reviewed_by"]), (self.digest, "Tester"))
        self.assertEqual(saved["contexts"], [{"file": "create.jsonl", "line": 1, "path": "reference_json.tls.private_key"}])
        self.assertEqual(self.problems(), [])                        # allowed: same value, same place
        self.write([{"description": "x"}, self.row()]); self.assertEqual(len(self.problems()), 1)   # other line
        self.write([self.row(field="other_key")]); self.assertEqual(len(self.problems()), 1)         # other field
        self.write([self.row(), self.row()]); self.assertEqual(len(self.problems()), 1)             # unexpected extra occurrence
        other = pem(self.body[:60] + ("A" if self.body[60] != "A" else "B") + self.body[61:])
        self.write([self.row(block=other)]); self.assertEqual(len(self.problems()), 1)              # changed value
        self.write([self.row(block=pem(rsa_container()))]); self.assertEqual(len(self.problems()), 1)  # complete key

    def test_several_bodies_under_one_header_only_the_chosen_digest_is_reviewed(self):
        """Regression: the header group holds placeholders and other bodies; the review must select by the body digest."""
        other_trunc = pem(rsa_container()[:80])                      # also eligible in structure, but NOT the reviewed one
        random_body = pem(rbytes(96, 41))
        header_only = HEAD                                            # an already-classified placeholder
        rows = [self.row(), self.row(block=other_trunc), self.row(block=random_body), self.row(block=header_only),
                self.row(field="copy_of_reviewed")]
        self.write(rows)
        self.assertEqual(len(self.manual()), 4)                      # chosen body x2, other truncated body, random body
        out = self.record(expect=2)                                  # the chosen body appears in two places
        self.no_value(out)
        self.assertIn("structural verdict synthetic_non_key_body", out)       # the others are listed by digest only
        self.assertEqual(len(self.manual()), 2)                      # only the other truncated body and the random body remain
        text = " ".join(self.problems())
        self.assertNotIn("copy_of_reviewed", text)
        self.assertIn("create.jsonl", text)
        saved = json.load(open(self.reviews))["reviews"][0]
        self.assertEqual({c["path"] for c in saved["contexts"]},
                         {"reference_json.tls.private_key", "reference_json.tls.copy_of_reviewed"})
        # the other eligible body is not covered by this review: it needs its own digest and its own review
        other_digest = hashlib.sha256(base64.b64encode(rsa_container()[:80]).decode().encode()).hexdigest()
        self.assertNotEqual(other_digest, self.digest)
        with self.assertRaises(SystemExit):
            self.record(digest="0" * 64, expect=1)                   # unknown digest: refused, table of digests printed

    def test_wrong_digest_prints_table_without_values(self):
        from unittest import mock
        self.write([self.row(), self.row(block=pem(rbytes(96, 41)))])
        err = io.StringIO()
        with mock.patch("builtins.input", side_effect=["x", "y"]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                self.record(digest="3bbf6cc3" + "456abcdeedda0d9c99463e8cc49eaee79d033d207b69af87f0276592")
        msg = str(cm.exception)
        self.assertIn(self.digest, msg); self.assertIn("occurrences 1", msg); self.no_value(msg)
        self.assertFalse(os.path.exists(self.reviews))

    def test_occurrence_count_must_match(self):
        self.write([self.row(), self.row(field="second")])
        for expect in (1, 3):
            with self.assertRaises(SystemExit):
                self.record(expect=expect)
        self.assertFalse(os.path.exists(self.reviews))
        self.record(expect=2)
        self.assertEqual(self.problems(), [])

    def test_ineligible_bodies_cannot_be_reviewed(self):
        for block in (pem(rbytes(96, 11)), pem(rsa_container()[:400]), pem(rsa_container())):
            self.write([self.row(block=block)])
            digest = hashlib.sha256(re.sub(r"\s", "", block.split("\n")[1]).encode()).hexdigest()
            with self.assertRaises(SystemExit):
                self.record(digest=digest, answers=["x", "y"])
            self.assertFalse(os.path.exists(self.reviews))

    def test_wrong_confirmation_writes_nothing(self):
        self.write([self.row()])
        with self.assertRaises(SystemExit):
            self.record(answers=["not the digest", "Tester"])
        from unittest import mock
        with mock.patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                hb.cmd_record_credential_review(type("A", (), {"work_dir": self.tmp, "body_sha256": self.digest,
                                                               "expect_occurrences": 1, "reviews_path": self.reviews})())
        self.assertFalse(os.path.exists(self.reviews))

    def test_digest_argument_must_be_the_full_digest(self):
        self.write([self.row()])
        for bad in ("", self.digest[:8], self.digest.upper(), self.digest[:-1], self.digest + "0"):
            args = type("A", (), {"work_dir": self.tmp, "body_sha256": bad, "expect_occurrences": 1, "reviews_path": self.reviews})()
            with self.assertRaises(SystemExit):
                hb.cmd_record_credential_review(args)
        self.assertFalse(os.path.exists(self.reviews))

    def test_inspect_filters_by_body_and_lists_digests(self):
        self.write([self.row(), self.row(block=pem(rbytes(96, 41)))])
        group = hashlib.sha256(HEAD.encode()).hexdigest()[:8]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            hb.cmd_inspect_credential(type("A", (), {"work_dir": self.tmp, "group": group, "body_sha256": self.digest})())
        text = out.getvalue()
        self.assertIn("1 occurrence(s)", text); self.assertIn(self.digest, text); self.no_value(text)

    def test_malformed_review_file_is_rejected(self):
        json.dump({"reviews": [{"body_sha256": "3bbf6cc3", "verdict": "truncated_rsa_key_prefix_public_part_only"}]},
                  open(self.reviews, "w"))
        with self.assertRaises(SystemExit):
            hb.load_reviews(self.reviews)


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests for the overnight chunk chain: sealed-parent discovery, Slurm dependency submission, failure,
preemption/requeue, duplicate successful attempts and restart behaviour (stubbed Slurm, real validation and seals)."""
import contextlib, io, json, os, re, shutil, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import chunked_run as cr
from data_generation import anchor as A
from data_generation import chunking
from test_chunking import ChunkBase, tree_hash

WRAPPER = os.path.join(ROOT, "slurm", "full_chunk_chain_job.sbatch")
SUBMITTER = os.path.join(ROOT, "slurm", "submit_chunk_chain.sh")
INNER = os.path.join(ROOT, "slurm", "full_chunk_granite_v2.sbatch")
TOTALS = {"create": 12, "fix": 12, "modify": 12}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class ChainBase(ChunkBase):
    """A throw-away checkout (src linked, stub chunk script) whose chunk directory is where ChunkBase builds its chunks."""

    def setUp(self):
        super().setUp()
        self.co = os.path.join(self.tmp.name, "checkout")
        os.makedirs(os.path.join(self.co, "slurm")); os.symlink(os.path.join(ROOT, "src"), os.path.join(self.co, "src"))
        pub = os.path.join(self.co, "data_published", "7962b0947f7d"); os.makedirs(pub)
        shutil.copy(self.reserved, os.path.join(pub, "reserved_schemas.json"))
        shutil.copy(self.anchor_file, os.path.join(pub, "published_train_schemas.json"))
        self.root = os.path.join(self.co, "data_scale_only", "full", "chunks"); os.makedirs(self.root)
        text = read(WRAPPER)
        for old, new in (("N_CHUNKS=8\n", "N_CHUNKS=3\n"), ('TOTALS="create=1334,fix=1333,modify=1333"', 'TOTALS="create=12,fix=12,modify=12"')):
            assert text.count(old) == 1; text = text.replace(old, new)
        self.write_exec("slurm/full_chunk_chain_job.sbatch", text)
        self.write_exec("slurm/full_chunk_granite_v2.sbatch",
                        '#!/bin/bash\necho "CHUNK_INDEX=$CHUNK_INDEX PARENT_CHUNK=${PARENT_CHUNK-unset} RESTART=${SLURM_RESTART_COUNT-unset} JOB=$SLURM_JOB_ID" '
                        '>> "$SLURM_SUBMIT_DIR/inner_calls.txt"\nexit "${INNER_RC:-0}"\n')
        self.bin = os.path.join(self.tmp.name, "bin"); os.makedirs(self.bin)
        self.write_exec(os.path.join("..", "bin", "git"), '#!/bin/bash\necho "${STUB_BRANCH:-improve-data-generation}"\n', base=self.co)

    def write_exec(self, rel, text, base=None):
        path = os.path.normpath(os.path.join(base or self.co, rel))
        with open(path, "w") as f:
            f.write(text)
        os.chmod(path, 0o700)

    # --- building chunks the way the jobs do
    def sealed(self, index, job, attempt, parent=None):
        return os.path.basename(self.run_chunk(index, f"c{index:02d}_{job}-r{attempt}", parent))

    def unsealed(self, index, job, attempt, parent=None):   # a finished build whose final validation failed: state exists, no marker
        return os.path.basename(self.run_chunk(index, f"c{index:02d}_{job}-r{attempt}", parent, seal=False))

    def preempted(self, index, job, attempt, parent=None):  # killed mid-chunk: no state file at all
        with self.assertRaises(BaseException):
            self.run_chunk(index, f"c{index:02d}_{job}-r{attempt}", parent, extra=("--max-llm-calls", "6"), seal=False)
        return f"c{index:02d}_{job}-r{attempt}"

    # --- running the chain job
    def chain_job(self, index, job, restart=0, **env):
        e = {**os.environ, "SLURM_JOB_ID": str(job), "SLURM_SUBMIT_DIR": self.co, "SLURM_RESTART_COUNT": str(restart),
             "CHUNK_INDEX": str(index), "BENCHSON_PYTHON": sys.executable, "PATH": self.bin + os.pathsep + os.environ["PATH"],
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONWARNINGS": "ignore"}
        e.update({k: str(v) for k, v in env.items()})
        for key in [k for k, v in env.items() if v is None]:
            e.pop(key, None)
        return subprocess.run(["bash", os.path.join(self.co, "slurm", "full_chunk_chain_job.sbatch")], cwd=self.co, env=e,
                              capture_output=True, text=True, timeout=170)

    def inner_calls(self):
        p = os.path.join(self.co, "inner_calls.txt")
        return read(p).splitlines() if os.path.exists(p) else []


class ResolveTest(ChainBase):
    def resolve(self, index, pin=None):
        return cr.resolve_chunk(self.root, index, self.anchor_file, self.reserved, self.CHUNKS, TOTALS, pin)

    def test_none_one_and_several_sealed_attempts(self):
        self.assertEqual(self.resolve(1)[0], cr.RESOLVE_NONE)
        a = self.sealed(1, 111, 0)
        self.assertEqual(self.resolve(1)[:2], (0, a))
        b = self.sealed(1, 111, 1)                                   # a second successful attempt of the same chunk
        code, name, valid, _ = self.resolve(1)
        self.assertEqual((code, name, valid), (cr.RESOLVE_AMBIGUOUS, None, [a, b]))   # never guessed
        self.assertEqual(self.resolve(1, pin=b)[:2], (0, b))         # only an explicit choice among valid ones
        self.assertEqual(self.resolve(1, pin="c01_999-r0")[0], cr.RESOLVE_NONE)

    def test_partial_unsealed_failed_and_tampered_attempts_are_never_selected(self):
        pre = self.preempted(1, 111, 0); uns = self.unsealed(1, 111, 1); good = self.sealed(1, 111, 2)
        tam = self.sealed(1, 111, 3)
        victim = os.path.join(self.root, tam, "benchmark_fix", "train"); f = sorted(os.listdir(victim))[0]
        p = os.path.join(victim, f); open(p, "w").write(read(p).replace("{", "{ ", 1))
        os.symlink(os.path.join(self.root, good), os.path.join(self.root, "c01_555-r0"))
        os.makedirs(os.path.join(self.root, "notes")); open(os.path.join(self.root, "c01_x.txt"), "w").close()
        before = tree_hash(self.root)
        code, name, valid, ignored = self.resolve(1)
        self.assertEqual((code, name, valid), (0, good, [good]))
        self.assertEqual({n for n, _ in ignored}, {pre, uns, tam, "c01_555-r0"})
        self.assertEqual(before, tree_hash(self.root))               # read-only

    def test_a_chunk_of_another_plan_is_ignored(self):
        self.sealed(1, 111, 0)
        self.assertEqual(cr.resolve_chunk(self.root, 1, self.anchor_file, self.reserved, 4, TOTALS)[0], cr.RESOLVE_NONE)
        self.assertEqual(cr.resolve_chunk(self.root, 1, self.anchor_file, self.reserved, 3, {"create": 12, "fix": 12, "modify": 13})[0],
                         cr.RESOLVE_NONE)

    def test_a_valid_chunk_with_a_broken_ancestor_is_not_valid(self):
        self.sealed(1, 111, 0, None); c2 = self.sealed(2, 222, 0, "c01_111-r0")
        self.assertEqual(self.resolve(2)[:2], (0, c2))
        os.remove(os.path.join(self.root, "c01_111-r0", chunking.SEAL_FILE))
        self.assertEqual(self.resolve(2)[0], cr.RESOLVE_NONE)

    def test_cli_prints_only_the_name_and_uses_distinct_exit_codes(self):
        a = self.sealed(1, 111, 0)
        argv = ["resolve", "--root", self.root, "--index", "1", "--anchor-file", self.anchor_file, "--reserved-file", self.reserved,
                "--chunks", "3", "--totals", "create=12,fix=12,modify=12"]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(cr.main(argv), 0)
        self.assertEqual(out.getvalue().strip(), a); self.assertEqual(json.loads(err.getvalue())["result"], a)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cr.main(argv[:4] + ["2"] + argv[5:]), cr.RESOLVE_NONE)


class ChainJobTest(ChainBase):
    def test_hands_over_the_exact_sealed_parent_among_preempted_attempts_and_ignores_leaked_env(self):
        self.preempted(1, 111, 0); good = self.sealed(1, 111, 1)
        before = tree_hash(self.root)
        r = self.chain_job(2, 222, PARENT_CHUNK="c01_111-r0")        # a stale PARENT_CHUNK from the submitting shell
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual(self.inner_calls(), [f"CHUNK_INDEX=2 PARENT_CHUNK={good} RESTART=0 JOB=222"])
        self.assertEqual(before, tree_hash(self.root))               # the link job writes nothing into the chunk directory

    def test_chunk_three_needs_the_validated_chain_c1_c2(self):
        self.sealed(1, 111, 0); c2 = self.sealed(2, 222, 0, "c01_111-r0")
        r = self.chain_job(3, 333)
        self.assertEqual(r.returncode, 0, r.stderr[-800:]); self.assertIn(f"PARENT_CHUNK={c2}", self.inner_calls()[0])

    def test_failed_or_unsealed_parent_stops_the_chain(self):
        self.unsealed(1, 111, 0)                                     # TECHNICAL_FAILURE after the state was written
        r = self.chain_job(2, 222)
        self.assertEqual(r.returncode, 10); self.assertIn("no fully valid, sealed chunk 1", r.stderr)
        self.assertEqual(self.inner_calls(), [])
        self.preempted(1, 111, 1)
        self.assertEqual(self.chain_job(2, 222).returncode, 10); self.assertEqual(self.inner_calls(), [])
        shutil.rmtree(self.root); os.makedirs(self.root)             # (test fixture only) nothing at all
        self.assertEqual(self.chain_job(2, 222).returncode, 10); self.assertEqual(self.inner_calls(), [])

    def test_duplicate_successful_attempts_need_an_explicit_validated_pin(self):
        a = self.sealed(1, 111, 0); b = self.sealed(1, 111, 1)
        r = self.chain_job(2, 222)
        self.assertEqual(r.returncode, 10); self.assertIn("PARENT_CHUNK_PIN", r.stderr); self.assertEqual(self.inner_calls(), [])
        r = self.chain_job(2, 222, PARENT_CHUNK_PIN=b)
        self.assertEqual(r.returncode, 0, r.stderr[-800:]); self.assertEqual(self.inner_calls(), [f"CHUNK_INDEX=2 PARENT_CHUNK={b} RESTART=0 JOB=222"])
        for bad in ("c01_999-r0", "c02_111-r0", "c01_111-r0; rm -rf x", "../c01_111-r0"):
            self.assertEqual(self.chain_job(2, 222, PARENT_CHUNK_PIN=bad).returncode, 10, bad)
        self.assertEqual(len(self.inner_calls()), 1)
        self.assertIsNotNone(a)

    def test_tampered_sealed_parent_is_refused_but_a_valid_sibling_is_used(self):
        a = self.sealed(1, 111, 0)
        victim = os.path.join(self.root, a, "benchmark_modify", "train"); f = sorted(os.listdir(victim))[0]
        p = os.path.join(victim, f); original = read(p); open(p, "w").write(original.replace("{", "{ ", 1))
        self.assertEqual(self.chain_job(2, 222).returncode, 10); self.assertEqual(self.inner_calls(), [])
        b = self.sealed(1, 111, 1)
        r = self.chain_job(2, 222)
        self.assertEqual(r.returncode, 0, r.stderr[-800:]); self.assertIn(f"PARENT_CHUNK={b}", self.inner_calls()[0])

    def test_requeue_keeps_attempts_separate_and_does_not_touch_earlier_ones(self):
        self.sealed(1, 111, 0)
        part = self.preempted(2, 222, 0, "c01_111-r0")               # attempt 0 of chunk 2 was preempted
        frozen = tree_hash(os.path.join(self.root, part))
        r = self.chain_job(2, 222, restart=1)                        # Slurm re-runs the same job with SLURM_RESTART_COUNT=1
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual(self.inner_calls(), ["CHUNK_INDEX=2 PARENT_CHUNK=c01_111-r0 RESTART=1 JOB=222"])
        self.assertEqual(frozen, tree_hash(os.path.join(self.root, part)))

    def test_a_job_preempted_after_sealing_does_not_rebuild_its_chunk(self):
        self.sealed(1, 111, 0); self.sealed(2, 222, 0, "c01_111-r0")
        before = tree_hash(self.root)
        r = self.chain_job(2, 222, restart=1)
        self.assertEqual(r.returncode, 0, r.stderr[-800:]); self.assertIn("already sealed by this job", r.stdout)
        self.assertEqual(self.inner_calls(), []); self.assertEqual(before, tree_hash(self.root))   # afterok is satisfied, no new directory

    def test_a_chunk_sealed_by_another_job_is_never_rebuilt(self):
        self.sealed(1, 111, 0); self.sealed(2, 333, 0, "c01_111-r0")
        r = self.chain_job(2, 222)
        self.assertEqual(r.returncode, 10); self.assertIn("another job", r.stderr); self.assertEqual(self.inner_calls(), [])

    def test_chunk_script_failure_is_passed_through_so_slurm_stops_the_chain(self):
        self.sealed(1, 111, 0)
        for rc in (2, 3, 4, 11):
            r = self.chain_job(2, 222, INNER_RC=rc)
            self.assertEqual(r.returncode, rc)                       # exec: the chunk script's own exit code, never masked

    def test_bad_arguments_and_environment_are_refused(self):
        self.sealed(1, 111, 0)
        for idx in ("1", "9", "x", ""):
            self.assertEqual(self.chain_job(idx, 222).returncode, 10, idx)
        self.assertEqual(self.chain_job(4, 222).returncode, 10)      # beyond the (test) 3-chunk plan
        self.assertEqual(self.chain_job(2, 222, STUB_BRANCH="main").returncode, 10)
        self.assertEqual(self.chain_job(2, "abc").returncode, 10)
        self.assertEqual(self.chain_job(2, 222, restart="-1").returncode, 10)
        self.assertEqual(self.chain_job(2, 222, SLURM_SUBMIT_DIR="").returncode, 10)
        self.assertEqual(self.inner_calls(), [])


class SubmitterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.co = os.path.join(self.tmp.name, "co"); os.makedirs(os.path.join(self.co, "slurm")); os.makedirs(os.path.join(self.tmp.name, "bin"))
        for name in ("submit_chunk_chain.sh", "full_chunk_chain_job.sbatch", "full_chunk_granite_v2.sbatch"):
            shutil.copy(os.path.join(ROOT, "slurm", name), os.path.join(self.co, "slurm", name))
        self.state = os.path.join(self.tmp.name, "state"); os.makedirs(self.state); open(os.path.join(self.state, "counter"), "w").write("5000")
        self.stub("git", '#!/bin/bash\necho "${STUB_BRANCH:-improve-data-generation}"\n')
        self.stub("sbatch", f'''#!/bin/bash
S="{self.state}"
echo "$*|PARENT_CHUNK=${{PARENT_CHUNK-unset}}|PIN=${{PARENT_CHUNK_PIN-unset}}" >> "$S/sbatch_calls"
n=$(wc -l < "$S/sbatch_calls")
[ "${{SBATCH_FAIL_AT:-0}}" = "$n" ] && {{ echo "sbatch: error" >&2; exit 1; }}
echo $((5000 + n))
''')
        self.stub("squeue", '''#!/bin/bash
case "$*" in
  *"-j "*) printf '%s' "${SQ_JOB_LINE-24044333|benchson-full|RUNNING}"; [ -z "${SQ_JOB_LINE-x}" ] || echo ;;
  *"-u "*) printf '%s\\n' "${SQ_NAMES:-}" ;;
esac
''')

    def stub(self, name, text):
        p = os.path.join(self.tmp.name, "bin", name); open(p, "w").write(text); os.chmod(p, 0o700)

    def run_sub(self, *args, **env):
        e = {**os.environ, "PATH": os.path.join(self.tmp.name, "bin") + os.pathsep + os.environ["PATH"], "USER": "tester"}
        e.update({k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", os.path.join(self.co, "slurm", "submit_chunk_chain.sh"), *args], cwd=self.tmp.name, env=e,
                              capture_output=True, text=True, timeout=60)

    def calls(self):
        p = os.path.join(self.state, "sbatch_calls")
        return read(p).splitlines() if os.path.exists(p) else []

    def test_dry_run_is_the_default_and_submits_nothing(self):
        r = self.run_sub("--after", "24044333")
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(self.calls(), [])
        self.assertIn("Dry run only", r.stdout); self.assertEqual(len(re.findall(r"^  chunk [2-8]:", r.stdout, re.M)), 7)

    def test_submits_chunks_two_to_eight_as_a_strict_afterok_chain(self):
        r = self.run_sub("--after", "24044333", "--submit", PARENT_CHUNK="c01_stale-r0", PARENT_CHUNK_PIN="c01_stale-r0")
        self.assertEqual(r.returncode, 0, r.stderr); calls = self.calls(); self.assertEqual(len(calls), 7)
        for i, call in enumerate(calls):
            k = i + 2; prev = "24044333" if i == 0 else str(5000 + i)
            self.assertIn(f"--dependency=afterok:{prev}", call); self.assertIn("--kill-on-invalid-dep=yes", call)
            self.assertIn(f"--job-name=benchson-full-c{k:02d}", call); self.assertIn(f"--export=ALL,CHUNK_INDEX={k}", call)
            self.assertIn("slurm/full_chunk_chain_job.sbatch", call); self.assertIn("|PARENT_CHUNK=unset|PIN=unset", call)   # no leaks
            self.assertEqual(call.count("--dependency"), 1)
        rec = read(os.path.join(self.co, "logs", "full-chain-24044333-c02-c08.jobs")); self.assertEqual(len(rec.splitlines()), 7)
        again = self.run_sub("--after", "24044333", "--submit")                       # never twice, never overwriting the record
        self.assertEqual(again.returncode, 10); self.assertEqual(len(self.calls()), 7); self.assertEqual(read(os.path.join(self.co, "logs", "full-chain-24044333-c02-c08.jobs")), rec)

    def test_previous_job_must_be_an_active_benchson_chunk_job(self):
        for line in ("", "24044333|other-job|RUNNING", "24044333|benchson-full|SUSPENDED", "24044333|benchson-full|FAILED", "999|benchson-full|RUNNING"):
            r = self.run_sub("--after", "24044333", "--submit", SQ_JOB_LINE=line)
            self.assertEqual(r.returncode, 10, line); self.assertEqual(self.calls(), [])
        for state in ("PENDING", "RUNNING", "REQUEUED", "COMPLETING", "CONFIGURING"):
            self.assertEqual(self.run_sub("--after", "24044333", SQ_JOB_LINE=f"24044333|benchson-full|{state}").returncode, 0, state)

    def test_duplicate_chunk_jobs_are_refused(self):
        r = self.run_sub("--after", "24044333", "--submit", SQ_NAMES="benchson-full-c05\nsomething")
        self.assertEqual(r.returncode, 10); self.assertIn("benchson-full-c05", r.stderr); self.assertEqual(self.calls(), [])
        self.assertEqual(self.run_sub("--after", "24044333", "--from", "2", "--to", "4", SQ_NAMES="benchson-full-c05").returncode, 0)

    def test_sbatch_failure_stops_the_chain_and_keeps_what_was_submitted(self):
        r = self.run_sub("--after", "24044333", "--submit", SBATCH_FAIL_AT=3)
        self.assertEqual(r.returncode, 11); self.assertEqual(len(self.calls()), 3)
        self.assertEqual(len(read(os.path.join(self.co, "logs", "full-chain-24044333-c02-c08.jobs")).splitlines()), 2)
        self.assertIn("chain stops", r.stderr)

    def test_continuing_without_a_dependency_after_a_manually_verified_chunk(self):
        r = self.run_sub("--after", "none", "--from", "4", "--to", "5", "--submit", SQ_JOB_LINE="")
        self.assertEqual(r.returncode, 0, r.stderr); calls = self.calls()
        self.assertNotIn("--dependency", calls[0]); self.assertIn("--dependency=afterok:5001", calls[1]); self.assertIn("CHUNK_INDEX=4", calls[0])

    def test_invalid_arguments_and_wrong_branch_are_refused(self):
        for args in (["--after", "abc"], [], ["--after", "1", "--from", "1"], ["--after", "1", "--to", "9"], ["--after", "1", "--from", "5", "--to", "3"],
                     ["--after", "1", "--bogus"], ["--after"], ["--after", "1", "--from"]):
            self.assertEqual(self.run_sub(*args, "--submit").returncode, 10, args)
        self.assertEqual(self.run_sub("--after", "24044333", "--submit", STUB_BRANCH="main").returncode, 10)
        self.assertEqual(self.calls(), [])


class ChainScriptStaticTest(unittest.TestCase):
    def setUp(self):
        self.wrapper, self.inner, self.submitter = read(WRAPPER), read(INNER), read(SUBMITTER)

    @staticmethod
    def directives(text):
        return [l for l in text.splitlines() if l.startswith("#SBATCH") and not l.startswith("#SBATCH --job-name")]

    def test_resources_and_limits_equal_the_chunk_script(self):
        self.assertEqual(self.directives(self.wrapper), self.directives(self.inner)); self.assertTrue(self.directives(self.wrapper))

    def test_plan_and_paths_equal_the_chunk_script_and_totals_are_exact(self):
        for pattern in (r'N_CHUNKS=(\d+)', r'TOTALS="([^"]+)"', r'PUBLISHED_REVISION=([0-9a-f]{40})', r'CHUNKS_ROOT="([^"]+)"',
                        r'BASELINE="([^"]+)"', r'ANCHOR="([^"]+)"'):
            w, i = re.search(pattern, self.wrapper), re.search(pattern, self.inner)
            if pattern.startswith("CHUNKS_ROOT"):
                self.assertEqual(w.group(1), "data_scale_only/full/chunks"); self.assertIn('CHUNKS_ROOT="$FULL_ROOT/chunks"', self.inner); continue
            self.assertEqual(w.group(1), i.group(1), pattern)
        totals = cr.parse_totals(re.search(r'TOTALS="([^"]+)"', self.wrapper).group(1))
        self.assertEqual(totals, {"create": 1334, "fix": 1333, "modify": 1333})
        rows = [chunking.increments(totals, 8, k) for k in range(1, 9)]
        self.assertEqual({t: sum(r[t] for r in rows) for t in totals}, totals)

    def test_wrapper_never_generates_deletes_or_picks_parents_by_guess(self):
        body = "\n".join(l for l in self.wrapper.splitlines() if not l.lstrip().startswith("#"))
        self.assertIn("exec bash \"$CHUNK_SCRIPT\"", body); self.assertIn("unset PARENT_CHUNK", body)
        for forbidden in (r"\brm\s", r"\brmdir\b", "scancel", r"\bmv\s", "ls -t", r"\bfind\s", r"(^|\s)sbatch\s", "vllm", "--max-model-len", "temperature", "max_tokens"):
            self.assertIsNone(re.search(forbidden, body), forbidden)
        self.assertIn("[2-8]", body)

    def test_submitter_is_a_dry_run_by_default_and_uses_afterok_with_invalid_dependency_kill(self):
        self.assertIn("DO_SUBMIT=0", self.submitter); self.assertIn("--kill-on-invalid-dep=yes", self.submitter)
        self.assertIn("afterok", self.submitter); self.assertNotIn("afterany", self.submitter); self.assertNotIn("scancel", self.submitter)
        self.assertIsNone(re.search(r"\brm\s", "\n".join(l for l in self.submitter.splitlines() if not l.lstrip().startswith("#"))))

    def test_chunk_one_script_is_unchanged_in_behaviour_by_the_chain(self):
        for needle in ("PARENT_CHUNK must be the directory name", "chunked_run.py seal", "SLURM_RESTART_COUNT"):
            self.assertIn(needle, self.inner)


if __name__ == "__main__":
    unittest.main()

import os, subprocess, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _have_main():
    try:
        return subprocess.run(["git", "rev-parse", "--verify", "main"], cwd=ROOT, capture_output=True,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).returncode == 0
    except OSError:
        return False


@unittest.skipUnless(_have_main(), "git ref 'main' not available")
class StandardEquivalenceTest(unittest.TestCase):
    def test_standard_mode_matches_original(self):
        import verify_standard_equivalence as v
        diffs, stats = v.compare("main", n=6)
        self.assertEqual(diffs, [])
        self.assertTrue(all(s["old"] > 0 and s["old"] == s["new"] for s in stats.values()))

    def test_legacy_default_is_not_original_modify(self):
        """Documents why the scale-only config needs the generation block."""
        import verify_standard_equivalence as v
        diffs, _ = v.compare("main", n=6, with_generation_block=False)
        self.assertTrue(any(d.startswith("modify") for d in diffs))


if __name__ == "__main__":
    unittest.main()

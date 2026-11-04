"""Checks for the standard/complex mix configuration and integer quota allocation.

Plain unittest. From the repo root:
    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data_generation.generation_mix import (GenerationSettings, allocate_mix, split_total,
                                            validate_ratios)


def modes(table):
    return {t: (m["standard"], m["complex"]) for t, m in table.items()}


class AllocationTests(unittest.TestCase):

    def test_smoke_test_four_examples_three_standard_one_complex(self):
        table = allocate_mix(split_total(4, ["create", "fix", "modify"]), 0.75, 0.25)
        self.assertEqual(sum(m["standard"] for m in table.values()), 3)
        self.assertEqual(sum(m["complex"] for m in table.values()), 1)

    def test_quotas_always_sum_to_the_requested_counts(self):
        for total in range(0, 40):
            for s, c in ((1, 0), (0, 1), (0.8, 0.2), (0.7, 0.3), (0.5, 0.5), (0.9, 0.1)):
                counts = split_total(total, ["create", "fix", "modify"])
                table = allocate_mix(counts, s, c)
                for task, n in counts.items():
                    self.assertEqual(table[task]["standard"] + table[task]["complex"], n)
                complex_total = sum(m["complex"] for m in table.values())
                # within one example of the ideal share
                self.assertLessEqual(abs(complex_total - total * c), 1 + 1e-9, (total, s, c))

    def test_pure_modes(self):
        counts = {"create": 3, "fix": 3, "modify": 3}
        self.assertEqual(modes(allocate_mix(counts, 1, 0)),
                         {"create": (3, 0), "fix": (3, 0), "modify": (3, 0)})
        self.assertEqual(modes(allocate_mix(counts, 0, 1)),
                         {"create": (0, 3), "fix": (0, 3), "modify": (0, 3)})

    def test_eighty_twenty_on_a_hundred_per_task(self):
        table = allocate_mix({"create": 100, "fix": 100, "modify": 100}, 0.8, 0.2)
        self.assertEqual(modes(table), {"create": (80, 20), "fix": (80, 20), "modify": (80, 20)})

    def test_seventy_thirty_and_uneven_task_sizes(self):
        table = allocate_mix({"create": 10, "fix": 20, "modify": 30}, 0.7, 0.3)
        self.assertEqual(sum(m["complex"] for m in table.values()), 18)
        self.assertEqual(table["create"]["complex"], 3)
        self.assertEqual(table["fix"]["complex"], 6)
        self.assertEqual(table["modify"]["complex"], 9)

    def test_single_task_and_zero_counts(self):
        self.assertEqual(modes(allocate_mix({"modify": 4}, 0, 1)), {"modify": (0, 4)})
        self.assertEqual(modes(allocate_mix({"create": 0, "fix": 0, "modify": 0}, 0.8, 0.2)),
                         {"create": (0, 0), "fix": (0, 0), "modify": (0, 0)})

    def test_allocation_is_deterministic(self):
        counts = {"create": 3, "fix": 3, "modify": 3}
        self.assertEqual(allocate_mix(counts, 0.8, 0.2), allocate_mix(counts, 0.8, 0.2))

    def test_split_total_gives_the_remainder_to_earlier_tasks(self):
        self.assertEqual(split_total(4, ["create", "fix", "modify"]), {"create": 2, "fix": 1, "modify": 1})
        self.assertEqual(split_total(5, ["fix", "modify"]), {"fix": 3, "modify": 2})
        self.assertEqual(split_total(3, ["modify"]), {"modify": 3})

    def test_ratios_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            validate_ratios(0.8, 0.3)
        with self.assertRaises(ValueError):
            validate_ratios(-0.2, 1.2)
        self.assertEqual(validate_ratios(0.8, 0.2), (Fraction(4, 5), Fraction(1, 5)))


class SettingsTests(unittest.TestCase):

    def test_no_generation_block_means_legacy(self):
        settings = GenerationSettings.from_config({"counts": {}})
        self.assertFalse(settings.enabled)
        self.assertEqual(modes(settings.mix({"create": 5, "fix": 5, "modify": 5})),
                         {"create": (5, 0), "fix": (5, 0), "modify": (5, 0)})
        self.assertEqual(settings.standard_modify_strategy, "recursive_weighted")

    def test_empty_block_defaults_to_eighty_twenty_with_original_standard_modify(self):
        settings = GenerationSettings.from_config({"generation": {}})
        self.assertTrue(settings.enabled)
        self.assertEqual((settings.standard_ratio, settings.complex_ratio), (Fraction(4, 5), Fraction(1, 5)))
        self.assertEqual(settings.standard_modify_strategy, "original")

    def test_one_ratio_derives_the_other(self):
        settings = GenerationSettings.from_config({"generation": {"complex_ratio": 0.3}})
        self.assertEqual(settings.standard_ratio, Fraction(7, 10))

    def test_complex_overrides_merge_with_defaults(self):
        settings = GenerationSettings.from_config(
            {"generation": {"complex": {"modify": {"max_operations": 4}}}})
        self.assertEqual(settings.complex["modify"]["min_operations"], 2)
        self.assertEqual(settings.complex["modify"]["max_operations"], 4)
        self.assertEqual(settings.complex["fix"]["max_errors"], 3)

    def test_invalid_settings_are_rejected(self):
        for bad in ({"standard_ratio": 0.5, "complex_ratio": 0.2},
                    {"standard_modify_strategy": "nope"},
                    {"complex": {"modify": {"min_operations": 3, "max_operations": 2}}}):
            with self.assertRaises(ValueError):
                GenerationSettings.from_config({"generation": bad})


if __name__ == "__main__":
    unittest.main()

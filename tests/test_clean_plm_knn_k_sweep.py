import sys
import unittest
from pathlib import Path


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import build_clean_plm_knn_k_sweep as sweep  # noqa: E402


def rows(values, auprs=None):
    auprs = auprs or values
    return [
        {"k": k, "F_max per-gene": fmax, "AUPR per-gene": aupr}
        for k, fmax, aupr in zip([3, 5, 10, 20], values, auprs)
    ]


class KValueTests(unittest.TestCase):
    def test_k_values_are_sorted_and_deduplicated(self):
        self.assertEqual(sweep.parse_k_values([10, 3, 10, 5]), [3, 5, 10])

    def test_k_values_must_be_positive(self):
        with self.assertRaises(ValueError):
            sweep.parse_k_values([0, 3])


class SweepClassificationTests(unittest.TestCase):
    def test_two_consecutive_declines_confirm_a_crash(self):
        result = sweep.classify_k_sweep(rows([0.4, 0.6, 0.5, 0.45]))
        self.assertEqual(result["best_k"], 5)
        self.assertEqual(result["first_decline_k"], 10)
        self.assertEqual(result["confirmed_crash_k"], 20)
        self.assertTrue(result["crash_confirmed"])

    def test_single_dip_then_recovery_is_not_confirmed(self):
        result = sweep.classify_k_sweep(rows([0.4, 0.6, 0.5, 0.6]))
        self.assertEqual(result["first_decline_k"], 10)
        self.assertIsNone(result["confirmed_crash_k"])
        self.assertFalse(result["crash_confirmed"])

    def test_first_peak_wins_exact_tie(self):
        result = sweep.classify_k_sweep(rows([0.4, 0.6, 0.6, 0.5]))
        self.assertEqual(result["best_k"], 5)
        self.assertTrue(result["rows"][1]["is_primary_peak"])
        self.assertFalse(result["rows"][2]["is_primary_peak"])

    def test_aupr_decline_is_exposed_as_a_guardrail(self):
        result = sweep.classify_k_sweep(rows([0.4, 0.6, 0.5, 0.45], [0.5, 0.4, 0.3, 0.2]))
        self.assertTrue(result["rows"][1]["aupr_below_observed_peak"])

    def test_smin_uses_minimization_for_the_degradation_horizon(self):
        metric_rows = [
            {"k": 3, "smin per-gene": 0.5},
            {"k": 5, "smin per-gene": 0.3},
            {"k": 10, "smin per-gene": 0.4},
            {"k": 20, "smin per-gene": 0.45},
        ]
        result = sweep.classify_metric_k_sweep(metric_rows, "smin per-gene")
        self.assertEqual(result["optimization"], "minimize")
        self.assertEqual(result["best_k"], 5)
        self.assertEqual(result["first_worse_k"], 10)
        self.assertEqual(result["confirmed_degradation_k"], 20)
        self.assertTrue(result["degradation_confirmed"])


if __name__ == "__main__":
    unittest.main()

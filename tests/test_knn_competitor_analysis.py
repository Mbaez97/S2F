import json
import sys
import unittest
from pathlib import Path

import numpy as np


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import build_knn_competitor_analysis as analysis  # noqa: E402


class ThresholdMetricTests(unittest.TestCase):
    def test_best_threshold_uses_most_conservative_tied_cutoff(self):
        result = analysis.evaluate_binary(
            np.asarray([1, 0, 1, 0]),
            np.asarray([0.9, 0.8, 0.7, 0.1]),
        )
        self.assertAlmostEqual(result["best_threshold"], 0.7)
        self.assertAlmostEqual(result["precision_at_best"], 2 / 3)
        self.assertAlmostEqual(result["recall_at_best"], 1.0)

    def test_degenerate_vector_matches_legacy_defaults(self):
        result = analysis.evaluate_binary(np.ones(3), np.asarray([0.5, 0.5, 0.5]))
        self.assertEqual(result["AUC"], 0.5)
        self.assertEqual(result["AUPR"], 1.0)
        self.assertEqual(result["F_max"], 0.0)
        self.assertTrue(result["degenerate"])

    def test_shared_per_gene_threshold_uses_macro_protein_scores(self):
        gold = np.asarray([[1, 0], [1, 1]])
        prediction = np.asarray([[0.9, 0.8], [0.7, 0.6]])
        result = analysis.evaluate_scope_at_threshold(
            gold, prediction, threshold=0.7, scope="per-gene"
        )
        self.assertAlmostEqual(result["precision"], 0.75)
        self.assertAlmostEqual(result["recall"], 0.75)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)


class CurrentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend_data = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
        cls.matrices, _inputs = analysis.load_method_matrices(
            cls.frontend_data, analysis.DEFAULT_ORGANISMS
        )

    def test_shared_matrix_shapes_and_positive_counts(self):
        ecoli = self.matrices[("83333", analysis.KNN10)]
        cyano = self.matrices[("1111708", analysis.KNN10)]
        self.assertEqual(ecoli.gold.shape, (160, 574))
        self.assertEqual(int(ecoli.gold.sum()), 1922)
        self.assertEqual(cyano.gold.shape, (33, 93))
        self.assertEqual(int(cyano.gold.sum()), 306)

    def test_ranking_metrics_reconstruct_and_legacy_smin_is_detected(self):
        rows, _thresholds, _units, comparable_difference, legacy_smin_difference = (
            analysis.build_metric_outputs(
                self.matrices,
                self.frontend_data / "competitor_context_metrics.csv",
                validate=True,
            )
        )
        self.assertLess(comparable_difference, 2e-6)
        self.assertGreater(legacy_smin_difference, 1.0)
        s2f_smin = next(
            row
            for row in rows
            if row["organism"] == "83333"
            and row["method"] == "S2F"
            and row["scope"] == "overall"
            and row["metric"] == "smin"
        )
        self.assertEqual(s2f_smin["saved_information_content_scope"], "full_organism_before_shared_filter")
        self.assertNotAlmostEqual(s2f_smin["value"], s2f_smin["harmonized_value"])

    def test_post_blacklist_neighbor_payload_has_no_selected_violations(self):
        _rows, _thresholds, units, _comparable, _legacy = analysis.build_metric_outputs(
            self.matrices,
            self.frontend_data / "competitor_context_metrics.csv",
            validate=True,
        )
        neighbor = analysis.build_neighbor_outputs(
            self.frontend_data,
            analysis.DEFAULT_ORGANISMS,
            self.matrices,
            units,
            blacklist_dir=None,
        )
        for row in neighbor["audit_rows"]:
            self.assertTrue(row["blacklist_filter_enabled"])
            self.assertTrue(row["blacklist_file_available"])
            self.assertEqual(row["selected_blacklist_violations"], 0)
            self.assertEqual(row["selected_benchmark_accession_violations"], 0)

    def test_frontend_index_uses_post_blacklist_diagnostics_only(self):
        payload = json.loads((self.frontend_data / "index.json").read_text())
        self.assertIn("knn_competitor_analysis", payload)
        self.assertNotIn("clean_plm_blacklist_comparison", payload)
        diagnostics = json.loads(
            (self.frontend_data / "knn_competitor_analysis.json").read_text()
        )
        self.assertFalse(diagnostics["metadata"]["pre_blacklist_results_used"])
        self.assertTrue(all(row["selected_blacklist_violations"] == 0 for row in diagnostics["neighbor_audit"]))


if __name__ == "__main__":
    unittest.main()

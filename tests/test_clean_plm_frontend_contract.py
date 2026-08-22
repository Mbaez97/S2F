import json
import sys
import unittest
from pathlib import Path

import pandas as pd


S2F_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
EXPORT_DIR = S2F_ROOT / "notebooks" / "exports" / "clean_plm_benchmark"
sys.path.insert(0, str(S2F_ROOT / "scripts"))

from clean_plm_method_registry import ACTIVE_KDE_MODEL, LEGACY_ADAPTIVE_KDE_MODEL  # noqa: E402


class CurrentFrontendKdeContractTests(unittest.TestCase):
    def test_method_comparison_replaces_knn_only_diagnostics(self):
        explorer_dir = S2F_ROOT / "notebooks" / "esm_go_explorer"
        html = (explorer_dir / "index.html").read_text()
        app = (explorer_dir / "app.js").read_text()

        for removed_text in (
            "GO-term Frequency",
            "Neighbor Evidence",
            "Example Protein",
            "Annotation frequency",
            "Comparison outcome",
            "Performance by annotation frequency",
            "Prediction agreement",
            "Score correlation",
            "Smin saved",
            "Smin harmonized",
            "Brier",
            "ECE-10",
        ):
            self.assertNotIn(removed_text, html)
        for removed_id in ("knn-frequency-table", "knn-neighbor-summary", "knn-example-table"):
            self.assertNotIn(removed_id, html)
            self.assertNotIn(removed_id, app)
        for removed_function in (
            "renderFrequencyTable",
            "renderNeighborSummary",
            "renderExampleTable",
            "openExampleNeighborhood",
            "openExampleDetails",
        ):
            self.assertNotIn(removed_function, app)

        self.assertIn("Method comparison", html)
        self.assertIn("renderMethodComparisonDiagnostics", app)
        for removed_id in (
            "method-comparison-protein",
            "method-comparison-term",
            "method-comparison-frequency",
            "method-comparison-outcome",
            "method-agreement-summary",
            "method-frequency-performance",
        ):
            self.assertNotIn(removed_id, html)
            self.assertNotIn(removed_id, app)
        for comparison_id in (
            "method-comparison-method-a",
            "method-comparison-method-b",
            "method-comparison-truth",
            "method-metric-chart",
            "method-score-summary",
            "method-score-scatter",
            "method-disagreement-table",
            "method-comparison-conclusion",
        ):
            self.assertIn(comparison_id, html)

    def test_method_comparison_artifact_covers_every_method_pair(self):
        payload = json.loads((FRONTEND_DATA / "method_comparison_analysis.json").read_text())
        index = json.loads((FRONTEND_DATA / "index.json").read_text())
        methods = [row["source_model"] for row in payload["metadata"]["methods"]]
        organisms = [str(value) for value in payload["metadata"]["organisms"]]

        self.assertEqual(payload["schema_version"], 2)
        self.assertFalse(payload["metadata"]["pre_blacklist_results_used"])
        self.assertEqual(len(methods), 9)
        self.assertEqual(set(organisms), {"83333", "1111708"})
        self.assertIn("S2F", methods)
        self.assertIn(ACTIVE_KDE_MODEL, methods)
        self.assertEqual(len(payload["calibration_rows"]), len(methods) * len(organisms))
        self.assertEqual(
            payload["metadata"]["smin_information_content_scope"],
            "full_organism_before_shared_evaluation_filter",
        )
        self.assertLess(
            payload["metadata"]["advanced_smin_validation_max_absolute_difference"],
            2e-6,
        )
        smin_rows = [row for row in payload["metric_rows"] if row["metric"] == "smin"]
        self.assertEqual(len(smin_rows), len(methods) * len(organisms) * 3)
        self.assertTrue(all("harmonized_value" not in row for row in smin_rows))
        self.assertTrue(all("saved_value" not in row for row in smin_rows))
        self.assertTrue(all(row["information_content_scope"] == payload["metadata"]["smin_information_content_scope"] for row in smin_rows))

        expected_pairs = len(methods) * (len(methods) - 1) // 2
        for organism in organisms:
            rows = [
                row for row in payload["pairwise_rows"]
                if str(row["organism"]) == organism
            ]
            self.assertEqual(len(rows), expected_pairs)
            self.assertTrue(all(row["method_a"] != row["method_b"] for row in rows))
            self.assertTrue(all(0 <= row["prediction_agreement_fraction"] <= 1 for row in rows))

        descriptor = index["method_comparison_analysis"]
        self.assertEqual(descriptor["schema_version"], 2)
        self.assertEqual(descriptor["path"], "data/method_comparison_analysis.json")

    def test_pfp_hides_shared_dataset_controls(self):
        app = (S2F_ROOT / "notebooks" / "esm_go_explorer" / "app.js").read_text()
        self.assertIn('["neighborhood", "pfp"].includes(page)', app)

    def test_active_kde_is_present_in_every_result_surface(self):
        clean_metrics = pd.read_csv(FRONTEND_DATA / "clean_plm_benchmark_metrics.csv")
        self.assertEqual(int((clean_metrics["model"] == ACTIVE_KDE_MODEL).sum()), 3)
        self.assertEqual(int((clean_metrics["model"] == LEGACY_ADAPTIVE_KDE_MODEL).sum()), 0)

        chart = pd.read_csv(FRONTEND_DATA / "competitor_context_metrics.csv")
        self.assertEqual(int((chart["model"] == ACTIVE_KDE_MODEL).sum()), 2)
        self.assertEqual(int((chart["model"] == LEGACY_ADAPTIVE_KDE_MODEL).sum()), 0)

        spreadsheet = json.loads((FRONTEND_DATA / "pfp_spreadsheet.json").read_text())
        spreadsheet_models = [str(row.get("method")) for row in spreadsheet["benchmark_rows"]]
        self.assertEqual(spreadsheet_models.count(ACTIVE_KDE_MODEL), 2)
        self.assertNotIn(LEGACY_ADAPTIVE_KDE_MODEL, spreadsheet_models)

        details = json.loads((FRONTEND_DATA / "pfp_prediction_details_index.json").read_text())
        detail_models = [str(row.get("model")) for row in details["details"]]
        self.assertEqual(detail_models.count(ACTIVE_KDE_MODEL), 2)
        self.assertNotIn(LEGACY_ADAPTIVE_KDE_MODEL, detail_models)

        score_comparison = json.loads((FRONTEND_DATA / "pfp_score_comparison_index.json").read_text())
        score_models = [str(row["source_model"]) for row in score_comparison["methods"]]
        self.assertIn(ACTIVE_KDE_MODEL, score_models)
        self.assertNotIn(LEGACY_ADAPTIVE_KDE_MODEL, score_models)

        diagnostics = json.loads((FRONTEND_DATA / "knn_competitor_analysis.json").read_text())
        diagnostics_text = json.dumps(diagnostics)
        self.assertIn(ACTIVE_KDE_MODEL, diagnostics_text)
        self.assertNotIn(LEGACY_ADAPTIVE_KDE_MODEL, diagnostics_text)

        neighbors = json.loads((FRONTEND_DATA / "plm_neighbor_embeddings.json").read_text())
        self.assertIn("kde", neighbors["methods"])
        self.assertNotIn("adaptive_kde", neighbors["methods"])
        self.assertTrue(neighbors["methods"]["kde"]["shared_across_organisms"])

    def test_kde_knn3_acceptance_artifact_passes(self):
        comparison = json.loads((EXPORT_DIR / "kde_vs_knn_k3_comparison.json").read_text())
        self.assertEqual(comparison["status"], "pass")
        self.assertTrue(comparison["all_organisms_have_distinct_predictions"])
        self.assertTrue(comparison["more_than_three_donors_observed"])
        self.assertTrue(
            all(not row["predictions_identical_within_1e_12"] for row in comparison["organisms"])
        )


if __name__ == "__main__":
    unittest.main()

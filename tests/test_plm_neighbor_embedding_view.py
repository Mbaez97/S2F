import sys
import unittest
from pathlib import Path

import numpy as np


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import build_plm_neighbor_embedding_view as view  # noqa: E402


class NeighborhoodSelectionTests(unittest.TestCase):
    def test_overlay_reproduces_knn_radius_and_shared_kde_selection(self):
        methods = view.method_neighbourhoods(
            indices=[0, 1, 2, 3, 4],
            similarities=[0.999, 0.995, 0.990, 0.980, 0.970],
            target_ids=["A", "B", "C", "D", "E"],
            kde_selection={
                "bandwidth": 0.1,
                "relative_weight_floor": 1e-6,
                "max_neighbors": 8192,
            },
        )
        self.assertEqual([row["protein_id"] for row in methods["knn_k_3"]], ["A", "B", "C"])
        self.assertEqual([row["protein_id"] for row in methods["knn_k_10"]], ["A", "B", "C", "D", "E"])
        self.assertEqual([row["protein_id"] for row in methods["radius_r_0_01"]], ["A", "B"])
        self.assertEqual([row["protein_id"] for row in methods["kde"]], ["A", "B", "C", "D", "E"])
        self.assertAlmostEqual(methods["kde"][0]["weight"], 1.0)
        self.assertGreater(methods["kde"][3]["weight"], 0.0)

    def test_payload_validation_rejects_test_donor_overlap(self):
        payload = {
            "points": [
                {"key": "test:1:P1", "protein_id": "P1", "dataset_source": "benchmark_test"},
                {"key": "swissprot:P1", "protein_id": "P1", "dataset_source": "swissprot_donor"},
            ],
            "projections": {
                key: {"coordinates": np.zeros((2, 3)).tolist()}
                for key in view.PROJECTION_KEYS
            },
            "neighborhoods": {},
        }
        with self.assertRaises(AssertionError):
            view.validate_payload(payload)


if __name__ == "__main__":
    unittest.main()

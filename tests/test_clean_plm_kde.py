import argparse
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import clean_plm_benchmark as bench  # noqa: E402
import plm  # noqa: E402


class KdeGeometryTests(unittest.TestCase):
    def test_effective_radius_matches_weight_floor(self):
        bandwidth = 0.05
        floor = 0.01
        radius = bench.kde_effective_cosine_radius(bandwidth, floor)
        weight = math.exp(-radius / (bandwidth * bandwidth))
        self.assertAlmostEqual(weight, floor, places=12)

    def test_gaussian_weights_decrease_with_cosine_distance(self):
        weights = bench.gaussian_kde_weights([1.0, 0.99, 0.98], bandwidth=0.05)
        self.assertAlmostEqual(float(weights[0]), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_relative_gaussian_weights_preserve_normalized_support(self):
        similarities = [0.97, 0.95, 0.91, 0.88]
        absolute = bench.gaussian_kde_weights(similarities, bandwidth=0.08)
        relative = bench.relative_gaussian_kde_weights(similarities, bandwidth=0.08)
        absolute /= absolute.sum()
        relative /= relative.sum()
        np.testing.assert_allclose(relative, absolute, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(relative.sum()), 1.0)

    def test_shared_kde_can_retain_more_than_three_donors(self):
        indices, _scores, weights, diagnostics = bench.select_kde_neighbors(
            [[0, 1, 2, 3, 4, 5]],
            [[0.999, 0.998, 0.997, 0.996, 0.995, 0.994]],
            bandwidth=0.1,
            weight_floor=1e-6,
        )
        self.assertEqual(indices, [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(len(weights[0]), 6)
        self.assertEqual(diagnostics[0]["retained_neighbor_count"], 6)

    def test_bandwidth_candidates_depend_on_distance_gaps_not_neighbor_rank(self):
        first = bench.derive_kde_bandwidth_candidates(
            [[0.99, 0.98, 0.97, 0.96]], 1e-6
        )
        shifted = bench.derive_kde_bandwidth_candidates(
            [[0.89, 0.88, 0.87, 0.86]], 1e-6
        )
        np.testing.assert_allclose(first, shifted, rtol=1e-12, atol=1e-12)

    def test_kde_selection_respects_kernel_contour(self):
        bandwidth = 0.05
        floor = 0.01
        radius = bench.kde_effective_cosine_radius(bandwidth, floor)
        similarities = [[1.0, 1.0 - radius * 0.5, 1.0 - radius * 1.1]]
        indices, scores, weights, diagnostics = bench.select_kde_neighbors(
            [[4, 5, 6]], similarities, bandwidth, floor
        )
        self.assertEqual(indices, [[4, 5]])
        self.assertEqual(len(scores[0]), 2)
        self.assertEqual(len(weights[0]), 2)
        self.assertEqual(diagnostics[0]["retained_neighbor_count"], 2)
        self.assertGreaterEqual(min(weights[0]), floor)

    def test_adaptive_rank_is_on_relative_kernel_contour(self):
        scores = [0.995, 0.990, 0.985]
        relative_weights, _bandwidth, _d1, _dk, _radius = bench.adaptive_relative_gaussian_weights(
            scores,
            neighbor_rank=3,
            relative_weight_floor=0.01,
            bandwidth_scale=1.0,
        )
        self.assertAlmostEqual(float(relative_weights[0]), 1.0)
        self.assertAlmostEqual(float(relative_weights[2]), 0.01, places=10)

    def test_adaptive_relative_floor_and_minimum_fallback(self):
        selected = bench.select_adaptive_kde_neighbors(
            [[0, 1, 2, 3, 4]],
            [[0.995, 0.990, 0.985, 0.970, 0.950]],
            neighbor_rank=5,
            bandwidth_scale=0.25,
            relative_weight_floor=0.01,
            min_neighbors=3,
            confidence_distance_scale=0.01,
        )
        indices, _scores, weights, confidences, diagnostics = selected
        self.assertEqual(len(indices[0]), 3)
        self.assertTrue(diagnostics[0]["minimum_neighbor_fallback_applied"])
        self.assertAlmostEqual(weights[0][0], 1.0)
        self.assertGreater(confidences[0], 0.0)

    def test_density_confidence_penalizes_larger_nearest_distance(self):
        dense = bench.select_adaptive_kde_neighbors(
            [[0, 1, 2]], [[0.999, 0.994, 0.989]], 3, 1.0, 0.01, 3, 0.05
        )
        sparse = bench.select_adaptive_kde_neighbors(
            [[0, 1, 2]], [[0.990, 0.985, 0.980]], 3, 1.0, 0.01, 3, 0.05
        )
        self.assertGreater(dense[3][0], sparse[3][0])


class KdeGoSupportTests(unittest.TestCase):
    def test_kernel_weights_replace_similarity_only_for_weighted_support(self):
        rows = bench.direct_rows_for_neighbors(
            ["query"],
            ["source_a", "source_b"],
            [[0, 1]],
            [[0.99, 0.98]],
            {
                "source_a": {"GO:0000001"},
                "source_b": {"GO:0000001", "GO:0000002"},
            },
            "weighted_support",
            neighbor_weights=[[1.0, 0.25]],
        )
        scores = {row["GO ID"]: row["Score"] for row in rows}
        self.assertAlmostEqual(scores["GO:0000001"], 1.0)
        self.assertAlmostEqual(scores["GO:0000002"], 0.2)

        confidence_rows = bench.direct_rows_for_neighbors(
            ["query"],
            ["source_a"],
            [[0]],
            [[0.99]],
            {"source_a": {"GO:0000001"}},
            "weighted_support",
            query_confidences=[0.25],
        )
        self.assertAlmostEqual(confidence_rows[0]["Score"], 0.25)

        max_rows = bench.direct_rows_for_neighbors(
            ["query"],
            ["source_a", "source_b"],
            [[0, 1]],
            [[0.99, 0.98]],
            {"source_a": {"GO:0000001"}, "source_b": {"GO:0000001"}},
            "max_similarity",
            neighbor_weights=[[1.0, 0.25]],
        )
        self.assertAlmostEqual(max_rows[0]["Score"], 0.99)


class KdeCalibrationConfigurationTests(unittest.TestCase):
    def test_calibration_sampling_is_deterministic_and_excludes_benchmark_ids(self):
        target_ids = [f"P{i}" for i in range(20)]
        annotations = {protein: {"GO:0000001"} for protein in target_ids}
        first = bench.select_kde_calibration_ids(
            annotations, target_ids, {"P0", "P1"}, calibration_size=5, random_seed=17
        )
        second = bench.select_kde_calibration_ids(
            annotations, target_ids, {"P0", "P1"}, calibration_size=5, random_seed=17
        )
        self.assertEqual(first, second)
        self.assertTrue(set(first).isdisjoint({"P0", "P1"}))

    def test_explicit_bandwidths_are_sorted_and_deduplicated(self):
        selected = bench.derive_kde_bandwidth_candidates(
            [[0.99]], 0.01, explicit_bandwidths=[0.2, 0.1, 0.2]
        )
        self.assertEqual(selected, [0.1, 0.2])

    def test_default_strategies_and_radius_backward_compatibility(self):
        args = argparse.Namespace(strategies=None, radius=None, radii=None)
        self.assertEqual(bench.selected_strategies(args), ["knn", "kde"])
        args.radius = 0.01
        self.assertEqual(bench.selected_strategies(args), ["knn", "radius", "kde"])

    def test_active_kde_model_name_is_shared_gaussian_support(self):
        self.assertEqual(
            bench.clean_plm_model_name("kde", "kernel_support"),
            bench.ACTIVE_KDE_MODEL,
        )
        self.assertNotEqual(bench.ACTIVE_KDE_MODEL, bench.LEGACY_ADAPTIVE_KDE_MODEL)


class BlacklistFilteringTests(unittest.TestCase):
    def test_portable_bundle_data_root_is_discovered_next_to_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_root = Path(directory)
            repository_root = bundle_root / "repo"
            repository_root.mkdir()
            goa_path = bundle_root / "data" / "uniprot" / "filtered_goa"
            goa_path.parent.mkdir(parents=True)
            goa_path.write_text("!gaf-version: 2.2\n", encoding="utf-8")
            self.assertEqual(bench.resolve_bundle_root(repository_root), bundle_root.resolve())

    def test_blacklisted_source_taxa_are_excluded_per_test_organism_before_search(self):
        target_cache = SimpleNamespace(
            ids=["P_ECOLI", "P_PSEUDO", "P_CYANO", "P_OTHER"],
            headers=[
                "sp|P_ECOLI|X OS=E. coli OX=83333",
                "sp|P_PSEUDO|X OS=P. syringae OX=223283",
                "sp|P_CYANO|X OS=Synechocystis OX=1111708",
                "sp|P_OTHER|X OS=Other OX=999999",
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blacklists = {}
            paths = {}
            for organism in bench.ORGANISMS:
                path = root / f"{organism}.blacklist"
                path.write_text(f"{organism}\n", encoding="utf-8")
                blacklists[organism] = {organism}
                paths[organism] = path
            exclusions, audit = bench.build_source_exclusions_by_organism(
                target_cache,
                {"P_OTHER"},
                blacklists,
                paths,
            )
        self.assertEqual(exclusions["83333"], {"P_ECOLI", "P_OTHER"})
        self.assertEqual(exclusions["223283"], {"P_PSEUDO", "P_OTHER"})
        self.assertEqual(exclusions["1111708"], {"P_CYANO", "P_OTHER"})
        self.assertTrue(audit["blacklist_filter_enabled"].all())

    def test_plm_blacklist_removes_donor_before_knn_rank_assignment(self):
        target_cache = plm.EmbeddingCache(
            ids=["P_BLOCKED", "P_ALLOWED"],
            headers=[
                "sp|P_BLOCKED|X OS=Blocked OX=1",
                "sp|P_ALLOWED|X OS=Allowed OX=2",
            ],
            lengths=[1, 1],
            embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            path=Path("."),
            metadata={},
        )
        excluded = plm.blacklisted_target_indices(target_cache, {"1"})
        indices, scores = plm.compute_knn(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            target_cache.embeddings,
            k=1,
            query_chunk_size=1,
            excluded_target_indices=excluded,
        )
        self.assertEqual(indices.tolist(), [[1]])
        self.assertAlmostEqual(float(scores[0, 0]), 0.0)

    def test_blacklist_comparison_normalizes_organism_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            comparison_dir = output_dir / "blacklist_comparison"
            comparison_dir.mkdir()
            pd.DataFrame(
                [{"organism": 83333, "model": "Clean PLM + KNN", "overall::F_max": 0.5}]
            ).to_csv(comparison_dir / "pre_blacklist_metrics.csv", index=False)
            filtered = pd.DataFrame(
                [{"organism": "83333", "model": "Clean PLM + KNN", "overall::F_max": 0.4}]
            )
            audit = pd.DataFrame(
                [{"organism": 83333, "blacklisted_source_protein_count": 12}]
            )
            original_frontend_data = bench.FRONTEND_DATA
            bench.FRONTEND_DATA = output_dir / "frontend"
            try:
                comparison = bench.write_blacklist_comparison(output_dir, audit, filtered)
            finally:
                bench.FRONTEND_DATA = original_frontend_data
        self.assertEqual(comparison.loc[0, "organism"], "83333")
        self.assertAlmostEqual(comparison.loc[0, "delta::overall::F_max"], -0.1)


if __name__ == "__main__":
    unittest.main()

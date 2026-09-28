import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT))
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import plm  # noqa: E402
import clean_plm_benchmark as clean  # noqa: E402
from commands.Predict import Predict  # noqa: E402


def cache(ids, embeddings):
    return plm.EmbeddingCache(
        ids=list(ids),
        headers=[f"{protein} OX=1" for protein in ids],
        lengths=[10] * len(ids),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        path=Path("."),
        metadata={},
    )


class PlmTransferTests(unittest.TestCase):
    def test_propagated_assignments_are_deduplicated_and_written(self):
        class FakeGeneOntology:
            def __init__(self):
                self.annotations = None

            def load_annotations(self, annotations, _source):
                self.annotations = annotations.copy()

            def up_propagate_annotations(self, _source):
                pass

            def get_annotations(self, _source):
                return self.annotations

        direct_rows = [
            {"Protein": "P1", "GO ID": "GO:1", "Score": 0.4},
            {"Protein": "P1", "GO ID": "GO:1", "Score": 0.8},
            {"Protein": "P2", "GO ID": "GO:2", "Score": 0.5},
        ]
        with tempfile.TemporaryDirectory() as temp_name:
            output_path = Path(temp_name) / "assignments.tsv"
            plm.write_propagated_assignments(
                FakeGeneOntology(), direct_rows, output_path
            )
            written = pd.read_csv(output_path, sep="\t")
        self.assertEqual(2, len(written))
        self.assertEqual(0.8, written.loc[written["GO ID"] == "GO:1", "Score"].item())

    def test_embedding_cache_accepts_media_mount_alias(self):
        cached = {
            "fasta": [{"path": "/media/disk/input.fasta", "size": 10, "mtime_ns": 20}],
            "model_name": "model",
        }
        expected = {
            "fasta": [{"path": "/run/media/disk/input.fasta", "size": 10, "mtime_ns": 20}],
            "model_name": "model",
        }
        self.assertTrue(plm.cache_matches(cached, expected))

    def test_weighted_knn_matches_clean_baseline(self):
        query = cache(["Q"], [[1.0, 0.0]])
        target = cache(["A", "B", "C"], [[1, 0], [0.8, 0.6], [0, 1]])
        indices = np.asarray([[0, 1, 2]])
        scores = np.asarray([[1.0, 0.8, 0.0]])
        terms = {"A": {"GO:1"}, "B": {"GO:1", "GO:2"}, "C": {"GO:2"}}
        with tempfile.TemporaryDirectory() as temp_name:
            actual = plm.build_direct_assignments(
                query, target, indices, scores, terms, Path(temp_name),
                score_mode="weighted_support",
            )
        expected = clean.direct_rows_for_neighbors(
            query.ids, target.ids, indices.tolist(), scores.tolist(), terms,
            "weighted_support",
        )
        actual_by_term = {row["GO ID"]: row["Score"] for row in actual}
        expected_by_term = {row["GO ID"]: row["Score"] for row in expected}
        self.assertEqual(set(actual_by_term), set(expected_by_term))
        for term in actual_by_term:
            self.assertAlmostEqual(actual_by_term[term], expected_by_term[term])

    def test_kde_uses_data_dependent_support_and_continuous_weights(self):
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        target = np.asarray(
            [[1.0, 0.0], [0.999, 0.0447], [0.995, 0.0999], [0.0, 1.0]],
            dtype=np.float32,
        )
        indices, scores, weights, diagnostics = plm.compute_kde_neighbors(
            query, target, bandwidth=0.1, weight_floor=1e-3,
            max_neighbors=4, query_chunk_size=1,
        )
        self.assertGreater(len(indices[0]), 1)
        self.assertLess(len(indices[0]), 4)
        self.assertEqual(indices[0], sorted(indices[0], key=lambda i: -scores[0][indices[0].index(i)]))
        self.assertAlmostEqual(weights[0][0], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(weights[0], weights[0][1:])))
        self.assertFalse(diagnostics[0]["neighbor_cap_applied"])

    def test_kde_fails_instead_of_truncating_at_cap(self):
        with self.assertRaisesRegex(
            RuntimeError, "maximum is 3 contributors at query row 0"
        ):
            plm.compute_kde_neighbors(
                np.asarray([[1.0, 0.0]]),
                np.asarray([[1.0, 0.0], [0.999, 0.01], [0.998, 0.02]]),
                bandwidth=1.0, weight_floor=1e-6,
                max_neighbors=2, query_chunk_size=1,
            )

    def test_accession_and_taxon_exclusions_happen_before_ranking(self):
        target = cache(["A", "B", "C"], [[1, 0], [0.9, 0.1], [0, 1]])
        target.headers[:] = ["A OX=1", "B OX=2", "C OX=3"]
        taxon = plm.blacklisted_target_indices(target, {"1"})
        accessions = plm.excluded_accession_indices(target, {"B"})
        indices, _scores = plm.compute_knn(
            np.asarray([[1.0, 0.0]]), target.embeddings, 1, 1,
            excluded_target_indices=np.concatenate((taxon, accessions)),
        )
        self.assertEqual(2, int(indices[0, 0]))

    def test_benchmark_csv_and_line_exclusion_files_are_supported(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            csv_path = root / "ids.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handler:
                writer = csv.DictWriter(handler, fieldnames=["organism", "protein_id"])
                writer.writeheader()
                writer.writerow({"organism": "1", "protein_id": "A"})
                writer.writerow({"organism": "2", "protein_id": "B"})
            self.assertEqual({"A", "B"}, plm.read_accession_exclusions(str(csv_path)))
            line_path = root / "ids.txt"
            line_path.write_text("A\nB\n", encoding="utf-8")
            self.assertEqual({"A", "B"}, plm.read_accession_exclusions(str(line_path)))


class PlmOnlyOrchestrationTests(unittest.TestCase):
    def test_plm_only_progress_never_loads_interpro_hmmer_or_foldseek(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            predictor = Predict.__new__(Predict)
            predictor.alias = "fixture_plm_only"
            predictor.seed_dir_IP = str(root / "ip")
            predictor.seed_dir_H = str(root / "hmmer")
            predictor.seed_dir_F = str(root / "foldseek")
            predictor.seed_dir_P = str(root / "plm")
            predictor.output_dir = str(root / "output")
            predictor.installation_directory = str(root)
            predictor.combination_dir = str(root / "combined")
            predictor.use_ip_seed = False
            predictor.use_hmmer_seed = False
            predictor.use_foldseek_seed = False
            predictor.use_plm_seed = True
            predictor.write_diffusion_tsv = False
            predictor.combined_graph = "compute"
            predictor.check_progress()
            self.assertFalse(predictor.load_ip_seed)
            self.assertFalse(predictor.load_hmmer_seed)
            self.assertFalse(predictor.load_foldseek_seed)
            self.assertTrue(predictor.load_plm_seed)
            self.assertTrue(predictor.calculate_diffusion)
            self.assertTrue(predictor.load_graph_collection)
            self.assertTrue(predictor.load_homology_graph)

    def test_combined_graph_signature_changes_with_seed_scores(self):
        predictor = Predict.__new__(Predict)
        predictor.proteins = pd.DataFrame(
            {"protein idx": [0, 1]}, index=["P1", "P2"]
        )
        first = sparse.coo_matrix(([0.5], ([0], [0])), shape=(2, 2))
        second = sparse.coo_matrix(([0.6], ([0], [0])), shape=(2, 2))
        self.assertNotEqual(
            predictor._combined_graph_signature(first),
            predictor._combined_graph_signature(second),
        )


if __name__ == "__main__":
    unittest.main()

import gzip
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT))

from commands.Predict import Predict
from diffusion import Diffusion
from graphs.collection import Collection
from graphs.homology import vectorized_homology_graph


class FastCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.orthologs = self.root / "orthologs"
        self.orthologs.mkdir()
        self.string_dir = self.root / "string"
        self.string_dir.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.fasta = self.root / "target.faa"
        self.fasta.write_text(">Q1\nAAAA\n>Q2\nAAAA\n>Q3\nAAAA\n")
        (self.root / "target.faa.phr").touch()
        (self.root / "target.faa.pin").touch()
        self.core = self.root / "coreIds"
        self.core.write_text("23\n")
        self.links = self.root / "links.txt.gz"
        with gzip.open(self.links, "wt") as handler:
            handler.write(
                "protein1 protein2 neighborhood neighborhood_transferred "
                "fusion cooccurence homology coexpression "
                "coexpression_transferred experiments experiments_transferred "
                "database database_transferred textmining "
                "textmining_transferred combined_score\n"
                "23.A 23.B 10 0 0 0 0 20 0 30 0 40 0 50 0 100\n"
                "23.A 23.C 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
                "23.A 23.X 99 0 0 0 0 0 0 0 0 0 0 0 0 99\n"
                "24.A 24.B 99 0 0 0 0 0 0 0 0 0 0 0 0 99\n"
            )
        orthologs = pd.DataFrame({
            "query": ["Q1", "Q2", "Q3"],
            "target": ["23.A", "23.B", "23.C"],
            "query_evalue": [1e-20, 1e-30, 1e-40],
            "query_pi": [90.0, 91.0, 92.0],
            "query_perc": [90.0, 91.0, 92.0],
            "target_evalue": [1e-25, 1e-35, 1e-45],
            "target_pi": [90.0, 91.0, 92.0],
            "target_perc": [90.0, 91.0, 92.0],
            "pos": [90.0, 91.0, 92.0],
        })
        orthologs.to_pickle(self.orthologs / "fixture_AND_23")
        self.proteins = pd.DataFrame(
            {"protein idx": [0, 1, 2]}, index=["Q1", "Q2", "Q3"])

    def tearDown(self):
        self.temp.cleanup()

    def build(self, engine):
        graph_dir = self.root / ("graphs-" + engine)
        graph_dir.mkdir()
        collection = Collection(
            str(self.fasta), self.proteins, str(self.string_dir),
            str(self.links), str(self.core), str(self.output),
            str(self.orthologs), str(graph_dir), "fixture", 1, None,
            1e-6, 80.0, 60.0, "entire_id", False,
            recompute_orthologs=False, chunk_size=2, engine=engine)
        collection.compute_graph()
        return collection.collection

    @unittest.skipUnless(shutil.which("pigz") and shutil.which("grep"),
                         "native filter tools are unavailable")
    def test_native_filter_matches_legacy_transfer(self):
        legacy = self.build("legacy")
        native = self.build("native_filter")
        columns = ["query1", "query2", "max_evalue", "neighborhood",
                   "experiments", "coexpression", "textmining", "database"]
        legacy = legacy[columns].sort_values(columns[:2]).reset_index(drop=True)
        native = native[columns].sort_values(columns[:2]).reset_index(drop=True)
        pd.testing.assert_frame_equal(legacy, native, check_dtype=False)


class VectorizedHomologyTests(unittest.TestCase):
    def test_vectorized_weights_match_legacy_loops(self):
        homology = {
            "P1": {"P1": 0.0, "P2": 1e-20},
            "P2": {"P2": 0.0, "P1": 1e-10, "P3": 0.0},
            "P3": {"P3": 0.0},
        }
        proteins, actual = vectorized_homology_graph(homology)
        expected = np.zeros_like(actual)
        maxi = -1.0
        for i, p1 in enumerate(proteins):
            for j in range(i, len(proteins)):
                p2 = proteins[j]
                e12 = homology[p1].get(p2, 10.0)
                e21 = homology[p2].get(p1, 10.0)
                value = max(e12, e21)
                transformed = 0.0
                if value == 0 or p1 == p2:
                    transformed = 1.0
                elif 0 < value < 11:
                    transformed = -np.log(value / 11.0)
                    maxi = max(maxi, transformed)
                expected[i, j] = transformed
                expected[j, i] = transformed
        expected[expected != 1] *= 1.0 / maxi
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class ChunkedPredictionWriterTests(unittest.TestCase):
    def test_chunked_prediction_matches_legacy_output(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            proteins = pd.DataFrame(
                {"protein idx": [0, 1]}, index=["P1", "P2"])
            proteins.index.name = "protein id"
            terms = pd.DataFrame(
                {"term idx": [0, 1]}, index=["GO:1", "GO:2"])
            terms.index.name = "term id"
            matrix = sparse.coo_matrix(
                ([0.2, 0.8, 0.5], ([1, 0, 1], [0, 1, 1])), shape=(2, 2))
            legacy = root / "legacy.tsv"
            chunked = root / "chunked.tsv"
            Diffusion._write_results(matrix, proteins, terms, legacy)

            predictor = Predict.__new__(Predict)
            predictor.prediction = matrix
            predictor.proteins = proteins
            predictor.terms = terms
            predictor.prediction_chunk_rows = 1
            predictor.write_prediction(str(chunked))
            self.assertEqual(legacy.read_bytes(), chunked.read_bytes())


if __name__ == "__main__":
    unittest.main()

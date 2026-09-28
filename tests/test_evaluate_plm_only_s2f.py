import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import build_knn_competitor_analysis as analysis  # noqa: E402
import evaluate_plm_only_s2f as evaluator  # noqa: E402
import run_plm_only_s2f as runner  # noqa: E402


class ExperimentScopeTests(unittest.TestCase):
    def test_only_selected_organisms_are_run_and_evaluated(self):
        self.assertEqual(("83333", "1111708"), evaluator.ORGANISMS)
        self.assertEqual(4, len(runner.RUNS))
        self.assertTrue(
            all("223283" not in alias and "223283" not in config
                for alias, config in runner.RUNS)
        )


class SparsePredictionSubsetTests(unittest.TestCase):
    def test_npz_is_subset_without_densifying_full_organism(self):
        with tempfile.TemporaryDirectory() as temp_name:
            run = Path(temp_name)
            proteins = pd.DataFrame(
                {"protein idx": [0, 1, 2]}, index=["P1", "P2", "P3"]
            )
            terms = pd.DataFrame(
                {"term idx": [0, 1]}, index=["GO:1", "GO:2"]
            )
            proteins.to_pickle(run / "proteins.df")
            terms.to_pickle(run / "terms.df")
            sparse.save_npz(
                run / "prediction.npz",
                sparse.coo_matrix(
                    ([0.2, 0.8, 0.9], ([0, 1, 2], [0, 1, 0])),
                    shape=(3, 2),
                ),
            )
            result = evaluator.prediction_table(run, {"P1", "P3"})
            self.assertEqual(
                {("P1", "GO:1", 0.2), ("P3", "GO:1", 0.9)},
                set(result.itertuples(index=False, name=None)),
            )


class BootstrapConclusionTests(unittest.TestCase):
    @staticmethod
    def matrix(method, prediction):
        return analysis.MethodMatrix(
            organism="1",
            method=method,
            proteins=["P1", "P2", "P3", "P4"],
            terms=["GO:1", "GO:2"],
            prediction=np.asarray(prediction, dtype=float),
            gold=np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float),
            information_content=np.ones(2),
            term_domains=["molecular_function"] * 2,
            term_names=["one", "two"],
            detail_path=Path("fixture"),
        )

    def test_interval_crossing_zero_is_unresolved(self):
        old = self.matrix("old", [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
        new = self.matrix("new", [[0.8, 0.2], [0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
        rows = evaluator.bootstrap_deltas("1", "new", new, old, 200, 1)
        self.assertTrue(all(row["conclusion"] == "unresolved" for row in rows))


if __name__ == "__main__":
    unittest.main()

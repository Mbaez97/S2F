import itertools
import json
import unittest
from pathlib import Path

import numpy as np


S2F_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DATA = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"


class MethodComparisonDisagreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = json.loads(
            (FRONTEND_DATA / "method_comparison_analysis.json").read_text(encoding="utf-8")
        )
        cls.index = json.loads(
            (FRONTEND_DATA / "pfp_score_comparison_index.json").read_text(encoding="utf-8")
        )
        cls.methods = {
            str(method["source_model"]): str(method["key"])
            for method in cls.index["methods"]
        }
        cls.rounding = {
            (str(row["organism"]), str(row["method"])): row["decimals"]
            for row in cls.analysis["metadata"]["score_rounding"]
        }
        cls.thresholds = {
            (str(row["organism"]), str(row["method"]), str(row["scope"])): float(
                row["shared_threshold"]
            )
            for row in cls.analysis["shared_threshold_summary"]
        }

    def load_organism(self, descriptor):
        relative = Path(str(descriptor["path"]))
        if relative.parts[0] == "data":
            relative = Path(*relative.parts[1:])
        payload = json.loads((FRONTEND_DATA / relative).read_text(encoding="utf-8"))
        columns = {key: index for index, key in enumerate(payload["column_keys"])}
        truth = np.asarray(
            [int(row[columns["ground_truth"]]) == 1 for row in payload["rows"]],
            dtype=bool,
        )
        proteins = np.asarray(
            [str(row[columns["protein_id"]]) for row in payload["rows"]], dtype=object
        )
        terms = np.asarray(
            [str(row[columns["term_id"]]) for row in payload["rows"]], dtype=object
        )
        domains = np.asarray(
            [str(row[columns["go_domain"]]) for row in payload["rows"]], dtype=object
        )
        organism = str(descriptor["organism"])
        scores = {}
        for method, key in self.methods.items():
            values = np.asarray(
                [
                    0.0 if row[columns[key]] is None else float(row[columns[key]])
                    for row in payload["rows"]
                ],
                dtype=float,
            )
            decimals = self.rounding[(organism, method)]
            if decimals is not None:
                rounded = np.around(values, decimals=int(decimals))
                display_rounded = np.asarray(
                    [float(f"{value:.{int(decimals)}f}") for value in values]
                )
                np.testing.assert_array_equal(rounded, display_rounded)
                values = rounded
            scores[method] = values
        return organism, truth, proteins, terms, domains, scores

    def test_every_possible_displayed_disagreement_is_threshold_correct(self):
        checked = 0
        ontology_values = ("all", "biological_process", "molecular_function", "cellular_component")
        truth_values = ("all", "positive", "negative")
        for descriptor in self.index["organisms"]:
            organism, truth, proteins, terms, domains, scores = self.load_organism(descriptor)
            display_order = np.lexsort((terms, proteins))
            for method_a, method_b in itertools.combinations(self.methods, 2):
                for scope in ("overall", "per-gene", "per-term"):
                    predicted_a = scores[method_a] >= self.thresholds[(organism, method_a, scope)]
                    predicted_b = scores[method_b] >= self.thresholds[(organism, method_b, scope)]
                    correct_a = predicted_a == truth
                    correct_b = predicted_b == truth
                    disagreements = correct_a ^ correct_b
                    for ontology in ontology_values:
                        ontology_mask = np.ones(truth.size, dtype=bool) if ontology == "all" else domains == ontology
                        for truth_filter in truth_values:
                            truth_mask = np.ones(truth.size, dtype=bool)
                            if truth_filter == "positive":
                                truth_mask = truth
                            elif truth_filter == "negative":
                                truth_mask = ~truth
                            selected_mask = disagreements & ontology_mask & truth_mask
                            displayed = display_order[selected_mask[display_order]][:30]
                            for index in displayed:
                                self.assertNotEqual(bool(correct_a[index]), bool(correct_b[index]))
                                correct_method = method_a if correct_a[index] else method_b
                                if correct_method == method_a:
                                    self.assertEqual(bool(predicted_a[index]), bool(truth[index]))
                                    self.assertNotEqual(bool(predicted_b[index]), bool(truth[index]))
                                else:
                                    self.assertEqual(bool(predicted_b[index]), bool(truth[index]))
                                    self.assertNotEqual(bool(predicted_a[index]), bool(truth[index]))
                                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

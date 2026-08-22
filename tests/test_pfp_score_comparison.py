import json
import sys
import tempfile
import unittest
from pathlib import Path


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import build_pfp_score_comparison as comparison  # noqa: E402


def detail_payload(model, scores, truth):
    pairs = set(scores) | set(truth)
    rows = []
    for protein, term in sorted(pairs):
        is_predicted = (protein, term) in scores
        rows.append(
            {
                "protein_id": protein,
                "term_id": term,
                "go_name": f"Name {term}",
                "go_domain": "molecular_function",
                "score": scores.get((protein, term), 0.0),
                "true_label": 1 if (protein, term) in truth else 0,
                "is_predicted": is_predicted,
                "model": model,
            }
        )
    return {"rows": rows}


class ComparisonPayloadTests(unittest.TestCase):
    def method_payloads(self):
        truth = {("P1", "GO:1"), ("P2", "GO:2")}
        payloads = {}
        for index, method in enumerate(comparison.METHODS):
            scores = {("P1", "GO:1"): 0.9 - index * 0.01}
            if method["source_model"] == "TALE":
                scores[("P1", "GO:2")] = 0.0
            payloads[method["source_model"]] = detail_payload(
                method["source_model"], scores, truth
            )
        return payloads

    def test_builds_complete_matrix_and_preserves_missing_vs_zero(self):
        payload = comparison.build_organism_payload(
            "123", self.method_payloads(), "2026-07-22T00:00:00+00:00"
        )
        self.assertEqual(payload["summary"]["protein_count"], 2)
        self.assertEqual(payload["summary"]["term_count"], 2)
        self.assertEqual(payload["summary"]["row_count"], 4)
        self.assertEqual(payload["column_keys"], comparison.COLUMN_KEYS)

        indices = {key: index for index, key in enumerate(payload["column_keys"])}
        rows = {(row[0], row[1]): row for row in payload["rows"]}
        self.assertEqual(rows[("P1", "GO:2")][indices["tale_score"]], 0.0)
        self.assertIsNone(rows[("P1", "GO:2")][indices["s2f_score"]])
        self.assertEqual(rows[("P2", "GO:2")][indices["ground_truth"]], 1)
        self.assertEqual(rows[("P2", "GO:1")][indices["ground_truth"]], 0)

    def test_rejects_inconsistent_ground_truth(self):
        payloads = self.method_payloads()
        payloads["TALE"] = detail_payload(
            "TALE", {("P1", "GO:1"): 0.5}, {("P1", "GO:1")}
        )
        with self.assertRaisesRegex(RuntimeError, "Ground truth differs"):
            comparison.build_organism_payload(
                "123", payloads, "2026-07-22T00:00:00+00:00"
            )

    def test_full_build_excludes_frontend_blacklist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data = Path(temp_dir)
            (data / "pfp_prediction_details").mkdir()
            (data / "index.json").write_text(
                json.dumps(
                    {
                        "frontend_excluded_benchmark_organisms": [
                            {"taxon": "999", "reason": "test"}
                        ]
                    }
                )
            )
            details = []
            payloads = self.method_payloads()
            for organism in ("123", "999"):
                for method in comparison.METHODS:
                    model = method["source_model"]
                    filename = f"{organism}_{method['key']}.json"
                    relative = f"data/pfp_prediction_details/{filename}"
                    (data / "pfp_prediction_details" / filename).write_text(
                        json.dumps(payloads[model])
                    )
                    details.append(
                        {
                            "detail_key": f"{organism}::{model}",
                            "organism": organism,
                            "model": model,
                            "available": True,
                            "path": relative,
                        }
                    )
            (data / comparison.DETAIL_INDEX_NAME).write_text(
                json.dumps({"generated_at_utc": "source-time", "details": details})
            )

            index = comparison.build_comparison(data)
            self.assertEqual([item["organism"] for item in index["organisms"]], ["123"])
            self.assertEqual(index["excluded_organisms"], ["999"])
            self.assertTrue((data / comparison.OUTPUT_DIR_NAME / "123.json").is_file())
            self.assertFalse((data / comparison.OUTPUT_DIR_NAME / "999.json").exists())


class CurrentComparisonArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = S2F_ROOT / "notebooks" / "esm_go_explorer" / "data"
        cls.index = json.loads(
            (cls.data / comparison.OUTPUT_INDEX_NAME).read_text(encoding="utf-8")
        )
        cls.payloads = {
            descriptor["organism"]: json.loads(
                comparison.detail_path(cls.data, descriptor["path"]).read_text(
                    encoding="utf-8"
                )
            )
            for descriptor in cls.index["organisms"]
        }

    def test_active_organism_matrix_shapes(self):
        descriptors = {
            item["organism"]: item for item in self.index["organisms"]
        }
        self.assertEqual(set(descriptors), {"83333", "1111708"})
        self.assertEqual(self.index["excluded_organisms"], ["223283"])
        self.assertEqual(
            (
                descriptors["83333"]["protein_count"],
                descriptors["83333"]["term_count"],
                descriptors["83333"]["row_count"],
            ),
            (160, 574, 91840),
        )
        self.assertEqual(
            (
                descriptors["1111708"]["protein_count"],
                descriptors["1111708"]["term_count"],
                descriptors["1111708"]["row_count"],
            ),
            (33, 93, 3069),
        )

    def test_rows_follow_schema_and_keep_real_zero(self):
        payload = self.payloads["83333"]
        self.assertEqual(payload["column_keys"], comparison.COLUMN_KEYS)
        self.assertEqual({len(row) for row in payload["rows"]}, {len(comparison.COLUMN_KEYS)})
        indices = {key: index for index, key in enumerate(payload["column_keys"])}
        tale_values = [row[indices["tale_score"]] for row in payload["rows"]]
        self.assertIn(0.0, tale_values)
        self.assertIn(None, tale_values)



if __name__ == "__main__":
    unittest.main()

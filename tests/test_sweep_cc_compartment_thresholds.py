import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import sweep_cc_compartment_thresholds as sweep


class ThresholdSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"

        pair_fields = [
            "protein_id", "compartment_id", "compartment_label", "s2f_score",
            "threshold", "confidence_tier", "supporting_go_ids",
            "supporting_go_paths", "compartment_lineage", "stage_flag",
            "curated_sources",
        ]
        pairs = [
            {
                "protein_id": "P1", "compartment_id": "GO:1",
                "compartment_label": "A", "s2f_score": "0.7",
                "threshold": "0.1", "confidence_tier": "s2f_only_hypothesis",
            },
            {
                "protein_id": "P1", "compartment_id": "GO:2",
                "compartment_label": "B", "s2f_score": "0.3",
                "threshold": "0.1", "confidence_tier": "s2f_only_hypothesis",
            },
            {
                "protein_id": "P2", "compartment_id": "GO:1",
                "compartment_label": "A", "s2f_score": "0.15",
                "threshold": "0.1", "confidence_tier": "s2f_only_hypothesis",
            },
        ]
        self.write_tsv(
            self.source / "protein_compartments.tsv", pair_fields, pairs
        )

        summary_fields = [
            "protein_id", "status", "compartment_ids", "compartment_labels",
            "confidence_tiers", "scores", "n_high_confidence_compartments",
            "n_mapped_cc_rows", "n_unresolved_cc_rows",
            "n_non_spatial_cc_rows", "n_unknown_or_obsolete_go_rows",
        ]
        summaries = [
            {
                "protein_id": "P1", "status": "assigned",
                "compartment_ids": "GO:1;GO:2", "compartment_labels": "A;B",
                "confidence_tiers": "s2f_only_hypothesis;s2f_only_hypothesis",
                "scores": "0.7;0.3", "n_high_confidence_compartments": "2",
                "n_mapped_cc_rows": "5", "n_unresolved_cc_rows": "0",
                "n_non_spatial_cc_rows": "0",
                "n_unknown_or_obsolete_go_rows": "0",
            },
            {
                "protein_id": "P2", "status": "assigned",
                "compartment_ids": "GO:1", "compartment_labels": "A",
                "confidence_tiers": "s2f_only_hypothesis", "scores": "0.15",
                "n_high_confidence_compartments": "1",
                "n_mapped_cc_rows": "5", "n_unresolved_cc_rows": "0",
                "n_non_spatial_cc_rows": "0",
                "n_unknown_or_obsolete_go_rows": "0",
            },
            {
                "protein_id": "P3", "status": "below_threshold",
                "compartment_ids": "", "compartment_labels": "",
                "confidence_tiers": "", "scores": "",
                "n_high_confidence_compartments": "0",
                "n_mapped_cc_rows": "5", "n_unresolved_cc_rows": "0",
                "n_non_spatial_cc_rows": "0",
                "n_unknown_or_obsolete_go_rows": "0",
            },
        ]
        self.write_tsv(
            self.source / "protein_compartment_summary.tsv",
            summary_fields,
            summaries,
        )
        (self.source / "run_metadata.json").write_text(
            json.dumps({"schema_version": 1}), encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_tsv(path, fields, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_tsv(path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def args(self, thresholds):
        return Namespace(
            source=self.source,
            output=self.output,
            thresholds=thresholds,
        )

    def test_build_creates_one_complete_result_folder_per_threshold(self):
        metadata = sweep.build(self.args(["0.10", "0.20", "0.60"]))
        self.assertEqual([0.1, 0.2, 0.6], metadata["thresholds"])

        rows_020 = self.read_tsv(
            self.output / "threshold_0p20" / "protein_compartments.tsv"
        )
        self.assertEqual(2, len(rows_020))
        self.assertTrue(all(row["threshold"] == "0.2" for row in rows_020))

        summary_020 = {
            row["protein_id"]: row
            for row in self.read_tsv(
                self.output / "threshold_0p20" / "protein_compartment_summary.tsv"
            )
        }
        self.assertEqual("assigned", summary_020["P1"]["status"])
        self.assertEqual("below_threshold", summary_020["P2"]["status"])
        self.assertEqual("2", summary_020["P1"]["n_high_confidence_compartments"])

        rows_060 = self.read_tsv(
            self.output / "threshold_0p60" / "protein_compartments.tsv"
        )
        self.assertEqual(["GO:1"], [row["compartment_id"] for row in rows_060])
        self.assertTrue(
            (self.output / "threshold_0p60" / "run_metadata.json").exists()
        )

    def test_threshold_below_source_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "below the source threshold"):
            sweep.build(self.args(["0.05"]))


if __name__ == "__main__":
    unittest.main()

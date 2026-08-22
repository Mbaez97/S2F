import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import cc_compartments as cc


MINI_OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: cellular component
namespace: cellular_component

[Term]
id: GO:0001000
name: compartment A
namespace: cellular_component
is_a: GO:0000001 ! cellular component

[Term]
id: GO:0001001
name: compartment A child
namespace: cellular_component
alt_id: GO:0091001
is_a: GO:0001000 ! compartment A

[Term]
id: GO:0001002
name: part of compartment A
namespace: cellular_component
relationship: part_of GO:0001000 ! compartment A

[Term]
id: GO:0002000
name: compartment B
namespace: cellular_component
is_a: GO:0000001 ! cellular component

[Term]
id: GO:0002001
name: shared child
namespace: cellular_component
is_a: GO:0001000 ! compartment A
relationship: part_of GO:0002000 ! compartment B

[Term]
id: GO:0003000
name: protein-containing complex
namespace: cellular_component
is_a: GO:0000001 ! cellular component

[Term]
id: GO:0003001
name: example complex
namespace: cellular_component
is_a: GO:0003000 ! protein-containing complex

[Term]
id: GO:0004000
name: obsolete location
namespace: cellular_component
is_obsolete: true
replaced_by: GO:0001001

[Term]
id: GO:0005000
name: binding
namespace: molecular_function
"""


def write_fixture(root: Path):
    obo = root / "mini.obo"
    obo.write_text(MINI_OBO, encoding="utf-8")
    config = root / "slim.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "relations": ["is_a", "part_of"],
        "anchors": [
            {"go_id": "GO:0001000", "label": "A"},
            {"go_id": "GO:0002000", "label": "B"},
        ],
        "non_spatial_roots": ["GO:0003000"],
        "stage_flags": {
            "GO:0002000": {"present_in": ["stage-x"]}
        },
    }), encoding="utf-8")
    return obo, config


class OntologyMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.obo, self.config = write_fixture(self.root)
        self.mapper = cc.CompartmentMapper.from_files(self.obo, self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_alt_obsolete_and_safe_paths(self):
        resolution, status, paths = self.mapper.map_term("GO:0091001")
        self.assertEqual("alt_id", resolution.status)
        self.assertEqual("mapped", status)
        self.assertEqual("GO:0001000", paths[0].compartment_id)
        self.assertEqual(("is_a",), paths[0].relations)

        resolution, status, paths = self.mapper.map_term("GO:0004000")
        self.assertEqual("obsolete_replaced", resolution.status)
        self.assertEqual("GO:0001001", resolution.resolved_id)
        self.assertEqual("mapped", status)

        _, status, paths = self.mapper.map_term("GO:0001002")
        self.assertEqual("mapped", status)
        self.assertEqual(("part_of",), paths[0].relations)

    def test_multi_label_non_spatial_and_unresolved(self):
        _, status, paths = self.mapper.map_term("GO:0002001")
        self.assertEqual("mapped", status)
        self.assertEqual({"GO:0001000", "GO:0002000"}, {item.compartment_id for item in paths})

        _, status, paths = self.mapper.map_term("GO:0003001")
        self.assertEqual("non_spatial_cc", status)
        self.assertFalse(paths)

        resolution, status, _ = self.mapper.map_term("GO:9999999")
        self.assertEqual("unknown_go_id", resolution.status)
        self.assertEqual("unknown_go_id", status)

        _, status, _ = self.mapper.map_term("GO:0005000")
        self.assertEqual("non_cc", status)

    def test_assign_is_max_aggregated_multi_label_and_fail_closed(self):
        prediction = self.root / "prediction.df"
        prediction.write_text(
            "P1\tGO:0001001\t0.8\n"
            "P1\tGO:0001000\t0.7\n"
            "P1\tGO:0002001\t0.9\n"
            "P2\tGO:0003001\t0.95\n"
            "P3\tGO:0001001\t0.2\n"
            "P4\tGO:0005000\t0.99\n",
            encoding="utf-8",
        )
        output = self.root / "out"
        result = cc.main([
            "assign", "--obo", str(self.obo), "--slim", str(self.config),
            "--prediction", str(prediction), "--threshold", "0.75",
            "--output", str(output),
        ])
        self.assertEqual(0, result)
        with (output / "protein_compartments.tsv").open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(
            {("P1", "GO:0001000", "0.90000000000000002"),
             ("P1", "GO:0002000", "0.90000000000000002")},
            {(row["protein_id"], row["compartment_id"], row["s2f_score"]) for row in rows},
        )
        self.assertTrue(next(row for row in rows if row["compartment_id"] == "GO:0002000")["stage_flag"])
        with (output / "protein_compartment_summary.tsv").open(newline="") as handle:
            summary = {row["protein_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
        self.assertEqual("assigned", summary["P1"]["status"])
        self.assertEqual("unresolved", summary["P2"]["status"])
        self.assertEqual("below_threshold", summary["P3"]["status"])
        self.assertEqual("unresolved", summary["P4"]["status"])

        reversed_prediction = self.root / "prediction-reversed.df"
        reversed_prediction.write_text(
            "".join(reversed(prediction.read_text(encoding="utf-8").splitlines(keepends=True))),
            encoding="utf-8",
        )
        reversed_output = self.root / "out-reversed"
        self.assertEqual(0, cc.main([
            "assign", "--obo", str(self.obo), "--slim", str(self.config),
            "--prediction", str(reversed_prediction), "--threshold", "0.75",
            "--output", str(reversed_output),
        ]))
        for filename in ("protein_compartments.tsv", "protein_compartment_summary.tsv"):
            self.assertEqual(
                (output / filename).read_bytes(),
                (reversed_output / filename).read_bytes(),
                f"{filename} changed when input row order changed",
            )

    def test_malformed_prediction_is_rejected(self):
        path = self.root / "bad.df"
        path.write_text("P1\tGO:0001000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected 3"):
            list(cc.iter_s2f_predictions(path))


class CalibrationTests(unittest.TestCase):
    def test_evidence_tiers_require_explicit_corroboration_or_conflict(self):
        self.assertEqual("s2f_only_hypothesis", cc.evidence_tier(True, []))
        self.assertEqual(
            "reviewed_experimental_agreement",
            cc.evidence_tier(True, [{"assertion": "in", "evidence_level": "reviewed_experimental"}]),
        )
        self.assertEqual(
            "reviewed_inferred_agreement",
            cc.evidence_tier(True, [{"assertion": "in", "evidence_level": "reviewed_inferred"}]),
        )
        self.assertEqual(
            "conflict",
            cc.evidence_tier(True, [{"assertion": "not_in", "evidence_level": "reviewed_experimental"}]),
        )

    def test_network_free_uniprot_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source.tsv"
            source.write_text(
                "Entry\tSequence\tSubcellular location [CC]\n"
                "P1\tAAAA\tSUBCELLULAR LOCATION: Nucleus {ECO:0000269}.\n"
                "P2\tBBBB\t\n",
                encoding="utf-8",
            )
            output = root / "snapshot"
            self.assertEqual(0, cc.main([
                "fetch-uniprot", "--source-tsv", str(source), "--output", str(output),
            ]))
            metadata = json.loads(
                (output / "uniprot_reviewed_tcruzi.metadata.json").read_text()
            )
            self.assertEqual(2, metadata["counts"]["reviewed_entries"])
            self.assertEqual(1, metadata["counts"]["with_experimental_location_eco"])

    def test_threshold_selection_and_fail_closed(self):
        rows = [
            {"protein_id": "P1", "compartment_id": "C", "ground_truth": 1, "score": 0.9},
            {"protein_id": "P2", "compartment_id": "C", "ground_truth": 0, "score": 0.8},
            {"protein_id": "P3", "compartment_id": "C", "ground_truth": 1, "score": 0.7},
        ]
        self.assertEqual(0.9, cc.select_global_threshold(rows, "score", 0.9))
        impossible = [dict(row, ground_truth=0) for row in rows]
        self.assertIsNone(cc.select_global_threshold(impossible, "score", 0.9))

    def test_real_go_and_tcruzi_slim_load(self):
        obo = S2F_ROOT / "go.obo"
        slim = S2F_ROOT / "conf" / "cc_compartments_tcruzi.json"
        if not obo.exists():
            self.skipTest("Local go.obo is absent")
        mapper = cc.CompartmentMapper.from_files(obo, slim)
        _, status, paths = mapper.map_term("GO:0106123")
        self.assertEqual("mapped", status)
        self.assertEqual("GO:0106123", paths[0].compartment_id)
        self.assertIn("GO:0106123", mapper.stage_flags)

    def test_exact_target_sequence_audit(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            uniprot = root / "uniprot.tsv"
            uniprot.write_text(
                "Entry\tEntry Name\tSequence\tSubcellular location [CC]\tGene Ontology (cellular component)\n"
                "P1\tONE_TEST\tMPEPTIDE\tSUBCELLULAR LOCATION: Nucleus.\tNucleus [GO:0005634]\n"
                "P2\tTWO_TEST\tAAAA\t\t\n",
                encoding="utf-8",
            )
            fasta = root / "target.fasta"
            fasta.write_text(
                ">target-one description\nMPEP\nTIDE\n>target-two\nBBBB\n",
                encoding="utf-8",
            )
            output = root / "audit"
            self.assertEqual(0, cc.main([
                "audit-target", "--uniprot-tsv", str(uniprot),
                "--target-fasta", str(fasta), "--output", str(output),
            ]))
            metadata = json.loads((output / "target_sequence_audit.json").read_text())
            self.assertEqual(2, metadata["target_protein_count"])
            self.assertEqual(1, metadata["exactly_matched_target_protein_count"])
            with (output / "exact_sequence_matches.tsv").open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual("target-one", rows[0]["target_protein_id"])
            self.assertEqual("P1", rows[0]["uniprot_accession"])


if __name__ == "__main__":
    unittest.main()

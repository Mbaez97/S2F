import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import kegg_compartments as bridge


MINI_OBO = """format-version: 1.2

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component

[Term]
id: GO:0005737
name: cytoplasm
namespace: cellular_component
is_a: GO:0005575 ! cellular_component

[Term]
id: GO:0005829
name: cytosol
namespace: cellular_component
is_a: GO:0005737 ! cytoplasm

[Term]
id: GO:0005739
name: mitochondrion
namespace: cellular_component
is_a: GO:0005575 ! cellular_component

[Term]
id: GO:0005634
name: nucleus
namespace: cellular_component
is_a: GO:0005575 ! cellular_component

[Term]
id: GO:0009507
name: chloroplast
namespace: cellular_component
is_a: GO:0005575 ! cellular_component
"""


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshots"
        self.snapshot.mkdir()
        self.output = self.root / "output"

        self.obo = self.root / "go.obo"
        self.obo.write_text(MINI_OBO, encoding="utf-8")
        self.slim = self.root / "slim.json"
        self.slim.write_text(json.dumps({
            "schema_version": 1,
            "anchors": [
                {"go_id": "GO:0005737", "label": "cytoplasm", "scope": "generic"},
                {"go_id": "GO:0005829", "label": "cytosol", "scope": "generic"},
                {"go_id": "GO:0005739", "label": "mitochondrion", "scope": "generic"},
                {"go_id": "GO:0005634", "label": "nucleus", "scope": "generic"},
                {"go_id": "GO:0009507", "label": "chloroplast", "scope": "cross-organism validation"},
            ],
        }), encoding="utf-8")

        header = (
            "Entry\tEntry Name\tReviewed\tProtein names\tGene Names (primary)\t"
            "Gene Names\tOrganism (ID)\tEC number\tKEGG\tSubcellular location [CC]\n"
        )
        (self.snapshot / "uniprot_dm28c.tsv").write_text(
            header
            + "DMP1\tDMP1_TCRUZ\tunreviewed\tEnzyme one\tTCDM_00001\tTCDM_00001\t1416333\t1.1.1.1\t\tSUBCELLULAR LOCATION: Cytoplasm.\n"
            + "DMP2\tDMP2_TCRUZ\tunreviewed\tEnzyme two\tTCDM_00002\tTCDM_00002\t1416333\t2.2.2.2\t\tSUBCELLULAR LOCATION: Cytoplasm {ECO:0000250}.\n"
            + "DMP3\tDMP3_TCRUZ\tunreviewed\tEnzyme three\tTCDM_00003\tTCDM_00003\t1416333\t3.3.3.3\t\tSUBCELLULAR LOCATION: Cytoplasm.\n",
            encoding="utf-8",
        )
        (self.snapshot / "uniprot_tcr_reference.tsv").write_text(
            header
            + "REF1\tREF1_TCRUZ\treviewed\tReference one\tGENE1\tGENE1\t353153\t1.1.1.1\ttcr:GENE1;\tSUBCELLULAR LOCATION: Mitochondrion {ECO:0000269}.\n"
            + "REF1U\tREF1U_TCRUZ\tunreviewed\tDuplicate\tGENE1\tGENE1\t353153\t1.1.1.1\ttcr:GENE1;\tSUBCELLULAR LOCATION: Nucleus.\n"
            + "REF3\tREF3_TCRUZ\tunreviewed\tReference three\tGENE3\tGENE3\t353153\t3.3.3.3\ttcr:GENE3;\tSUBCELLULAR LOCATION: Nucleus. Note=Also detected in chloroplast.\n",
            encoding="utf-8",
        )
        (self.snapshot / "kegg_enzyme_reaction.tsv").write_text(
            "ec:1.1.1.1\trn:R00001\n"
            "ec:2.2.2.2\trn:R00002\n"
            "ec:3.3.3.3\trn:R00003\n",
            encoding="utf-8",
        )
        (self.snapshot / "kegg_tcr_genes_by_enzyme.tsv").write_text(
            "ec:1.1.1.1\ttcr:GENE1\n"
            "ec:3.3.3.3\ttcr:GENE3\n",
            encoding="utf-8",
        )
        (self.snapshot / "kegg_tcr_uniprot.tsv").write_text(
            "tcr:GENE1\tup:REF1\n"
            "tcr:GENE1\tup:REF1U\n"
            "tcr:GENE3\tup:REF3\n",
            encoding="utf-8",
        )
        (self.snapshot / "kegg_reaction_names.tsv").write_text(
            "rn:R00001\tReaction one\n"
            "rn:R00002\tReaction two\n"
            "rn:R00003\tReaction three\n",
            encoding="utf-8",
        )

        fields = [
            "protein_id", "compartment_id", "compartment_label", "s2f_score",
            "threshold", "confidence_tier", "supporting_go_ids",
            "supporting_go_paths", "compartment_lineage", "stage_flag",
            "curated_sources",
        ]
        with (self.root / "protein_compartments.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow({"protein_id": "TCDM_00001-t26_1-p1", "compartment_id": "GO:0005739", "compartment_label": "mitochondrion", "s2f_score": "0.8"})
            writer.writerow({"protein_id": "TCDM_00002-t26_1-p1", "compartment_id": "GO:0005829", "compartment_label": "cytosol", "s2f_score": "0.7"})
            writer.writerow({"protein_id": "TCDM_00003-t26_1-p1", "compartment_id": "GO:0005739", "compartment_label": "mitochondrion", "s2f_score": "0.9"})
            writer.writerow({"protein_id": "TCDM_00003-t26_1-p1", "compartment_id": "GO:0009507", "compartment_label": "chloroplast", "s2f_score": "0.95"})
        (self.root / "protein_summary.tsv").write_text(
            "protein_id\tstatus\n"
            "TCDM_00001-t26_1-p1\tassigned\n"
            "TCDM_00002-t26_1-p1\tassigned\n"
            "TCDM_00003-t26_1-p1\tassigned\n"
            "TCDM_99999-t26_1-p1\tbelow_threshold\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def args(self):
        return Namespace(
            protein_compartments=self.root / "protein_compartments.tsv",
            protein_summary=self.root / "protein_summary.tsv",
            interproscan=None,
            snapshot_dir=self.snapshot,
            output=self.output,
            slim=self.slim,
            obo=self.obo,
        )

    def test_bridge_builds_independent_hypotheses_and_comparison(self):
        summary = bridge.build(self.args())
        with (self.output / "protein_reaction_compartment_comparison.tsv").open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        by_reaction = {row["reaction_id"]: row for row in rows}
        self.assertEqual("exact_agreement", by_reaction["R00001"]["comparison_status"])
        self.assertEqual("tcr_reference_reaction_enzyme", by_reaction["R00001"]["reaction_location_basis"])
        self.assertEqual("GO:0005739", by_reaction["R00001"]["reaction_compartment_ids"])

        self.assertEqual("compatible_agreement", by_reaction["R00002"]["comparison_status"])
        self.assertEqual("dm28c_exact_uniprot_fallback", by_reaction["R00002"]["reaction_location_basis"])
        self.assertEqual("GO:0005737", by_reaction["R00002"]["reaction_compartment_ids"])

        self.assertEqual("conflict", by_reaction["R00003"]["comparison_status"])
        self.assertEqual("GO:0005634", by_reaction["R00003"]["reaction_compartment_ids"])
        self.assertNotIn("GO:0009507", by_reaction["R00003"]["s2f_compartment_ids"])

        with (self.output / "reaction_compartment_hypotheses.tsv").open(newline="") as handle:
            hypotheses = list(csv.DictReader(handle, delimiter="\t"))
        reaction_one = [row for row in hypotheses if row["reaction_id"] == "R00001"]
        self.assertEqual(["REF1"], [row["reference_uniprot_accession"] for row in reaction_one])
        self.assertEqual("experimental", reaction_one[0]["location_evidence_level"])

        self.assertEqual(1, summary["counts"]["excluded_s2f_rows"])
        self.assertEqual(3, summary["counts"]["evaluable_comparisons"])
        self.assertEqual(2, summary["counts"]["aligned_comparisons"])

    def test_location_parser_ignores_note_and_cross_organism_anchor(self):
        labels, allowed, _ = bridge.load_slim(self.slim)
        assertions = bridge.location_assertions(
            "SUBCELLULAR LOCATION: Nucleus. Note=Detected in chloroplast.",
            labels,
            allowed,
        )
        self.assertEqual(["GO:0005634"], [row.compartment_id for row in assertions])

    def test_interpro_go_terms_map_only_to_complete_ec_xrefs(self):
        obo = self.root / "ec.obo"
        obo.write_text(
            "[Term]\n"
            "id: GO:0000001\n"
            "xref: EC:1.2.3.4\n\n"
            "[Term]\n"
            "id: GO:0000002\n"
            "xref: EC:1.2.-.-\n",
            encoding="utf-8",
        )
        interpro = self.root / "interpro.tsv"
        columns = ["P4"] + ["-"] * 12 + ["GO:0000001(InterPro)|GO:0000002(InterPro)"] + ["-"]
        interpro.write_text("\t".join(columns) + "\n", encoding="utf-8")
        assignments = bridge.interpro_go_ec_assignments(
            interpro, bridge.ontology_go_ec(obo)
        )
        self.assertEqual({"1.2.3.4": {"GO:0000001"}}, assignments["P4"])

    def test_cli_build_is_deterministic(self):
        argv = [
            "build",
            "--protein-compartments", str(self.root / "protein_compartments.tsv"),
            "--protein-summary", str(self.root / "protein_summary.tsv"),
            "--snapshot-dir", str(self.snapshot),
            "--output", str(self.output),
            "--slim", str(self.slim),
            "--obo", str(self.obo),
        ]
        self.assertEqual(0, bridge.main(argv))
        first = (self.output / "protein_reaction_compartment_comparison.tsv").read_bytes()
        self.assertEqual(0, bridge.main(argv))
        self.assertEqual(first, (self.output / "protein_reaction_compartment_comparison.tsv").read_bytes())


if __name__ == "__main__":
    unittest.main()

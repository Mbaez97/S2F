import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


S2F_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2F_ROOT / "scripts"))

import kegg_compartments as cli
import strict_kegg_compartments as strict


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
id: GO:0005739
name: mitochondrion
namespace: cellular_component
is_a: GO:0005575 ! cellular_component

[Term]
id: GO:0005634
name: nucleus
namespace: cellular_component
is_a: GO:0005575 ! cellular_component
"""


def archived_record(accession, gene, sequence, ec="", location=""):
    ec_text = f" EC={ec};" if ec else ""
    location_text = (
        f"CC   -!- SUBCELLULAR LOCATION: {location}.\n" if location else ""
    )
    return (
        f"ID   {accession}_TCRUZ Unreviewed; {len(sequence)} AA.\n"
        f"AC   {accession};\n"
        "DT   01-JAN-2020, entry version 3.\n"
        f"DE   RecName: Full=Fixture enzyme;{ec_text}\n"
        f"GN   Name={gene};\n"
        f"{location_text}"
        f"SQ   SEQUENCE   {len(sequence)} AA;\n"
        f"     {sequence} {len(sequence)}\n"
        "//\n"
    )


class StrictBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "archive"
        self.records = self.archive / "records" / "AAA"
        self.records.mkdir(parents=True)
        self.snapshot = self.root / "snapshots"
        self.snapshot.mkdir()
        self.output = self.root / "output"

        (self.archive / "unisave_fetch_metadata.json").write_text(
            json.dumps({"complete": True, "expected_accessions": 5}),
            encoding="utf-8",
        )
        (self.records / "AAA001.txt").write_text(
            archived_record(
                "AAA001", "TCDM_00001", "MAAA", "1.1.1.1", "Cytoplasm"
            ),
            encoding="utf-8",
        )
        (self.records / "AAA002.txt").write_text(
            archived_record("AAA002", "TCDM_00002", "MBBB", "2.2.2.2"),
            encoding="utf-8",
        )
        (self.records / "AAA003.txt").write_text(
            archived_record("AAA003", "TCDM_00003", "MCCC", "3.3.-.-"),
            encoding="utf-8",
        )
        (self.records / "AAA004.txt").write_text(
            archived_record("AAA004", "TCDM_99999", "MDDD", "4.4.4.4"),
            encoding="utf-8",
        )
        (self.records / "AAA005.txt").write_text(
            archived_record(
                "AAA005", "TCDM_00004", "MEEE", "5.5.5.5", "Nucleus"
            ),
            encoding="utf-8",
        )

        self.target_fasta = self.root / "targets.faa"
        self.target_fasta.write_text(
            ">TCDM_00001-t26_1-p1\nMAAA*\n"
            ">TCDM_00002-t26_1-p1\nMBBB\n"
            ">TCDM_00003-t26_1-p1\nMCCC\n"
            ">TCDM_00004-t26_1-p1\nMEEE\n"
            ">TCDM_00005-t26_1-p1\nMEEE\n",
            encoding="utf-8",
        )

        self.obo = self.root / "go.obo"
        self.obo.write_text(MINI_OBO, encoding="utf-8")
        self.slim = self.root / "slim.json"
        self.slim.write_text(
            json.dumps({
                "schema_version": 1,
                "anchors": [
                    {"go_id": "GO:0005737", "label": "cytoplasm", "scope": "generic"},
                    {"go_id": "GO:0005739", "label": "mitochondrion", "scope": "generic"},
                    {"go_id": "GO:0005634", "label": "nucleus", "scope": "generic"},
                ]
            }),
            encoding="utf-8",
        )

        header = (
            "Entry\tEntry Name\tReviewed\tProtein names\tGene Names (primary)\t"
            "Gene Names\tOrganism (ID)\tEC number\tKEGG\tSubcellular location [CC]\n"
        )
        (self.snapshot / "uniprot_tcr_reference.tsv").write_text(
            header
            + "REF2\tREF2_TCRUZ\treviewed\tReference\tGENE2\tGENE2\t353153\t"
            "2.2.2.2\ttcr:GENE2;\tSUBCELLULAR LOCATION: Mitochondrion {ECO:0000269}.\n",
            encoding="utf-8",
        )
        (self.snapshot / "kegg_enzyme_reaction.tsv").write_text(
            "ec:1.1.1.1\trn:R00001\n"
            "ec:2.2.2.2\trn:R00002\n"
            "ec:5.5.5.5\trn:R00005\n",
            encoding="utf-8",
        )
        (self.snapshot / "kegg_tcr_genes_by_enzyme.tsv").write_text(
            "ec:2.2.2.2\ttcr:GENE2\n", encoding="utf-8"
        )
        (self.snapshot / "kegg_tcr_uniprot.tsv").write_text(
            "tcr:GENE2\tup:REF2\n", encoding="utf-8"
        )
        (self.snapshot / "kegg_reaction_names.tsv").write_text(
            "rn:R00001\tReaction one\n"
            "rn:R00002\tReaction two\n"
            "rn:R00005\tReaction five\n",
            encoding="utf-8",
        )

        fields = [
            "protein_id", "compartment_id", "compartment_label", "s2f_score"
        ]
        self.compartments = self.root / "protein_compartments.tsv"
        with self.compartments.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow({
                "protein_id": "TCDM_00001-t26_1-p1",
                "compartment_id": "GO:0005737",
                "compartment_label": "cytoplasm",
                "s2f_score": "0.9",
            })
            writer.writerow({
                "protein_id": "TCDM_00002-t26_1-p1",
                "compartment_id": "GO:0005634",
                "compartment_label": "nucleus",
                "s2f_score": "0.8",
            })
            writer.writerow({
                "protein_id": "TCDM_00004-t26_1-p1",
                "compartment_id": "GO:0005634",
                "compartment_label": "nucleus",
                "s2f_score": "0.7",
            })

    def tearDown(self):
        self.temp.cleanup()

    def args(self):
        return Namespace(
            archive_dir=self.archive,
            target_fasta=self.target_fasta,
            protein_compartments=self.compartments,
            protein_summary=None,
            snapshot_dir=self.snapshot,
            output=self.output,
            slim=self.slim,
            obo=self.obo,
            allow_incomplete_archive=False,
        )

    def read_output(self, filename):
        with (self.output / filename).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def test_build_uses_only_direct_complete_ec_and_exact_sequence(self):
        summary = strict.build_strict(self.args())
        known = self.read_output("known_protein_ec.tsv")
        self.assertEqual(
            {"TCDM_00001-t26_1-p1", "TCDM_00002-t26_1-p1", "TCDM_00004-t26_1-p1"},
            {row["protein_id"] for row in known},
        )
        self.assertTrue(all(row["sequence_match"] == "exact" for row in known))
        self.assertTrue(all(row["direct_ec_source"] == "uniprot_unisave_explicit_ec" for row in known))
        self.assertNotIn("InterPro", "\n".join("\t".join(row.values()) for row in known))

        excluded = self.read_output("excluded_direct_ec_records.tsv")
        reasons = {row["reason"] for row in excluded}
        self.assertIn("no_explicit_complete_ec", reasons)
        self.assertIn("no_exact_sequence_match", reasons)
        self.assertEqual(3, summary["counts"]["direct_ec_proteins"])

    def test_exact_dm28c_location_precedes_reference_ec_fallback(self):
        strict.build_strict(self.args())
        comparison = {
            row["reaction_id"]: row
            for row in self.read_output("protein_reaction_compartment_vs_s2f.tsv")
        }
        self.assertEqual("GO:0005737", comparison["R00001"]["reaction_compartment_ids"])
        self.assertEqual("dm28c_archived_exact_protein", comparison["R00001"]["reaction_location_sources"])
        self.assertEqual("exact_agreement", comparison["R00001"]["comparison_status"])
        self.assertEqual("GO:0005739", comparison["R00002"]["reaction_compartment_ids"])
        self.assertEqual("tcr_cl_brener_ec_transfer", comparison["R00002"]["reaction_location_sources"])
        self.assertEqual("conflict", comparison["R00002"]["comparison_status"])

    def test_reaction_compartment_catalog_is_unique_and_s2f_independent(self):
        strict.build_strict(self.args())
        path = self.output / "reaction_compartment_pairs.tsv"
        header = path.read_text(encoding="utf-8").splitlines()[0].lower()
        self.assertNotIn("s2f", header)
        rows = self.read_output("reaction_compartment_pairs.tsv")
        keys = [(row["reaction_id"], row["compartment_id"]) for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            {("R00001", "GO:0005737"), ("R00002", "GO:0005739"), ("R00005", "GO:0005634")},
            set(keys),
        )

    def test_incomplete_archive_fails_closed(self):
        (self.archive / "unisave_fetch_metadata.json").write_text(
            json.dumps({"complete": False}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "archive is incomplete"):
            strict.build_strict(self.args())

    def test_missing_archive_metadata_fails_closed(self):
        (self.archive / "unisave_fetch_metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "metadata is missing"):
            strict.build_strict(self.args())

    def test_strict_cli_does_not_accept_interproscan(self):
        argv = [
            "build-strict",
            "--archive-dir", str(self.archive),
            "--target-fasta", str(self.target_fasta),
            "--protein-compartments", str(self.compartments),
            "--snapshot-dir", str(self.snapshot),
            "--output", str(self.output),
            "--interproscan", "forbidden.tsv",
        ]
        with self.assertRaises(SystemExit):
            cli.create_parser().parse_args(argv)

    def test_accession_inventory_unions_both_local_sources(self):
        found = self.root / "found.txt"
        found.write_text("AAA001\nAAA002\n", encoding="utf-8")
        missing = self.root / "missing.fasta"
        missing.write_text(
            ">tr|AAA002|SECOND\nMAAA\n>sp|AAA003|THIRD\nMBBB\n",
            encoding="utf-8",
        )
        self.assertEqual(
            ["AAA001", "AAA002", "AAA003"],
            strict.accession_inventory(found, missing),
        )

    def test_latest_unisave_version_uses_first_tsv_data_row(self):
        content = (
            b"Entry version\tSequence version\tEntry name\n"
            b"23\t1\tAAA001_TCRUZ\n22\t1\tAAA001_TCRUZ\n"
        )
        self.assertEqual("23", strict.latest_unisave_version(content))


if __name__ == "__main__":
    unittest.main()

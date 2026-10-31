# Direct EC reaction-compartment comparison

`scripts/kegg_compartments.py` implements the reaction bridge described in the collaborator note at `/home/marcelo_baez/paccanaro-lab/MP-Research/CBM_L02_compartmentsBridge.md`.

The recommended `*-strict` commands keep the two evidence routes independent:

1. Direct route: exact Dm28c sequence → explicit complete EC in that protein's archived UniProt record → KEGG reaction → functional-location evidence → GO CC slim.
2. S2F route: S2F CC score → the same conservative GO CC slim.
3. Comparison: the two compartment sets are compared only after both have been constructed.

The strict route does not use InterPro, GO-to-EC conversion, protein-name parsing, partial EC numbers, or ortholog EC transfer to assign an EC to a Dm28c protein. Historical UniProt entries are used because most Dm28c TrEMBL entries are no longer in the current UniProt release. The archived sequence must exactly match the S2F target sequence; ambiguous exact duplicates are accepted only when the archived TCDM gene identifier resolves one target.

## Strict run

The UniSave archive is resumable. Each completed accession is stored separately, and rerunning the command reuses it. Progress logs include completed accessions, failures, rate, and an ETA.

From `S2F/`:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python -u \
  scripts/kegg_compartments.py run-strict \
  --found-accessions /home/marcelo_baez/paccanaro-lab/MP-Research/data/query/Dm28c_alphafold_pdbs/found_ids.txt \
  --missing-fasta /home/marcelo_baez/paccanaro-lab/MP-Research/data/query/Dm28c_alphafold_pdbs/not_found_DM28C_sequences.fasta \
  --archive-dir results/dm28c_reaction_compartment_direct_ec_2026-08-22/archive \
  --target-fasta /run/media/marcelo_baez/HD_Disc1/databases/TriTrypDB-68_TcruziDm28c2014_AnnotatedProteins_clean.faa \
  --protein-compartments results/dm28c_cc_compartments_manual_t0p1_2026-08-22/protein_compartments.tsv \
  --snapshot-dir results/dm28c_reaction_compartment_comparison_2026-08-22/snapshots \
  --output results/dm28c_reaction_compartment_direct_ec_2026-08-22 \
  --workers 8
```

Use `fetch-direct-ec` to populate only the archived protein records. Use `build-strict` to rebuild all result tables from the frozen archive and KEGG snapshots without network access. By default, `build-strict` refuses to create a result when the archive metadata reports failed downloads.

## Strict outputs

- `known_protein_ec.tsv`: only direct, complete protein-EC records with exact sequence matching and provenance.
- `protein_reaction_compartment_vs_s2f.tsv`: reaction-level comparison with the existing S2F CC predictions.
- `reaction_compartment_pairs.tsv`: independent unique reaction-compartment catalog; it contains no S2F fields.
- `reaction_compartment_evidence.tsv`: protein/reference-level evidence behind the catalog.
- `excluded_direct_ec_records.tsv`: incomplete ECs, missing ECs, sequence mismatches, ambiguous matches, and parsing failures.
- `unresolved_reaction_compartments.tsv`: explicit EC→reaction or reaction→location failures.
- `excluded_s2f_compartments.tsv`: cross-organism validation anchors removed before comparison.
- `run_metadata.json`: hashes, methods, forbidden EC sources, caveats, and observed counts.
- `report.md`: concise observed-result report.

Comparison statuses are `exact_agreement`, `compatible_agreement`, `partial_agreement`, `conflict`, `reaction_location_missing`, `s2f_location_missing`, and `no_evidence`.

## Reaction-location evidence

For an accepted direct-EC protein, an explicit functional-location statement in its exact archived Dm28c UniProt record has priority. If that statement is absent, the collaborator bridge is followed: EC → KEGG `tcr` gene → UniProt functional-location statement. This fallback is visibly labelled `tcr_cl_brener_ec_transfer` and carries a strain/isozyme caveat.

Raw UniProt GO CC lists are not used to locate reactions. The script does not invent transport reactions or metabolite compartments.

## Interpretation limits

- An explicit historical TrEMBL EC is a database annotation and is not necessarily experimental evidence.
- KEGG `tcr` represents T. cruzi CL Brener, not Dm28c; a shared EC can have compartment-specific isozymes.
- The S2F assignments currently use the manually selected 0.1 threshold.
- Agreement is independent computational corroboration, not biological ground truth.

## Earlier exploratory route

The older `fetch`, `build`, and `run` commands remain available only to reproduce the earlier exploratory analysis. Those commands can use InterProScan GO molecular-function terms mapped through GO EC cross-references. Their outputs must not be mixed with the strict direct-EC outputs above.

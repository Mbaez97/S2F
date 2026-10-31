# Explainable Cellular Component to compartment mapping

`scripts/cc_compartments.py` converts S2F Cellular Component scores into
multi-label protein-compartment hypotheses. It does not infer an EC number,
reaction, or reaction compartment. Catalytic activity must be established from
an independent protein-to-EC or protein-to-Rhea source in a later phase.

## Mapping rules

- Only `cellular_component` terms are mapped.
- Alternate IDs and unambiguous `replaced_by` values are resolved. Ambiguous
  obsolete terms remain unresolved.
- Only `is_a` and `part_of` edges are followed upward.
- The nearest configured spatial anchor is selected. Equal-distance anchors are
  retained as multiple hypotheses.
- The compartment score is the maximum supporting S2F score, never a sum.
- Protein-complex terms without a path to a spatial anchor are classified as
  `non_spatial_cc`.
- No best compartment is forced when the evidence is unresolved or below the
  calibrated threshold.

The versioned T. cruzi-aware slim is
`conf/cc_compartments_tcruzi.json`. Reservosome calls carry an explicit
life-stage warning because the GO definition limits this structure to the
epimastigote form.

## Commands

Use the S2F interpreter used for prediction artifacts:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python \
  scripts/cc_compartments.py validate \
  --output results/cc_compartment_validation
```

`validate` performs organism-held-out threshold calibration using the local
83333 and 1111708 shared score matrices, compares all available methods, and
streams over completed `cafa3_*/prediction.df` files as regression inputs. A
non-zero exit status means the high-precision scientific gate was not met, the
benchmark was not declared clean, or a software check failed. This is an
intentional fail-closed behavior.

Generate Dm28c assignments only after `prediction.df` exists and a calibration
artifact has been reviewed:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python \
  scripts/cc_compartments.py assign \
  --prediction /run/media/marcelo_baez/HD_Disc1/.S2F/output/dm28c/prediction.df \
  --protein-list /run/media/marcelo_baez/HD_Disc1/databases/TriTrypDB-68_TcruziDm28c2014_AnnotatedProteins_clean.faa \
  --calibration results/cc_compartment_validation/calibration.json \
  --curated-evidence conf/dm28c_compartment_gold.tsv \
  --output results/dm28c_cc_compartments
```

For diagnostic work, `--threshold FLOAT` may replace `--calibration`; this is
recorded as a manual, non-calibrated threshold.

Freeze reviewed UniProt evidence with:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python \
  scripts/cc_compartments.py fetch-uniprot \
  --output results/uniprot_tcruzi_snapshot
```

Network-free runs may use `--source-tsv`. The snapshot preserves raw UniProt
location text and ECO codes; it is not automatically treated as a Dm28c gold
standard.

Audit conservative exact-sequence overlap with the Dm28c proteome:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python \
  scripts/cc_compartments.py audit-target \
  --uniprot-tsv results/uniprot_tcruzi_reviewed_2026-08-20/uniprot_reviewed_tcruzi.tsv \
  --target-fasta /run/media/marcelo_baez/HD_Disc1/databases/TriTrypDB-68_TcruziDm28c2014_AnnotatedProteins_clean.faa \
  --output results/uniprot_tcruzi_reviewed_2026-08-20/dm28c_sequence_audit
```

The audit writes all exact matches and their raw localization fields. It does
not treat similar but non-identical homologs as direct Dm28c evidence.

## Inputs and outputs

S2F predictions are headerless, three-column TSV rows:

```text
protein_id<TAB>GO_id<TAB>score
```

Optional curated evidence follows
`conf/dm28c_compartment_gold_template.tsv`. `assertion` must be `in` or
`not_in`; only an explicit `not_in` assertion creates a conflict. A positive
annotation in a different compartment is not interpreted as disagreement,
because proteins may be multi-localized and curated annotations may be
incomplete.

The assignment output contains:

- `protein_compartment_evidence.tsv`: CC-term resolution, ontology paths,
  thresholds, and stage flags. It preserves source-row order.
- `protein_compartments.tsv`: deterministic retained protein-compartment pairs.
- `protein_compartment_summary.tsv`: deterministic one-row-per-protein status,
  including below-threshold and unresolved proteins.
- `report.md` and `run_metadata.json`: interpretation, counts, exact paths,
  hashes, interpreter, and generation time.

The validation output additionally includes `calibration.json`,
`validation_metrics.tsv`, `validation_predictions.tsv`, and
`cafa3_regression.json`. CAFA3 artifacts establish parser, ontology, scale, and
determinism behavior only. Biological performance comes from held-out labeled
data, and a Dm28c-specific claim requires a blinded, stage-aware expert set.

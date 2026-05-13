# Foldseek GO Transfer Overview

## Purpose

The old structural script used this chain:

1. Foldseek search
2. target UniProt accession -> InterPro
3. InterPro -> GO
4. aggregate GO assignments

The new script removes the InterPro dependency and instead uses:

1. Foldseek search
2. accepted SwissProt hit -> experimental GO terms from GOA
3. GO up-propagation
4. aggregate GO assignments

This keeps the existing structural similarity filtering logic but changes the
annotation transfer source to experimental GO terms from SwissProt proteins.

## Current Files

- old script path: `scripts/foldseek_interpro2go.py`
- current script path: `foldseek.py`
- old instruction path: `scripts/foldseek_interpro2go_INSTRUCTIONS.md`
- current instruction path: `doc/foldseek_INSTRUCTIONS.md`

The old script and old instruction file were intentionally removed from the
active workflow. The active script is root-level `foldseek.py` so it can import
the existing S2F packages without path hacks.

## Current Script Behavior

`foldseek.py` still supports:

- FASTA input
- direct structure input
- existing structure lookup
- ColabFold structure generation
- ProstT5 fallback
- Foldseek thresholding using:
  - `--evalue-max`
  - `--min-qcov`
  - `--min-tcov`
  - `--min-avg-tm`
  - `--max-seqs` for raw Foldseek candidate retrieval before filtering

The main functional change is in how annotations are transferred after hit
selection.

There is no `topk` selection step. `foldseek.py` asks Foldseek for raw hits
using `--max-seqs`, then applies the threshold function to every returned hit.
The default `--max-seqs 0` means "request all available prefilter hits" and is
translated internally to Foldseek's large integer limit.

Large raw Foldseek TSV files are now processed with streaming passes instead of
being loaded entirely into memory before thresholding.

## Annotation Transfer

For each accepted Foldseek hit:

- the target identifier is mapped to a UniProt accession
- the accession is looked up in the provided GOA file
- GO terms are accepted only if they survive the same GO logic used by S2F:
  - `NOT` annotations are ignored
  - obsolete GO terms are ignored
  - optional taxon blacklist filtering can be applied

The script then aggregates direct GO support per query protein and
up-propagates the assignments through the ontology.

This means Foldseek transfers experimental GO terms directly from the SwissProt
donor proteins used in the target database. It does not assign InterPro entries
and it does not use `interpro2go`.

## Scoring Modes

Two score modes are supported:

- `binary`
  - default mode
  - if any accepted hit carries a GO term, that term gets score `1.0`

- `support_fraction`
  - score is `support_hits / accepted_hits`
  - example:
    - 2 supporting hits out of 5 accepted hits -> score `0.4`

These scores are written before GO up-propagation. The final
`assignments.tsv` contains propagated `Protein`, `GO ID`, `Score` rows.

## Output Files

The script writes:

- `<query>.hits.tsv`
  - accepted Foldseek hits for one query
  - includes hit metrics and `go_count`

- `<query>.summary.json`
  - accepted hits
  - resolved downstream protein identifier
  - runtime parameters, including thresholds and `max_seqs`
  - `go_support`
  - `direct_go_scores`

- `<query>.foldseek.log`
  - streamed Foldseek command output for that query

- `assignments.tsv`
  - aggregate seed-style output for downstream S2F integration
  - columns: `Protein`, `GO ID`, `Score`

## Intended Role In S2F

The file `assignments.tsv` is the interchange point between the standalone
Foldseek script and the S2F pipeline.

It can be used in two ways:

- precomputed mode
  - run `foldseek.py` manually
  - pass the resulting `assignments.tsv` to `predict --foldseek-output`

- compute mode
  - `predict` runs `foldseek.py` automatically
  - the generated `assignments.tsv` is consumed internally

The original InterPro and HMMER seeds remain part of the prediction pipeline.
Foldseek is an additive seed controlled by `gamma`; InterPro and HMMER remain
controlled by `alpha` and `beta`. If PLM is enabled, it uses the residual
weight `1 - alpha - beta - gamma`.

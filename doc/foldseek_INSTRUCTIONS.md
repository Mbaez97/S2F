# Instructions for `foldseek.py`

## Summary

`foldseek.py` builds a Foldseek-derived GO seed for S2F by:

1. comparing input proteins against a Foldseek database built from the SwissProt donor proteins,
2. keeping every retrieved hit that passes the current similarity thresholds,
3. transferring experimental GO terms from the accepted SwissProt hits, and
4. up-propagating the transferred GO terms through the Gene Ontology.

The script no longer uses `protein2ipr`, `protein2ipr_swiss.dat`, or
`interpro2go`. It also no longer has a `topk` filter. Hit selection is controlled
only by Foldseek retrieval through `--max-seqs` and by the biological thresholds
listed below.

Large Foldseek outputs are now processed as streams. The script no longer loads
the entire raw Foldseek TSV into memory before thresholding.

## Expected Inputs

### Software

- `foldseek` must be available in `PATH`.
- `colabfold_batch` is optional and only needed with `--structure-mode colabfold`.

### Data

- A Foldseek target database for the SwissProt donor proteins used for GO transfer.
- A GOA file for those SwissProt proteins. For S2F integration, use the same filtered experimental GOA used by HMMER: `filtered_goa` from `s2f.conf`.
- A GO OBO file for up-propagation.
- Optional ProstT5 `.gguf` model if you want sequence-only Foldseek fallback.

Pass the Foldseek database prefix, not an internal component. For example, use
`/path/to/afdb/afdb`, not `/path/to/afdb/afdb_ca` or `/path/to/afdb/afdb_ss`.
If a directory is passed and it contains a database with the same basename, the
script warns and uses that database prefix automatically.

## Main Arguments

| Argument | Description |
| :-- | :-- |
| `--fastas` | One or more FASTA files. |
| `--query-structures` | One or more input PDB/CIF files. |
| `--target-db` | Required Foldseek database prefix. |
| `--precomputed-tsv` | Optional existing Foldseek TSV to parse instead of running `foldseek easy-search`. |
| `--goa` | Required GOA file used for GO transfer. |
| `--go-obo` | GO OBO file used for up-propagation. |
| `--protein-id-mode` | Output protein IDs: `uniprot` or `entire_id`. |
| `--score-mode` | GO transfer scoring: `binary` or `support_fraction`. Default is `binary`. |
| `--blacklist` | Optional taxon blacklist file, one taxon ID per line. |
| `--structure-mode` | `existing` or `colabfold` for FASTA inputs. |
| `--structures-dir` | Directory used to find or write structures. |
| `--search-recursive` | Search `structures-dir` recursively for existing structures. |
| `--prostt5-model` | Optional ProstT5 `.gguf` model or weights directory for sequence-only fallback. |
| `--gpu` | Foldseek GPU switch: `1` enables CUDA, `0` disables it. |
| `--cuda-visible-devices` | CUDA device selector for Foldseek/ProstT5, for example `0` or `0,1`. |
| `--alignment-type` | Foldseek alignment type: `0` for local 3Di+AA, `1` for TM-align re-score/rank. ProstT5 forces `0`. |
| `--evalue-max` | Maximum accepted Foldseek E-value. |
| `--min-qcov` | Minimum query coverage. |
| `--min-tcov` | Minimum target coverage. |
| `--min-avg-tm` | Minimum average TM-score when TM is available. |
| `--max-seqs` | Foldseek candidate retrieval limit before threshold filtering. Use `0` to request every available prefilter hit. |
| `--outdir` | Output directory. |

## Hit Retrieval And Filtering

`--max-seqs` is Foldseek's raw candidate retrieval limit. It is not an S2F
acceptance threshold and it is not a replacement for the removed `topk` option.

- `--max-seqs 0` is the default and is converted internally to `2147483647` so Foldseek retrieves every available prefilter hit.
- Any positive value is passed directly to Foldseek as `--max-seqs`.
- A negative value is rejected.
- After Foldseek returns raw hits, `foldseek.py` applies `--evalue-max`, `--min-qcov`, `--min-tcov`, and `--min-avg-tm` when TM values are available.
- Duplicate target identifiers are collapsed after threshold filtering by keeping the first accepted target hit.

Use `--max-seqs 0` when you want the script to consider all Foldseek hits and
let the current threshold function decide which hits are accepted.

## Streaming Behavior

When Foldseek produces very large raw TSV files, `foldseek.py` now uses a
streaming two-pass parser:

1. first pass: count raw hits per query and detect whether TM-scores are available,
2. second pass: apply the acceptance thresholds and write only accepted hits to an internal streamed file,
3. final pass: load GO terms for the accepted target accessions and generate the seed outputs.

This avoids constructing a Python object for every raw Foldseek hit. It is the
intended mode for `--max-seqs 0` runs on large target databases such as AFDB.

## Reusing A Precomputed Raw TSV

If Foldseek already finished and produced a raw TSV, but the Python stage failed
later, you can reuse that TSV with `--precomputed-tsv` and skip the search step.

Example:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python foldseek.py \
    --fastas /media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/1111708.fasta \
    --precomputed-tsv /tmp/s2f_foldseek_3yutehm9/queries_prostt5.foldseek.tsv \
    --goa /media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_goa \
    --go-obo /home/marcelo_baez/paccanaro-lab/S2F/go.obo \
    --protein-id-mode uniprot \
    --structure-mode existing \
    --structures-dir /media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/structures \
    --search-recursive \
    --evalue-max 1e-5 \
    --min-qcov 0.7 \
    --min-tcov 0.7 \
    --min-avg-tm 0.5 \
    --max-seqs 0 \
    --score-mode binary \
    --outdir /media/marcelo_baez/HD_Disc1/.S2F/output/1111708/1111708_FS
```

In this mode, `--target-db`, `--gpu`, and `--prostt5-model` are not needed for
the rescue run because the raw Foldseek search has already happened.

## Structure Modes

### Existing Structures

With `--structure-mode existing`, the script looks for a structure for each
FASTA record in `--structures-dir`.

It checks direct names such as:

- `<record>.pdb`
- `<record>.cif`
- `<record>.pdb.gz`
- `<record>.cif.gz`

When `--protein-id-mode uniprot` is used, it also searches AlphaFold-style names
such as `AF-<accession>-F*-model*.pdb` and `.cif`. Add `--search-recursive` if
the structure directory contains nested folders.

If a protein has no existing structure and `--prostt5-model` is provided, the
protein is queued for ProstT5/Foldseek sequence fallback. If no ProstT5 model is
provided, the script stops and reports example missing proteins.

### ColabFold Structures

With `--structure-mode colabfold`, the script writes or reuses structures under:

```text
<structures-dir>/colabfold_<fasta-stem>/
```

Before running ColabFold, it scans this directory for existing `.pdb` or `.cif`
ranked model files. Proteins that already have predicted structures are reused
and are not predicted again. Only missing proteins are written to the sanitized
FASTA passed to `colabfold_batch`.

ColabFold logs are streamed to the console with a `[COLABFOLD]` prefix and also
written to:

```text
<structures-dir>/colabfold_<fasta-stem>/colabfold_batch.log
```

The script also logs the ColabFold JAX backend and visible devices. If ColabFold
sees only CPU devices, the script prints a warning because prediction will be
slow until the CUDA/JAX installation is fixed.

Supported ColabFold passthrough options:

- `--colabfold-msa-mode`
- `--colabfold-num-models`
- `--colabfold-num-recycle`
- `--colabfold-use-gpu-relax`
- `--colabfold-debug-logging`

## ProstT5 And GPU Notes

`--prostt5-model` accepts either a `.gguf` file or a directory. If a directory is
provided, the script first looks for `prostt5-f16.gguf`, then `prostt5.gguf`, and
then any `*.gguf` file in that directory.

Foldseek GPU selection has two parts:

- `--gpu 1` enables Foldseek CUDA mode.
- `--cuda-visible-devices 0` selects physical GPU `0`.

Do not use `--gpu 0` when you intend to use GPU. In Foldseek, `--gpu 0` means
"disable GPU". If `--cuda-visible-devices` is provided without `--gpu`, the
script enables Foldseek GPU mode automatically with `--gpu 1`.

Foldseek GPU search requires a padded target sequence database. Create it once
from the base database prefix:

```bash
foldseek makepaddedseqdb \
    /media/marcelo_baez/HD_Disc1/databases/foldseek_db/afdb/afdb \
    /media/marcelo_baez/HD_Disc1/databases/foldseek_db/afdb/afdb_pad \
    -v 3
```

Use the base database prefix in `foldseek.py`; if GPU is enabled and
`<target-db>_pad` exists, the script switches to the padded database
automatically. Do not run `makepaddedseqdb` on `afdb_ss`; that internal component
does not contain the header information needed by Foldseek.

## Scoring Modes

- `binary`: a GO term gets score `1.0` if at least one accepted hit carries it.
- `support_fraction`: a GO term gets score `support_hits / accepted_hits`.

These direct scores are up-propagated through the GO graph before writing
`assignments.tsv`.

## Logging

The script prints timestamped progress messages for each major phase:

- input inspection and selected structure mode
- number of proteins loaded from each FASTA
- existing structure reuse or missing-structure detection
- ColabFold resume status and per-protein prediction queue
- Foldseek command execution and per-protein raw hit counts
- threshold filtering counts
- GOA loading and GO transfer counts
- final output summary

Foldseek command output is streamed to the console with a `[FOLDSEEK]` prefix and
written per query to:

```text
<outdir>/<query>.foldseek.log
```

## Examples

### FASTA input with existing structures

```bash
python3 foldseek.py \
    --fastas /path/to/query.fasta \
    --structure-mode existing \
    --structures-dir /path/to/structures \
    --target-db /path/to/foldseek_swissprot_db \
    --goa /path/to/filtered_goa \
    --go-obo /path/to/go.obo \
    --max-seqs 0 \
    --outdir ./foldseek_results
```

### ProstT5 sequence fallback with GPU

```bash
python3 foldseek.py \
    --fastas /media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/1111708.fasta \
    --structure-mode existing \
    --target-db /media/marcelo_baez/HD_Disc1/databases/foldseek_db/afdb/afdb \
    --goa /media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_goa \
    --go-obo /home/marcelo_baez/paccanaro-lab/S2F/go.obo \
    --prostt5-model /media/marcelo_baez/HD_Disc1/protsT5/weights/prostt5-f16.gguf \
    --alignment-type 0 \
    --gpu 1 \
    --cuda-visible-devices 0 \
    --max-seqs 0 \
    --outdir ./foldseek_results/1111708_prostt5_all_hits
```

### FASTA input with support-fraction scoring

```bash
python3 foldseek.py \
    --fastas /path/to/query.fasta \
    --target-db /path/to/foldseek_swissprot_db \
    --goa /path/to/filtered_goa \
    --go-obo /path/to/go.obo \
    --score-mode support_fraction \
    --max-seqs 0 \
    --outdir ./foldseek_results_fractional
```

### Direct structure input

```bash
python3 foldseek.py \
    --query-structures /path/to/query.pdb \
    --target-db /path/to/foldseek_swissprot_db \
    --goa /path/to/filtered_goa \
    --go-obo /path/to/go.obo \
    --max-seqs 0 \
    --outdir ./foldseek_results
```

## Outputs

The script writes:

1. `<query>.hits.tsv`
   - accepted Foldseek hits for one query
   - includes similarity values and how many GO terms were available on each hit

2. `<query>.summary.json`
   - accepted hit list
   - query protein ID used in downstream S2F steps
   - `parameters`, including `max_seqs`, thresholds, alignment type, and scoring mode
   - `go_support`: GO-term support counts across accepted hits
   - `direct_go_scores`: direct transferred GO scores before GO up-propagation

3. `<query>.foldseek.log`
   - streamed Foldseek output for that query

4. `assignments.tsv`
   - aggregate Foldseek GO assignments across all queries
   - columns: `Protein`, `GO ID`, `Score`
   - this is the file consumed by `S2F.py predict --foldseek-output ...`

## S2F Integration

For testing with a precomputed Foldseek seed:

```bash
python3 foldseek.py ... --outdir ./foldseek_seed
python3 S2F.py predict \
    --foldseek-output ./foldseek_seed/assignments.tsv \
    --alpha 0.7 \
    --beta 0.1 \
    --gamma 0.2 \
    ...
```

For automatic execution from S2F:

```bash
python3 S2F.py predict \
    --foldseek-output compute \
    --foldseek-target-db /path/to/foldseek_swissprot_db \
    --foldseek-max-seqs 0 \
    --foldseek-gpu 1 \
    --foldseek-cuda-visible-devices 0 \
    --alpha 0.7 \
    --beta 0.1 \
    --gamma 0.2 \
    ...
```

When using `S2F.py predict --foldseek-output compute`, the predict command runs
the root-level `foldseek.py` script automatically and consumes the generated
`assignments.tsv`.

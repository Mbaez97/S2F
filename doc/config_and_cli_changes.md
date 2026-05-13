# Config And CLI Changes

## New `predict` CLI Arguments

The `predict` command now exposes Foldseek and PLM seed options.

### Seed selection and weighting

- `--foldseek-output`
  - `skip`: disable Foldseek
  - `compute`: run `foldseek.py`
  - file path: load a precomputed `assignments.tsv`

- `--plm-output`
  - `skip`: disable PLM
  - `compute`: run `plm.py`
  - file path: load a precomputed PLM `assignments.tsv`

- `--alpha`
  - weight for InterPro

- `--beta`
  - weight for HMMER

- `--gamma`
  - weight for Foldseek

The PLM weight is the residual:

`delta = 1 - alpha - beta - gamma`

### Foldseek runtime arguments

- `--foldseek-target-db`: Foldseek target database prefix used by `foldseek.py`.
- `--foldseek-structure-mode`: `existing` or `colabfold`.
- `--foldseek-structures-dir`: directory used to find or write query structures.
- `--foldseek-precomputed-tsv`: existing raw Foldseek TSV to parse instead of running Foldseek search again.
- `--foldseek-search-recursive`: search the structure directory recursively.
- `--foldseek-prostt5-model`: ProstT5 `.gguf` model or weights directory.
- `--foldseek-gpu`: Foldseek GPU switch. `1` enables CUDA; `0` disables it.
- `--foldseek-cuda-visible-devices`: physical CUDA device selector, for example `0`.
- `--foldseek-alignment-type`: Foldseek alignment type, `0` or `1`.
- `--foldseek-evalue-max`: maximum accepted Foldseek E-value.
- `--foldseek-min-qcov`: minimum accepted query coverage.
- `--foldseek-min-tcov`: minimum accepted target coverage.
- `--foldseek-min-avg-tm`: minimum accepted average TM-score when TM values exist.
- `--foldseek-max-seqs`: Foldseek raw candidate retrieval limit; `0` requests all available prefilter hits.
- `--foldseek-score-mode`: Foldseek GO transfer scoring mode, `binary` or `support_fraction`.

### PLM runtime arguments

- `--plm-target-fasta`: SwissProt FASTA used as the embedding target; defaults to `filtered_sprot`.
- `--plm-model-name`: ESM model name or local path; default is `facebook/esm1b_t33_650M_UR50S`.
- `--plm-model-dir`: model cache directory.
- `--plm-embeddings-dir`: reusable target embedding cache directory.
- `--plm-device`: `auto`, `cpu`, `cuda`, `cuda:0`, etc.
- `--plm-knn-k`: number of nearest SwissProt embeddings; default `10`.
- `--plm-long-sequence-mode`: `sliding_mean`, `truncate`, or `skip`.
- `--plm-long-window-size`: residue window size for long proteins; default `1022`.
- `--plm-long-overlap`: overlap between long-protein windows; default `128`.
- `--plm-score-mode`: currently `all_ones`.
- `--plm-batch-tokens`: approximate token budget for embedding batches.
- `--plm-query-chunk-size`: query chunk size for exact cosine KNN.
- `--plm-local-files-only`: require the ESM model to be present locally.
- `--plm-precomputed-target-embeddings`: reuse a target embedding cache.
- `--plm-precomputed-query-embeddings`: reuse a query embedding cache.
- `--plm-force-target-embeddings`: recompute target embeddings.
- `--plm-force-query-embeddings`: recompute query embeddings.

## New Run-Config Fields

### `[seeds]`

Added fields:

- `foldseek_output`
- `plm_output`
- `alpha`
- `beta`
- `gamma`

### `[foldseek]`

Added fields:

- `target_db`: Foldseek target database prefix.
- `structure_mode`: `existing` or `colabfold`.
- `structures_dir`: query structure directory.
- `precomputed_tsv`: optional raw Foldseek TSV to reuse in compute mode.
- `search_recursive`: recursive lookup for existing structures.
- `prostt5_model`: ProstT5 `.gguf` model or weights directory.
- `gpu`: Foldseek GPU switch. `1` enables CUDA; `0` disables it.
- `cuda_visible_devices`: physical CUDA device selector.
- `alignment_type`: Foldseek alignment type.
- `evalue_max`: maximum accepted Foldseek E-value.
- `min_qcov`: minimum accepted query coverage.
- `min_tcov`: minimum accepted target coverage.
- `min_avg_tm`: minimum accepted average TM-score.
- `max_seqs`: Foldseek raw candidate retrieval limit; `0` requests all hits.
- `score_mode`: `binary` or `support_fraction`.

There is no `topk` configuration anymore. The previous top-k cap was removed so
all raw hits returned by Foldseek can be evaluated by the threshold function.
Use `max_seqs = 0` to request every available Foldseek prefilter hit.

If `precomputed_tsv` is set, `predict` still runs `foldseek.py`, but the script
parses that TSV directly and skips `foldseek easy-search`.

### `[plm]`

Added fields:

- `target_fasta`: SwissProt target FASTA; empty means `filtered_sprot`.
- `model_name`: ESM model name or local model path.
- `model_dir`: model cache directory; empty means `<installation>/data/PLM/models`.
- `embeddings_dir`: reusable embedding cache directory; empty means `<installation>/data/PLM/embeddings`.
- `device`: `auto`, `cpu`, `cuda`, `cuda:0`, etc.
- `knn_k`: number of nearest SwissProt embeddings.
- `long_sequence_mode`: `sliding_mean`, `truncate`, or `skip`.
- `long_window_size`: residue window size for ESM1b.
- `long_overlap`: overlap between long-sequence windows.
- `score_mode`: currently `all_ones`.
- `batch_tokens`: approximate embedding batch token budget.
- `query_chunk_size`: exact KNN query chunk size.
- `local_files_only`: do not download the model.
- `precomputed_target_embeddings`: optional target embedding cache.
- `precomputed_query_embeddings`: optional query embedding cache.
- `force_target_embeddings`: recompute target embeddings.
- `force_query_embeddings`: recompute query embeddings.

### `[graphs]`

Configured graph paths are honored when Foldseek or PLM is enabled:

- `combined_graph`: if this points to an existing sparse `.npz`, `predict` loads
  it directly for diffusion and skips graph collection/homology loading.
- `graph_collection`: if `combined_graph = compute`, this can point to an
  existing pickled collection file such as
  `<installation>/graphs/collection/<alias>`.
- `homology_graph`: if `combined_graph = compute`, this can point to an existing
  pickled homology graph such as `<installation>/graphs/homology/<alias>`.
- `orthologs_alias`: optional alias used to read/write reciprocal-best-hit
  ortholog files during graph collection. This allows a new blacklisted graph
  collection to reuse orthologs computed by a previous run alias.

For an incremental PLM experiment where the old combined graph should be reused,
set all three graph paths to the artifacts from the previous run and set
`recompute_orthologs = false`.

## Updated Sample Configs

The following config files were updated to include the new sections:

- `run.conf`
- `cafa3_testset.conf`
- `1111708_plm_full.conf`
- `1111708_plm_reuse_existing.conf`

Defaults currently document the testing-safe setup:

- `foldseek_output = skip`
- `plm_output = skip`
- `alpha = 0.9`
- `beta = 0.1`
- `gamma = 0.0`
- `max_seqs = 0`
- `precomputed_tsv =`
- `score_mode = binary`

## Installation Config

`s2f.conf` does not need Foldseek-specific database keys at this stage. The
Foldseek target DB is run-specific and belongs in `[foldseek].target_db` or
`--foldseek-target-db`.

`s2f.conf` still matters for Foldseek because `predict --foldseek-output compute`
passes `[databases].filtered_goa` to `foldseek.py`. That GOA file should be the
same filtered experimental SwissProt GOA used by HMMER so Foldseek transfers GO
terms from the same annotation source.

`s2f.conf` also matters for PLM because `predict --plm-output compute` uses
`filtered_sprot` as the default PLM target FASTA and `filtered_goa` as the
annotation transfer source.

## Validation Rules

The implementation enforces:

- weights cannot be negative
- `alpha + beta + gamma <= 1`
- `delta = 1 - alpha - beta - gamma`
- if `gamma > 0`, then `foldseek_output` cannot be `skip`
- if `delta > 0`, then `plm_output` cannot be `skip`
- if `foldseek_output = compute`, then:
  - `foldseek.target_db` must be configured
  - `filtered_goa` must exist
- if `plm_output = compute`, then:
  - the PLM target FASTA must exist
  - `filtered_goa` must exist
  - `knn_k`, `batch_tokens`, and `query_chunk_size` must be greater than zero
  - `long_overlap` must be smaller than `long_window_size`
- if `foldseek.max_seqs` is negative, `foldseek.py` rejects it
- if GPU mode is enabled and no padded target DB exists, `foldseek.py` reports the exact `foldseek makepaddedseqdb` command to create it
- if `foldseek.precomputed_tsv` is set, that file must exist

## Blacklists

`hmmer_blacklist` is a GOA taxon blacklist. When it points to a file, HMMER,
Foldseek, and PLM use it during GO-transfer from SwissProt/GOA annotations.
It does not change the HMMER or Foldseek search itself; it only prevents
annotations from blacklisted taxons from being transferred.

`transfer_blacklist` is a STRING organism blacklist. It is used by graph
collection transfer to skip blacklisted STRING organisms when transferring
links. It does not require recomputing reciprocal-best-hit ortholog files if
those files already exist for the same alias; the link-transfer step can reuse
them and simply skip blacklisted organisms.

## Auto-Run Contract

When `foldseek_output = compute`, `predict` runs:

- the root-level `foldseek.py`

and passes:

- the input FASTA
- the configured Foldseek database
- `filtered_goa`
- `obo`
- the selected FASTA ID parser
- the optional HMMER blacklist file as the Foldseek blacklist
- Foldseek thresholds and runtime options from `[foldseek]`
- `max_seqs`; the default `0` means all returned Foldseek hits are considered before threshold filtering

If `precomputed_tsv` is configured, `predict` also passes that file to
`foldseek.py`, and the raw Foldseek search step is skipped.

This keeps the Foldseek GO transfer aligned with the rest of the S2F setup.

When `plm_output = compute`, `predict` runs:

- the root-level `plm.py`

and passes:

- the input FASTA
- `filtered_sprot` or `[plm].target_fasta`
- `filtered_goa`
- `obo`
- the selected FASTA ID parser
- the optional HMMER blacklist file as the PLM taxon blacklist
- ESM1b, KNN, long-sequence, cache, and runtime options from `[plm]`

## GPU Configuration Notes

In both CLI and config files, `gpu = 1` enables Foldseek CUDA mode. Use
`cuda_visible_devices = 0` to select GPU device `0`. Do not use `gpu = 0` when
you want CUDA because Foldseek interprets `--gpu 0` as GPU disabled.

For GPU searches, create the padded database once from the base database prefix:

```bash
foldseek makepaddedseqdb /path/to/afdb/afdb /path/to/afdb/afdb_pad -v 3
```

The source must be the base DB prefix, not `afdb_ss`.

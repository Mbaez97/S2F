# PLM Seed Instructions

## Purpose

`plm.py` builds a protein language model seed for S2F.

The workflow is:

1. Embed the filtered SwissProt proteins with ESM1b.
2. Embed the input FASTA proteins with the same model.
3. Find the `K` nearest SwissProt embeddings for each input protein.
4. Transfer all experimental GO annotations from those neighbors.
5. Up-propagate the transferred GO terms through the GO structure.
6. Write an `assignments.tsv` file that S2F converts into a sparse seed matrix.

The target SwissProt FASTA should be the same `filtered_sprot` file used by the
HMMER seed. Target identifiers are always resolved as UniProt accessions so
they match GOA. The query FASTA uses the configured S2F protein ID parser.

## Dependencies

The PLM seed has optional dependencies that are not required for the rest of
S2F:

```bash
pip install -r requirements-plm.txt
```

The current S2F environment must be able to import `torch` and `transformers`
before ESM1b embeddings can be computed. If the embeddings are already cached,
`plm.py` can run the KNN and GO-transfer stages without loading the model.

## Standalone Command

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python plm.py \
    --fastas /media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/1111708.fasta \
    --target-fasta /media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_sprot \
    --goa /media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_goa \
    --go-obo /home/marcelo_baez/paccanaro-lab/S2F/go.obo \
    --protein-id-mode uniprot \
    --model-name facebook/esm1b_t33_650M_UR50S \
    --model-dir /media/marcelo_baez/HD_Disc1/.S2F/data/PLM/models \
    --embeddings-dir /media/marcelo_baez/HD_Disc1/.S2F/data/PLM/embeddings \
    --device auto \
    --knn-k 10 \
    --long-sequence-mode sliding_mean \
    --long-window-size 1022 \
    --long-overlap 128 \
    --score-mode all_ones \
    --outdir /media/marcelo_baez/HD_Disc1/.S2F/output/1111708_plm/1111708_plm_PLM
```

## ESM1b Cache Behavior

`plm.py` caches embeddings in directories containing:

- `ids.tsv`
- `embeddings.npy`
- `meta.json`

Target SwissProt embeddings are reusable across experiments and are written
under `[plm].embeddings_dir`. Query embeddings are written under the PLM output
directory for the current run.

The cache metadata includes the model name, FASTA path, FASTA size/mtime,
protein ID mode, and long-sequence settings. If these settings change, the
cache is recomputed unless a compatible precomputed cache is provided.

## Long Sequences

ESM1b has a practical residue limit of about `1022`.

The default mode is:

```text
long_sequence_mode = sliding_mean
long_window_size = 1022
long_overlap = 128
```

For long proteins, the script embeds overlapping windows and averages the window
embeddings into one protein embedding.

## Output Files

The PLM output directory contains:

- `neighbors.tsv`: nearest SwissProt neighbors for each query protein.
- `assignments.tsv`: propagated seed-style GO assignments with `Protein`, `GO ID`, and `Score`.
- `<protein>.summary.json`: per-query neighbor and direct GO-transfer summary.
- `query_embeddings/`: cached query embeddings for this run.

`assignments.tsv` is the file consumed by `S2F.py predict`.

## S2F Integration

Enable PLM in a run config with:

```ini
[seeds]
plm_output = compute
alpha = 0.6
beta = 0.1
gamma = 0.2

[plm]
target_fasta = /media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_sprot
model_name = facebook/esm1b_t33_650M_UR50S
model_dir = /media/marcelo_baez/HD_Disc1/.S2F/data/PLM/models
embeddings_dir = /media/marcelo_baez/HD_Disc1/.S2F/data/PLM/embeddings
device = auto
knn_k = 10
long_sequence_mode = sliding_mean
long_window_size = 1022
long_overlap = 128
score_mode = all_ones
batch_tokens = 4096
query_chunk_size = 64
local_files_only = false
```

The PLM weight is not a separate config value. It is:

```text
delta = 1 - alpha - beta - gamma
```

If `delta > 0`, `plm_output` must not be `skip`.

## Full Experiment Config

The prepared four-seed experiment config is:

```text
conf/1111708_plm_full.conf
```

Run it from the repository root with:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python S2F.py predict \
    --run-config conf/1111708_plm_full.conf \
    --unattended
```

That config expects the Foldseek experiment to have produced:

```text
/media/marcelo_baez/HD_Disc1/.S2F/output/1111708/1111708_FS/assignments.tsv
```

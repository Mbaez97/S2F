# 1111708 Foldseek Full Experiment

## Purpose

This run config executes the full S2F prediction workflow with all three seed
sources enabled:

- InterPro seed
- HMMER seed
- Foldseek GO-transfer seed

The config file is:

```text
conf/1111708_foldseek_full.conf
```

If Foldseek search already ran and produced a raw TSV that should be reused, use
this recovery config instead:

```text
conf/1111708_foldseek_recover.conf
```

## Inputs

- FASTA: `/media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/1111708.fasta`
- S2F install config: `/home/marcelo_baez/paccanaro-lab/S2F/s2f.conf`
- GO OBO: `/home/marcelo_baez/paccanaro-lab/S2F/go.obo`
- Foldseek target DB: `/media/marcelo_baez/HD_Disc1/databases/foldseek_db/afdb/afdb`
- ProstT5 model: `/media/marcelo_baez/HD_Disc1/protsT5/weights/prostt5-f16.gguf`
- GOA source for Foldseek transfer: `filtered_goa` from `s2f.conf`

## Seed Weights

The configured weights are:

- `alpha = 0.7` for InterPro
- `beta = 0.1` for HMMER
- `gamma = 0.2` for Foldseek

These weights are used directly. The PLM residual weight is
`1 - alpha - beta - gamma`, which is `0.0` for this Foldseek-only experiment.

## Foldseek Settings

The Foldseek seed is computed automatically by `S2F.py predict` using the
root-level `foldseek.py` script.

Important settings:

- `structure_mode = existing`
- `prostt5_model` is configured, so proteins without existing structures are handled through ProstT5/Foldseek sequence mode
- `gpu = 1` enables Foldseek CUDA mode
- `cuda_visible_devices = 0` selects physical GPU `0`
- `max_seqs = 0` requests all available Foldseek prefilter hits before threshold filtering
- `score_mode = binary` assigns score `1.0` when at least one accepted hit transfers a GO term

The target DB is configured as the base prefix `afdb`. If GPU mode is enabled
and `afdb_pad` exists, `foldseek.py` switches to the padded DB automatically.

## Command

From the repository root:

```bash
python3 S2F.py predict \
    --run-config conf/1111708_foldseek_full.conf \
    --unattended
```

## Recovery Command

If the Foldseek search already completed and left this TSV behind:

```text
/tmp/s2f_foldseek_3yutehm9/queries_prostt5.foldseek.tsv
```

run the recovery config instead:

```bash
python3 S2F.py predict \
    --run-config conf/1111708_foldseek_recover.conf \
    --unattended
```

That config reuses the raw TSV through `[foldseek].precomputed_tsv` and skips the
Foldseek search stage.

Optional command with a terminal log:

```bash
python3 S2F.py predict \
    --run-config conf/1111708_foldseek_full.conf \
    --unattended \
    2>&1 | tee 1111708_foldseek_full.log
```

## Expected Artifacts

The exact output paths depend on the installation directory from `s2f.conf`,
currently `/media/marcelo_baez/HD_Disc1/.S2F`.

Expected Foldseek-related artifacts include:

- `output/1111708/1111708_FS/assignments.tsv`
- `output/1111708/1111708_FS/<query>.foldseek.log`
- `seeds/foldseek/1111708.seed.npz`
- `output/1111708/foldseek_seed.diffusion.npz`

The original InterPro and HMMER seed outputs are still generated and used.

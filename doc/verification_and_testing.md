# Verification And Testing

## Checks Already Run

The following smoke checks were run after the implementation:

- `python3 -m py_compile S2F.py commands/Predict.py commands/ExtractSeeds.py foldseek.py plm.py`
- `python3 S2F.py predict -h`
- `python3 foldseek.py -h`
- `python3 plm.py -h`
- `python3 foldseek.py -h | rg -n "max-seqs|topk|usage"`
- `python3 S2F.py predict -h | rg -n "foldseek-max-seqs|foldseek-topk"`

These checks verified:

- Python syntax for the changed files
- `predict` CLI wiring
- `foldseek.py` CLI wiring
- `plm.py` CLI wiring
- `max-seqs` is exposed and `topk` is not exposed

## Environment Issue Encountered

While testing, this environment initially failed importing `pandas` inside
`foldseek.py` because of a local `numpy`/`pandas` binary mismatch.

To avoid breaking simple help checks:

- `pandas` and `GeneOntology` imports in `foldseek.py` were moved
  inside `main()` after argument parsing

This means:

- `python3 foldseek.py -h` works even if the scientific stack is
  currently broken
- full execution of the script still requires a working runtime environment

The PLM seed has an additional optional dependency set. The S2F conda
environment used during implementation could run PLM KNN/GO-transfer from
precomputed embeddings, but it did not have `torch` installed for ESM1b
embedding computation. Install `requirements-plm.txt` before computing real
ESM1b embeddings.

## Manual End-To-End Checks Still Recommended

The following were not run automatically because they require external data and
tools:

- full Foldseek search against the target database
- automatic `predict --foldseek-output compute`
- automatic `predict --plm-output compute` with real ESM1b embeddings
- complete four-seed diffusion on a real organism dataset

## Recommended Manual Test Workflow

### 1. Standalone Foldseek seed generation

Run:

```bash
python3 foldseek.py \
    --fastas /path/to/query.fasta \
    --target-db /path/to/foldseek_db \
    --goa /path/to/filtered_goa \
    --go-obo /path/to/go.obo \
    --score-mode binary \
    --max-seqs 0 \
    --outdir ./foldseek_test
```

Check:

- `./foldseek_test/assignments.tsv` exists
- per-query `<query>.foldseek.log` files exist
- headers contain `Protein`, `GO ID`, `Score`
- protein identifiers match the intended FASTA parsing mode
- the run log reports raw hit counts and accepted hit counts per query

### 2. Precomputed Foldseek integration in `predict`

Run:

```bash
python3 S2F.py predict \
    --foldseek-output ./foldseek_test/assignments.tsv \
    --alpha 0.7 \
    --beta 0.1 \
    --gamma 0.2 \
    ...
```

Check:

- `seeds/foldseek/<alias>.seed.npz`
- `output/<alias>/foldseek_seed.diffusion.npz`
- final prediction file exists

### 3. Auto-run Foldseek integration in `predict`

Run:

```bash
python3 S2F.py predict \
    --foldseek-output compute \
    --foldseek-target-db /path/to/foldseek_db \
    --foldseek-score-mode binary \
    --foldseek-max-seqs 0 \
    --alpha 0.7 \
    --beta 0.1 \
    --gamma 0.2 \
    ...
```

Check:

- `output/<alias>/<alias>_FS/assignments.tsv`
- `seeds/foldseek/<alias>.seed.npz`
- graph and diffusion outputs are regenerated

### 4. ProstT5 GPU manual check

Create the padded target DB once if it does not already exist:

```bash
foldseek makepaddedseqdb \
    /media/marcelo_baez/HD_Disc1/databases/foldseek_db/afdb/afdb \
    /media/marcelo_baez/HD_Disc1/databases/foldseek_db/afdb/afdb_pad \
    -v 3
```

Then run:

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

Check:

- the log shows `Foldseek CUDA_VISIBLE_DEVICES=0`
- the log shows `Foldseek GPU switch: --gpu 1`
- if `/media/marcelo_baez/HD_Disc1/databases/foldseek_db/afdb/afdb_pad` exists, the script warns that it is using the padded target DB
- Foldseek output lines are visible in the console with the `[FOLDSEEK]` prefix

### 5. Precomputed raw TSV recovery

If Foldseek search already finished and left a raw TSV behind, test the recovery
path with:

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

Check:

- the log reports a first streaming pass and a second streaming pass
- `assignments.tsv` is created in the output directory
- no raw-hit Python list is built, so the process should no longer die with `SIGKILL` on very large TSVs

### 6. Standalone PLM seed generation

Run:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python plm.py \
    --fastas /path/to/query.fasta \
    --target-fasta /path/to/filtered_sprot \
    --goa /path/to/filtered_goa \
    --go-obo /path/to/go.obo \
    --protein-id-mode uniprot \
    --model-dir /path/to/.S2F/data/PLM/models \
    --embeddings-dir /path/to/.S2F/data/PLM/embeddings \
    --device auto \
    --knn-k 10 \
    --outdir ./plm_test
```

Check:

- `./plm_test/neighbors.tsv`
- `./plm_test/assignments.tsv`
- `./plm_test/query_embeddings/embeddings.npy`
- per-query `<protein>.summary.json` files

### 7. Four-seed `predict` integration

Run:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python S2F.py predict \
    --run-config conf/1111708_plm_full.conf \
    --unattended
```

Check:

- `seeds/plm/1111708_plm.seed.npz`
- `output/1111708_plm/plm_seed.diffusion.npz`
- `output/1111708_plm/prediction.df`

## Expected Failure Cases To Test

- `gamma > 0` with `foldseek_output = skip`
- residual PLM weight greater than zero with `plm_output = skip`
- `alpha + beta + gamma > 1`
- `foldseek_output = compute` without a configured target DB
- `plm_output = compute` without PLM dependencies or without a valid target FASTA
- assignment files with protein IDs that do not match the FASTA parser mode
- invalid or empty `assignments.tsv`
- GPU mode with an unpadded target DB
- `max_seqs < 0`

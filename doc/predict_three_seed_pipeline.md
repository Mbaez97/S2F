# Predict Pipeline Changes

## Summary

`commands/Predict.py` now treats Foldseek and PLM as additive seed sources in
addition to InterPro and HMMER.

The supported seeds are:

- InterPro seed
- HMMER seed
- Foldseek seed
- PLM seed

The original InterPro and HMMER flows are preserved. Foldseek and PLM are
additive and optional.

## Foldseek Modes

`predict` supports three Foldseek modes through `foldseek_output`:

- `skip`
  - Foldseek is disabled
  - this preserves the legacy two-seed behavior

- path to `assignments.tsv`
  - Foldseek assignments were generated outside `predict`
  - `predict` reads the file and converts it into a sparse seed

- `compute`
  - `predict` runs the root-level `foldseek.py`
  - the generated `assignments.tsv` is consumed automatically

The old InterPro and HMMER seeds are still produced exactly as before unless
their own output options are changed. Foldseek is additive and optional.

`predict` supports three PLM modes through `plm_output`:

- `skip`
  - PLM is disabled
  - this preserves the previous behavior when the residual PLM weight is zero

- path to `assignments.tsv`
  - PLM assignments were generated outside `predict`
  - `predict` reads the file and converts it into a sparse seed

- `compute`
  - `predict` runs the root-level `plm.py`
  - the generated `assignments.tsv` is consumed automatically

## Seed Conversion

Foldseek uses the same general TSV-to-seed conversion path as any assignment
file with:

- `Protein`
- `GO ID`
- optional `Score`

This conversion:

- keeps only positive scores
- collapses duplicate `(Protein, GO ID)` pairs using the maximum score
- matches protein identifiers against the FASTA-derived protein index
- matches GO IDs against the loaded ontology

## Weighting

Three user-facing coefficients are used directly:

- `alpha` for InterPro
- `beta` for HMMER
- `gamma` for Foldseek

PLM uses the residual coefficient:

- `delta = 1 - alpha - beta - gamma`

The implementation requires `alpha + beta + gamma <= 1`. If `delta > 0`,
`plm_output` must not be `skip`.

## Combined Graph Target

The graph-combination target is no longer just the InterPro seed.

It is now:

`target_seed = alpha * interpro_seed + beta * hmmer_seed + gamma * foldseek_seed + delta * plm_seed`

This means Foldseek and PLM can contribute to the target used by graph
combination.

## Final Prediction

Each available seed is diffused independently:

- InterPro diffusion
- HMMER diffusion
- Foldseek diffusion
- PLM diffusion

The final prediction is then:

`prediction = alpha * ip_diff + beta * hmmer_diff + gamma * foldseek_diff + delta * plm_diff`

## Clamp Behavior

When a GOA clamp is provided:

- raw InterPro seed is clamped
- raw HMMER seed is clamped
- raw Foldseek seed is clamped if Foldseek is enabled
- raw PLM seed is clamped if PLM is enabled
- diffused outputs are clamped again before final blending

Clamp loading is only performed when diffusion is actually being recomputed.

## Cache Behavior

Without Foldseek:

- the original cache behavior stays in place
- existing InterPro/HMMER diffusions can be reused

With Foldseek or PLM enabled:

- `Predict.check_progress()` forces diffusion to run so the added seed can
  contribute to the final prediction
- if `[graphs].combined_graph` points to an existing sparse `.npz`, that graph
  is loaded directly and graph collection/homology construction is skipped
- if `[graphs].combined_graph = compute`, `predict` can still reuse configured
  `[graphs].graph_collection` and `[graphs].homology_graph` paths while
  rebuilding only the combined graph for the new seed mixture

Seed files themselves may still be reused if they already exist.

## Foldseek Runtime Parameters

When `foldseek_output = compute`, `predict` forwards the `[foldseek]` runtime
configuration to `foldseek.py`, including:

- target database path
- optional precomputed raw Foldseek TSV
- structure mode and structure directory
- recursive structure lookup
- ProstT5 model path
- Foldseek GPU switch and CUDA device selector
- Foldseek alignment type
- E-value, query coverage, target coverage, and average TM-score thresholds
- `max_seqs`
- scoring mode

There is no `topk` runtime parameter. `max_seqs` is Foldseek's raw retrieval
limit before threshold filtering. The default `max_seqs = 0` asks Foldseek for
all available prefilter hits so the threshold function can evaluate all of them.

## PLM Runtime Parameters

When `plm_output = compute`, `predict` forwards the `[plm]` runtime
configuration to `plm.py`, including:

- SwissProt target FASTA
- ESM1b model name and cache directory
- reusable target embedding cache directory
- device selection
- `knn_k`
- long-sequence windowing settings
- scoring mode
- embedding batch size and KNN query chunk size
- optional precomputed embedding caches

## Additional Seed Artifacts

Additional seed outputs now live at:

- sparse seed: `seeds/foldseek/<alias>.seed.npz`
- diffusion: `output/<alias>/foldseek_seed.diffusion`
- sparse seed: `seeds/plm/<alias>.seed.npz`
- diffusion: `output/<alias>/plm_seed.diffusion`

`commands/ExtractSeeds.py` exports text versions of Foldseek and PLM seeds when
the corresponding seed files exist.

# Fast resume for large S2F prediction runs

Large STRING installations can spend most of their post-ortholog runtime
parsing the global interaction archive and materialising the dense homology
graph. The prediction workflow provides opt-in engines for these stages:

```ini
[performance]
collection_engine = native_filter
homology_engine = vectorized
write_diffusion_tsv = false
prediction_chunk_rows = 1000000
```

`native_filter` applies the same ortholog thresholds as the legacy collection
builder, then uses `pigz` and GNU fixed-string filtering to discard STRING rows
that cannot contain transferable endpoints before pandas sees them. Exact
endpoint and graph-score checks remain in Python. It requires both commands on
`PATH`.

`vectorized` preserves the legacy homology transformation and normalization,
but replaces Python pairwise loops and the dense pandas pickle with a sparse
`.npz` checkpoint.

When `write_diffusion_tsv` is false, the restartable diffusion `.npz` files are
retained but the redundant text copies are omitted. `prediction.df` is always
written; a positive `prediction_chunk_rows` writes it in bounded chunks and
publishes it atomically.

All settings default to the legacy behavior. A resume configuration should use
the same alias, FASTA, ontology, graph thresholds, and seed weights as the
original run, and set `recompute_orthologs = false` only after the reciprocal
BLAST worker pool has completed.

# Clean PLM neighbour-embedding explorer

## Purpose

The **Neighbour geometry** page in the ESM GO Embedding Explorer is an
inspection view for the Clean PLM benchmark. It places the shared benchmark
test proteins beside the Swiss-Prot donor proteins that the neighbour-based
methods can actually use. It is for checking whether a test protein's KNN,
fixed-radius, or Gaussian-KDE donors form a meaningful local neighbourhood in
the ESM-1b embedding representation; it is not an additional prediction
method or a claim of experimental validation.

## Joint-projection rule

For every projection method, the exporter builds one combined matrix containing
the test proteins and the displayed Swiss-Prot donors, L2-normalizes its rows,
and fits one model to that matrix. It never fits separate test and donor
projections. Therefore, a test point and a donor point in the same PCA, UMAP,
or t-SNE view use the same coordinate system.

PCA is fitted directly to the joint 1,280-dimensional ESM-1b embeddings. To
keep UMAP and t-SNE reproducible within the local memory budget, the exporter
first fits a **joint 50-component PCA** to that same combined matrix, then fits
UMAP or t-SNE to those shared 50-dimensional coordinates. This preprocessing
is shown in the frontend. It does not split the datasets or transform them
independently.

## Display cohort and neighbour overlays

The full Clean PLM source cache has 83,003 Swiss-Prot proteins. Before every
neighbour method selects donors, the benchmark excludes all 196 shared
evaluation accessions and the separate close-taxon blacklist belonging to the
query's test organism. In the current source snapshot, this removes 3,617
proteins for taxon 83333 (79,386 eligible donors), 204 for 223283 (82,799), and
283 for 1111708 (82,720). The exporter reuses exactly those per-organism donor
boundaries and verifies the blacklist-file hashes in the benchmark audit. The
current generated view contains:

- all 196 shared test proteins: 160 from taxon 83333, 3 from taxon 223283,
  and 33 from taxon 1111708;
- every Swiss-Prot donor selected by at least one displayed method, plus each
  query's closest donor candidates;
- no random background donors in this run, so no method-selected donor is
  displaced merely to make the visualization smaller.

When a user selects a test protein and an overlay, the page draws edges to the
exact donor accessions selected in the original high-dimensional cosine search:

- KNN with `k = 3, 5, 7, 10`;
- fixed-radius transfer with `R = 0.01` cosine distance;
- active Gaussian KDE, using the shared calibrated bandwidth and continuous
  relative kernel weights. No neighbor rank or minimum-neighbour fallback
  defines its donor set.

The displayed edge set is not recomputed from 3D coordinates. Selection always
uses the original L2-normalized 1,280-dimensional embeddings, so the visual is
an aid to interpretation rather than a replacement for benchmark retrieval.

The corrected 2026-07-22 export contains 31,863 retained Swiss-Prot donors and
196 test proteins (32,059 jointly projected points).

## Organism colours and point source

Colour identifies the benchmark test organism: *Escherichia coli* K-12
(taxon 83333) uses blue (`#0072B2`), *Pseudomonas syringae* pv. tomato DC3000
(taxon 223283) uses vermillion (`#D55E00`), and *Synechocystis* PCC 6803
(taxon 1111708) uses green (`#009E73`). A Swiss-Prot donor associated with one
benchmark organism uses that organism's colour; a donor used by more than one
benchmark organism is violet (`#6A3D9A`). A background donor without an
association is grey.

Point styling remains the source cue: filled, white-outlined marks are
benchmark test proteins, while smaller translucent marks are Swiss-Prot
donors. The filter buttons and sidebar colour key use the same mapping, and
the hover card names the display-colour category. This lets a user distinguish
organism without confusing it with test-versus-donor status.

## Hover information and GO annotations

Hovering a point shows the UniProt accession, source (shared benchmark test set
or Swiss-Prot donor), organism/taxon when available, sequence length, and
available direct experimental GO terms. GO terms are read from the same local
`filtered_goa` snapshot used by the Clean PLM benchmark.

The footer of the explorer reports the GOA provenance supplied by the active
data index. In the current bundle this is the UniProt GAF 2.2 snapshot generated
on **2025-05-03 13:14**, linked to the GO release dated **2025-04-27**.

## Regeneration

From the S2F repository with the requested conda environment:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python \
  scripts/build_plm_neighbor_embedding_view.py --project-root . \
  --blacklist-dir /run/media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/blacklist
```

The exporter reads the existing Clean PLM benchmark query caches, selection
JSON, and blacklist-filter audit. It refuses to make a view from an unfiltered
selection file, and it does not calculate new protein embeddings or alter
prediction metrics.
It needs `umap-learn`, `numba`, and `pynndescent` in the active Python
environment, in addition to the benchmark's existing NumPy, SciPy, and
scikit-learn dependencies.

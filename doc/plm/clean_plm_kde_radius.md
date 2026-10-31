# Shared-bandwidth Gaussian KDE for Clean PLM GO transfer

## Audit of the previous implementation

The previously active method was named adaptive KDE. For each target protein
`q`, it sorted eligible Swiss-Prot donors by cosine distance and used the
nearest and rank-`k` distances to define a query-specific bandwidth:

```text
d_i = 1 - cosine(q, x_i)
h_q = scale * sqrt((d_k - d_1) / -log(epsilon))
```

The selected configuration used `k = 3`, `scale = 1`, `epsilon = 0.01`, and a
minimum of three donors. With those values, the third donor lies exactly on the
relative kernel contour:

```text
relative_weight_i = exp(-(d_i - d_1) / h_q^2)
relative_weight_3 = epsilon
```

Donors below that contour were discarded. If fewer than three survived, the
method forced the three nearest donors back into the set. The final GO score
also had a query-level nearest-distance confidence multiplier.

This approach was introduced to solve a real weakness of the earlier global
KDE: a narrow global bandwidth gave many sparse targets no contributors. A
local bandwidth plus a three-donor fallback guaranteed coverage. The mistake
was presenting that coverage-oriented rule as a method meaningfully distinct
from KNN. With rank three defining the contour and three donors enforced as a
minimum, its selected donor set was normally exactly the KNN-3 set.

The old implementation did apply continuous Gaussian weights inside that set,
so its numeric scores were not necessarily bit-for-bit equal to KNN-3 scores.
However, its topology was almost identical to KNN-3. That explains why the two
methods had the same predicted protein/GO-term support in nearly all saved
cases and very similar benchmark behavior.

## Where the kernels are centred

The target protein was not used as a kernel centre. It was, and remains, the
evaluation point in embedding space.

For a target embedding `q` and eligible Swiss-Prot donor embeddings `x_i`, KDE
evaluates kernels centred on the donor points:

```text
K_h(q - x_i)
```

The third-neighbor distance in the previous implementation did not move a
kernel centre. It selected `h_q`, and because the same rank also determined the
relative cutoff and minimum set size, it indirectly restricted the
neighbourhood to approximately three donors. That distinction is important:
the centre geometry was correct, while bandwidth and support selection were
misleadingly KNN-like.

## Corrected active method

The active method is now a shared-bandwidth Gaussian KDE:

- One bandwidth `h` is calibrated and then shared by every benchmark organism.
- The target protein is the evaluation point.
- Every eligible Swiss-Prot embedding is a Gaussian-kernel centre.
- Donors receive continuous weights from their cosine distance to the target.
- No neighbor rank defines the bandwidth or forces a fixed number of donors.
- A small relative weight floor is only a numerical approximation to the
  infinite Gaussian support.
- A high donor limit is a checked safety boundary. Calibration candidates that
  reach it are ineligible, and a final target that reaches it fails the run,
  instead of silently turning KDE into a fixed-K selector.

For unit-normalized embeddings, squared Euclidean distance equals twice cosine
distance. The implementation therefore evaluates the Gaussian as

```text
d_i = 1 - cosine(q, x_i)
w_i = exp(-d_i / h^2)
```

For numerical stability it divides all weights for a target by the nearest
donor's weight:

```text
w_i_relative = exp(-(d_i - d_min) / h^2)
```

This rescaling does not change normalized GO support because the same factor
cancels from the numerator and denominator. The closest donor has relative
weight one, but it remains a donor-centred kernel evaluated at the target.

The default floor is `epsilon = 1e-6`. A donor is materialized when

```text
w_i_relative >= epsilon
```

which corresponds to a relative cosine-distance increment

```text
R_relative = -h^2 log(epsilon)
d_i <= d_min + R_relative
```

This is not a third-neighbor radius. Depending on local donor density, one,
three, dozens, or thousands of proteins can contribute. The default checked
safety cap is 8,192 donors.

## GO-term scoring

For target `q` and GO term `t`, direct support is

```text
score(q, t) = sum_i w_i I(t is annotated on donor i) / sum_i w_i
```

The numerator is the local kernel mass from Swiss-Prot proteins annotated with
`t`; the denominator is total retained local kernel mass. Scores are continuous
in `[0, 1]`. Direct scores are then propagated through the GO graph using the
same path as KNN, so the comparison changes only neighbor selection and
weighting—not ontology handling or evaluation.

## Shared source-only bandwidth calibration

Bandwidth calibration does not use benchmark target GO labels.

1. Build one deterministic annotated Swiss-Prot pseudo-query sample for each
   benchmark organism.
2. Apply that organism's related-taxon blacklist and exclude the complete
   shared benchmark evaluation set before neighbor search.
3. Exclude calibration pseudo-queries themselves from their donor pools.
4. Derive candidate bandwidths from pooled nearest-relative distance gaps,
   without selecting any particular neighbor rank.
5. Evaluate every candidate on every source-only calibration set.
6. Use only unlabeled benchmark embedding geometry to reject candidates that
   would reach the checked donor safety boundary for a target.
7. Select one bandwidth by macro-averaged calibration performance across the
   organisms from the remaining eligible candidates.
8. Freeze that shared bandwidth for held-out benchmark evaluation.

The calibration artifact records all candidate metrics and the frozen shared
selection. Per-target diagnostics record the nearest distance, relative and
absolute contours, candidate and retained donor counts, retained kernel mass,
effective sample size, and cap status.

## KNN-3 distinction and acceptance check

Weighted KNN-3 always selects exactly the three nearest eligible donors and
normalizes their contribution. Shared KDE selects every donor above the
Gaussian numerical floor and weights it by the calibrated kernel. Its donor
count is data-dependent and is not tied to three.

Each completed run writes `kde_vs_knn_k3_comparison.csv` and
`kde_vs_knn_k3_comparison.json`. These compare propagated protein/GO pairs,
score differences, benchmark Fmax/AUPR, and the number of targets with more
than three KDE donors. The benchmark fails rather than publishing results if
KDE and weighted KNN-3 remain identical for any evaluated organism or if KDE
never retains more than three donors.

## Observed corrected run: 2026-07-22

The shared source-only optimum was `h = 0.0270212`. The unlabeled target
support check found that this would exceed the 8,192-donor boundary, so the
best fully supported candidate, `h = 0.0254097`, was frozen. Its relative
cosine-distance increment at `epsilon = 1e-6` is `0.00892002`. No final target
was truncated.

| Taxon | KNN-3 Fmax / AUPR | KDE Fmax / AUPR | KNN-3 / KDE prediction pairs | KDE targets with >3 donors | KDE median donors |
| --- | --- | --- | ---: | ---: | ---: |
| 83333 | 0.282129 / 0.178939 | 0.300794 / 0.149306 | 8,890 / 558,349 | 147/160 | 230.5 |
| 223283 | 0.857143 / 0.558201 | 0.888889 / 0.572811 | 191 / 1,790 | 2/3 | 54 |
| 1111708 | 0.550060 / 0.383797 | 0.545946 / 0.415774 | 1,486 / 70,585 | 33/33 | 23 |

The propagated prediction pair sets differ for every organism, and the
maximum per-pair score differences are `0.708556`, `0.657891`, and `0.664428`
respectively. The acceptance artifact therefore reports `status: pass` and
confirms that corrected KDE is not reproducing KNN-3 predictions. These metric
changes are benchmark observations, not evidence that one method is
universally superior.

## Reproducible run

Use the requested S2F conda interpreter. The validated 83,003-protein ESM-1b
donor cache can be reused directly. Rebuilding it is intentionally CUDA-only:

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python scripts/clean_plm_benchmark.py \
  --strategies knn kde \
  --score-modes weighted_support \
  --goa-path /run/media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_goa \
  --target-fasta /run/media/marcelo_baez/HD_Disc1/.S2F/data/UniprotKB/filtered_sprot \
  --target-cache-dir /run/media/marcelo_baez/HD_Disc1/.S2F/data/PLM/embeddings/target_57e9174250d35f0d \
  --blacklist-dir /run/media/marcelo_baez/HD_Disc1/databases/ProjectData/swiss-prot/test_set/blacklist \
  --local-files-only

/home/marcelo_baez/anaconda3/envs/S2F/bin/python scripts/build_pfp_prediction_details.py
/home/marcelo_baez/anaconda3/envs/S2F/bin/python scripts/build_knn_competitor_analysis.py
/home/marcelo_baez/anaconda3/envs/S2F/bin/python scripts/build_plm_neighbor_embedding_view.py
```

## Active and archived artifacts

The active benchmark artifacts are under
`notebooks/exports/clean_plm_benchmark/`:

- `kde_bandwidth_calibration.csv`
- `kde_calibration_proteins.csv`
- `kde_bandwidth_selection.json`
- `clean_plm_kde_neighbor_diagnostics.csv`
- `clean_plm_benchmark_metrics.csv`
- `clean_plm_predictions.tsv`
- `kde_vs_knn_k3_comparison.csv` and `.json`
- `clean_plm_benchmark_summary.json`

The PFP chart, spreadsheet, per-method result details, complete score
comparison, KNN/competitor diagnostic report, and neighbour-geometry view are
all regenerated from the same active model identifier. The previous adaptive
KDE outputs are archived with `adaptive_kde_legacy_*` names and are not treated
as active frontend results.

KDE supplies computational hypotheses for experimental prioritization. It does
not establish causal biological similarity or replace experimental validation.

# Why S2F and simple transfer methods perform well

> Derived only from the current shared, post-blacklist prediction matrices. No prediction method was rerun.

## Evaluation boundary

The analysis compares S2F, TALE, ATGO, PANDA2, Gaussian KDE, and KNN with K = 3, 5, 7, and 10. All methods are evaluated on the same propagated GO truth within each organism. Taxon `83333` has 160 proteins, 574 terms, and 1,922 positive protein–GO pairs; taxon `1111708` has 33 proteins, 93 terms, and 306 positives. The 33-protein organism is informative but statistically fragile.

Per-gene Fmax is an oracle average: each protein receives its own best threshold. AUPR evaluates the complete ranking. AUROC evaluates positive-versus-negative ordering. Smin is lower-is-better and weights remaining uncertainty and misinformation by GO-term information content. The Smin shown here is recalculated for every method from the shared prediction matrix using information content computed from the complete experimentally annotated organism before the shared evaluation-protein filter.

## Research question 1: Why does S2F perform well?

The experiment directly supports a prediction-level explanation: when S2F has higher AUPR or AUROC, its saved scores order positive protein–GO pairs ahead of negatives more successfully; when it has higher Fmax, its best operating point has a better precision–recall balance. The tables below establish those differences. They do not isolate which S2F component caused them.

### S2F versus TALE, ATGO, and PANDA2

### `83333` — per-gene

| Method | Family | Fmax | AUPR | AUROC | Calculated Smin |
|---|---|---:|---:|---:|---:|
| S2F | proposed | 0.4787 | 0.2589 | 0.8931 | 0.1311 |
| TALE | competitor | 0.4076 | 0.1717 | 0.8679 | 0.1463 |
| ATGO | competitor | 0.2474 | 0.1462 | 0.6132 | 0.1403 |
| PANDA2 | competitor | 0.3830 | 0.2052 | 0.8500 | 0.1411 |

### `1111708` — per-gene

| Method | Family | Fmax | AUPR | AUROC | Calculated Smin |
|---|---|---:|---:|---:|---:|
| S2F | proposed | 0.5936 | 0.3458 | 0.8991 | 0.3244 |
| TALE | competitor | 0.4627 | 0.2284 | 0.8091 | 0.4527 |
| ATGO | competitor | 0.2760 | 0.2034 | 0.5075 | 0.4263 |
| PANDA2 | competitor | 0.4379 | 0.2646 | 0.7912 | 0.4740 |

### Observed prediction-score properties

These values use the same score rounding as the benchmark metrics. Missing outputs are score zero. Because method scores are not guaranteed to be calibrated probabilities, absolute score magnitudes should not be compared as confidence across methods.

| Organism | Method | Output coverage | True-pair nonzero | Negative-pair nonzero | Positive median | Negative p90 |
|---|---|---:|---:|---:|---:|---:|
| 83333 | S2F | 0.984 | 0.822 | 0.289 | 0.0999 | 0.0926 |
| 83333 | TALE | 0.215 | 0.715 | 0.204 | 0.1720 | 0.0900 |
| 83333 | ATGO | 0.135 | 0.531 | 0.127 | 0.0179 | 0.0183 |
| 83333 | PANDA2 | 0.130 | 0.660 | 0.119 | 0.2300 | 0.1100 |
| 1111708 | S2F | 1.000 | 1.000 | 1.000 | 0.0985 | 0.1004 |
| 1111708 | TALE | 0.434 | 0.765 | 0.397 | 0.3000 | 0.4008 |
| 1111708 | ATGO | 0.335 | 0.516 | 0.315 | 0.0260 | 0.1842 |
| 1111708 | PANDA2 | 0.324 | 0.660 | 0.287 | 0.2350 | 0.4600 |

These score properties can explain metric differences only at the output level. For example, greater recovery of true pairs can support recall, while high scores on negatives can reduce precision and AUPR. The current experiment does not contain an S2F component ablation, so it cannot establish why the pipeline produced those score patterns.

## Research question 2: Why do simpler algorithms sometimes perform better?

The saved results show whether a simple method's advantage is confined to Fmax or extends to ranking metrics. A higher Fmax demonstrates a better best threshold; it does not by itself demonstrate a better full ranking. AUPR and AUROC must be read separately.

### All methods

### `83333` — per-gene

| Method | Family | Fmax | AUPR | AUROC | Calculated Smin |
|---|---|---:|---:|---:|---:|
| S2F | proposed | 0.4787 | 0.2589 | 0.8931 | 0.1311 |
| TALE | competitor | 0.4076 | 0.1717 | 0.8679 | 0.1463 |
| ATGO | competitor | 0.2474 | 0.1462 | 0.6132 | 0.1403 |
| PANDA2 | competitor | 0.3830 | 0.2052 | 0.8500 | 0.1411 |
| Gaussian KDE | simple_embedding_transfer | 0.4309 | 0.1182 | 0.8464 | 0.1824 |
| KNN K=10 | simple_embedding_transfer | 0.4934 | 0.1195 | 0.8452 | 0.1682 |
| KNN K=7 | simple_embedding_transfer | 0.4435 | 0.1201 | 0.8251 | 0.1783 |
| KNN K=5 | simple_embedding_transfer | 0.4090 | 0.1105 | 0.8127 | 0.1953 |
| KNN K=3 | simple_embedding_transfer | 0.3517 | 0.1013 | 0.7692 | 0.2188 |

### `1111708` — per-gene

| Method | Family | Fmax | AUPR | AUROC | Calculated Smin |
|---|---|---:|---:|---:|---:|
| S2F | proposed | 0.5936 | 0.3458 | 0.8991 | 0.3244 |
| TALE | competitor | 0.4627 | 0.2284 | 0.8091 | 0.4527 |
| ATGO | competitor | 0.2760 | 0.2034 | 0.5075 | 0.4263 |
| PANDA2 | competitor | 0.4379 | 0.2646 | 0.7912 | 0.4740 |
| Gaussian KDE | simple_embedding_transfer | 0.7100 | 0.1962 | 0.9223 | 0.3042 |
| KNN K=10 | simple_embedding_transfer | 0.7262 | 0.1936 | 0.9157 | 0.3121 |
| KNN K=7 | simple_embedding_transfer | 0.6665 | 0.1947 | 0.8992 | 0.3701 |
| KNN K=5 | simple_embedding_transfer | 0.6744 | 0.2160 | 0.8979 | 0.3616 |
| KNN K=3 | simple_embedding_transfer | 0.6647 | 0.1817 | 0.8776 | 0.3713 |

## Evidence limits

The transfer audit reports high cosine similarity to selected donors and zero violations of the configured taxon blacklist and shared-benchmark accession exclusion. It does not measure whether a selected donor carries the correct GO annotation for each query, prove phylogenetic independence, or rule out PLM pretraining exposure. Therefore the available data can identify ranking, coverage, precision, recall, and threshold behavior, but cannot prove an architectural or biological cause for a method's advantage.

## Conclusions

1. S2F's advantages over TALE, ATGO, and PANDA2 are supported where its Fmax, AUPR, AUROC, or calculated Smin is better. Higher AUPR/AUROC is evidence of better score ordering; threshold precision and recall show the operating-point behavior. The experiment does not identify which S2F component causes those outputs.
2. A simple method's higher Fmax is evidence of a stronger best operating point, not automatically a stronger ranking. When its AUPR or AUROC is lower, the result is metric-specific. The donor audit is insufficient to prove that neighborhood quality caused the advantage.
3. Method order depends on organism, evaluation scope, ontology, and metric. Results for `1111708` require particular caution because the shared set contains only 33 proteins.

## Reproduction

```bash
/home/marcelo_baez/anaconda3/envs/S2F/bin/python scripts/build_method_comparison_analysis.py
```

The manifest in `notebooks/exports/method_comparison_analysis/analysis_manifest.json` records the interpreter and SHA-256 hashes of all consumed artifacts.

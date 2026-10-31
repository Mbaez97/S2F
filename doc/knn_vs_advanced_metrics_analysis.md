# Why can KNN obtain a strong Fmax?

> This report uses only the active post-blacklist benchmark artifacts. Archived pre-blacklist results are intentionally excluded.

## Evaluation boundary

All methods use the same propagated GO ground truth and shared evaluation proteins. The reconstructed matrices contain 160 proteins, 574 GO terms, and 1,922 positive pairs for `83333`; and 33 proteins, 93 GO terms, and 306 positive pairs for `1111708`. Smin is lower-is-better; Fmax, AUPR, and AUROC are higher-is-better.

The current code defines per-gene Fmax as the mean of a separately maximized F1 score for each protein. Therefore every protein may use a different oracle threshold. This is more permissive than selecting one threshold and deploying it for every protein.

**Smin comparability warning.** The saved S2F, TALE, ATGO, and PANDA2 rows calculate GO information content on the full organism before applying the shared evaluation-protein filter. The Clean PLM rows calculate it after that filter. Their published Smin values are therefore not strictly comparable. This report preserves them in the main reproduction tables and supplies a harmonized evaluation-set-IC Smin in the generated diagnostics.

## Metric comparison

### 83333 — overall

| Method | Fmax | AUPR | AUROC | Smin |
|---|---:|---:|---:|---:|
| S2F | 0.3158 | 0.2012 | 0.8342 | 92.6206 |
| TALE | 0.2968 | 0.1438 | 0.7963 | 95.4713 |
| ATGO | 0.2513 | 0.1432 | 0.7194 | 94.8421 |
| PANDA2 | 0.2907 | 0.1911 | 0.7887 | 94.4694 |
| KNN k=10 (weighted_support) | 0.2855 | 0.2138 | 0.7714 | 83.4648 |

### 83333 — per-gene

| Method | Fmax | AUPR | AUROC | Smin |
|---|---:|---:|---:|---:|
| S2F | 0.4787 | 0.2589 | 0.8931 | 0.1311 |
| TALE | 0.4076 | 0.1717 | 0.8679 | 0.1463 |
| ATGO | 0.2474 | 0.1462 | 0.6132 | 0.1403 |
| PANDA2 | 0.3830 | 0.2052 | 0.8500 | 0.1411 |
| KNN k=10 (weighted_support) | 0.4934 | 0.1195 | 0.8452 | 0.1573 |

### 83333 — per-term

| Method | Fmax | AUPR | AUROC | Smin |
|---|---:|---:|---:|---:|
| S2F | 0.3464 | 0.0835 | 0.7073 | 1.0823 |
| TALE | 0.2136 | 0.0587 | 0.6316 | 3.7192 |
| ATGO | 0.3297 | 0.0961 | 0.7197 | 2.3785 |
| PANDA2 | 0.2202 | 0.0660 | 0.6384 | 5.1649 |
| KNN k=10 (weighted_support) | 0.2324 | 0.0578 | 0.6263 | 1.7865 |

### 1111708 — overall

| Method | Fmax | AUPR | AUROC | Smin |
|---|---:|---:|---:|---:|
| S2F | 0.4125 | 0.3017 | 0.8318 | 60.9306 |
| TALE | 0.3759 | 0.2198 | 0.7414 | 57.6985 |
| ATGO | 0.2805 | 0.1747 | 0.6269 | 61.3064 |
| PANDA2 | 0.3452 | 0.2337 | 0.7123 | 61.0643 |
| KNN k=10 (weighted_support) | 0.5304 | 0.3669 | 0.8545 | 43.1995 |

### 1111708 — per-gene

| Method | Fmax | AUPR | AUROC | Smin |
|---|---:|---:|---:|---:|
| S2F | 0.5936 | 0.3458 | 0.8991 | 0.3244 |
| TALE | 0.4627 | 0.2284 | 0.8091 | 0.4527 |
| ATGO | 0.2760 | 0.2034 | 0.5075 | 0.4263 |
| PANDA2 | 0.4379 | 0.2646 | 0.7912 | 0.4740 |
| KNN k=10 (weighted_support) | 0.7262 | 0.1936 | 0.9157 | 0.2665 |

### 1111708 — per-term

| Method | Fmax | AUPR | AUROC | Smin |
|---|---:|---:|---:|---:|
| S2F | 0.6077 | 0.2397 | 0.8127 | 0.4767 |
| TALE | 0.4008 | 0.1609 | 0.6967 | 2.4478 |
| ATGO | 0.4501 | 0.2050 | 0.7366 | 1.6400 |
| PANDA2 | 0.3243 | 0.1655 | 0.6540 | 2.9371 |
| KNN k=10 (weighted_support) | 0.5998 | 0.2801 | 0.8300 | 0.7664 |

## Why the rankings differ

- **Fmax** keeps only the best precision/recall operating point. A method can score well if one threshold isolates a useful subset, even if scores below and above that point are poorly ordered.
- **AUPR** evaluates the full precision–recall ranking. It is sensitive to false positives appearing ahead of true annotations and is informative for the highly imbalanced protein–GO matrix. Davis and Goadrich give the classic explanation of why PR views are useful for skewed data ([ICML 2006](https://doi.org/10.1145/1143844.1143874)).
- **AUROC** measures positive-versus-negative ranking over all pairs. Because negatives vastly outnumber positives, it can remain high while precision is modest.
- **Smin** weights remaining uncertainty and misinformation by GO-term information content. Incorrect or missed specific terms can hurt Smin more than errors on common terms.

KNN-10 transfers a sparse, quantized set of scores based on how many of ten almost equally weighted donors support a term. That score geometry can create a good Fmax cutoff while producing ties and false-positive blocks that reduce AUPR. S2F is much denser, so its score ordering can support a stronger full PR curve even when its best single F1 point is lower. CAFA likewise notes that metric choice can affect method rankings and separates protein- from term-centric questions ([Radivojac et al., 2013](https://www.nature.com/articles/nmeth.2340)).

## Harmonized Smin sensitivity

The values below recompute Smin for every method using information content from the same shared evaluation set. They should be used for the fair Smin sensitivity analysis; they do not overwrite the historical benchmark table.

| Organism | Scope | Method | Saved Smin | Harmonized Smin |
|---|---|---|---:|---:|
| 83333 | overall | S2F | 92.6206 | 83.4654 |
| 83333 | overall | TALE | 95.4713 | 86.9193 |
| 83333 | overall | ATGO | 94.8421 | 85.5703 |
| 83333 | overall | PANDA2 | 94.4694 | 85.4708 |
| 83333 | overall | KNN k=10 (weighted_support) | 83.4648 | 83.4648 |
| 83333 | per-gene | S2F | 0.1311 | 0.1179 |
| 83333 | per-gene | TALE | 0.1463 | 0.1329 |
| 83333 | per-gene | ATGO | 0.1403 | 0.1265 |
| 83333 | per-gene | PANDA2 | 0.1411 | 0.1265 |
| 83333 | per-gene | KNN k=10 (weighted_support) | 0.1573 | 0.1573 |
| 83333 | per-term | S2F | 1.0823 | 0.8535 |
| 83333 | per-term | TALE | 3.7192 | 2.9405 |
| 83333 | per-term | ATGO | 2.3785 | 1.8605 |
| 83333 | per-term | PANDA2 | 5.1649 | 4.1288 |
| 83333 | per-term | KNN k=10 (weighted_support) | 1.7865 | 1.7865 |
| 1111708 | overall | S2F | 60.9306 | 50.0884 |
| 1111708 | overall | TALE | 57.6985 | 48.4077 |
| 1111708 | overall | ATGO | 61.3064 | 50.2296 |
| 1111708 | overall | PANDA2 | 61.0643 | 50.9482 |
| 1111708 | overall | KNN k=10 (weighted_support) | 43.1995 | 43.1995 |
| 1111708 | per-gene | S2F | 0.3244 | 0.2629 |
| 1111708 | per-gene | TALE | 0.4527 | 0.3774 |
| 1111708 | per-gene | ATGO | 0.4263 | 0.3488 |
| 1111708 | per-gene | PANDA2 | 0.4740 | 0.3881 |
| 1111708 | per-gene | KNN k=10 (weighted_support) | 0.2665 | 0.2665 |
| 1111708 | per-term | S2F | 0.4767 | 0.4053 |
| 1111708 | per-term | TALE | 2.4478 | 1.9480 |
| 1111708 | per-term | ATGO | 1.6400 | 1.3273 |
| 1111708 | per-term | PANDA2 | 2.9371 | 2.3413 |
| 1111708 | per-term | KNN k=10 (weighted_support) | 0.7664 | 0.7664 |

## Threshold diagnostics

The tables below contrast the oracle per-gene result with a shared threshold and with thresholds learned on training folds. The cross-validation values are diagnostics, not replacement benchmark metrics.

| Organism | Method | Per-gene oracle Fmax | Median local threshold | Best shared-threshold F1 | Held-out threshold F1 |
|---|---|---:|---:|---:|---:|
| 83333 | S2F | 0.4787 | 0.1001 | 0.3226 | 0.3226 |
| 83333 | TALE | 0.4076 | 0.6560 | 0.2838 | 0.2769 |
| 83333 | ATGO | 0.2474 | 0.0407 | 0.1799 | 0.1752 |
| 83333 | PANDA2 | 0.3830 | 0.5600 | 0.2622 | 0.2569 |
| 83333 | KNN k=10 (weighted_support) | 0.4934 | 0.2000 | 0.2921 | 0.2647 |
| 1111708 | S2F | 0.5936 | 0.1606 | 0.3416 | 0.3048 |
| 1111708 | TALE | 0.4627 | 0.6880 | 0.3324 | 0.3048 |
| 1111708 | ATGO | 0.2760 | 0.0000 | 0.2074 | 0.1996 |
| 1111708 | PANDA2 | 0.4379 | 0.4700 | 0.3064 | 0.2828 |
| 1111708 | KNN k=10 (weighted_support) | 0.7262 | 0.3002 | 0.5213 | 0.5193 |

## Paired-bootstrap stability of KNN versus S2F

The intervals resample the same proteins for both methods. Positive deltas favor KNN for Fmax, AUPR, and AUROC; positive Smin deltas favor S2F because lower is better. Smin uses harmonized shared-set information content.

| Organism | Metric | KNN − S2F | 95% paired interval | P(delta > 0) |
|---|---|---:|---:|---:|
| 83333 | AUC | -0.0479 | [-0.0638, -0.0320] | 0.000 |
| 83333 | AUPR | -0.1395 | [-0.1666, -0.1135] | 0.000 |
| 83333 | F_max | +0.0147 | [-0.0307, +0.0636] | 0.721 |
| 83333 | smin | +0.0394 | [+0.0258, +0.0534] | 1.000 |
| 1111708 | AUC | +0.0166 | [-0.0124, +0.0454] | 0.870 |
| 1111708 | AUPR | -0.1521 | [-0.2290, -0.0877] | 0.000 |
| 1111708 | F_max | +0.1327 | [+0.0366, +0.2295] | 0.995 |
| 1111708 | smin | +0.0036 | [-0.0669, +0.0756] | 0.545 |

## Score-distribution evidence

| Organism | Method | Nonzero pairs | Zero fraction | Unique nonzero scores | Median nonzero score |
|---|---|---:|---:|---:|---:|
| 83333 | S2F | 27,526 | 0.700 | 811 | 0.0005 |
| 83333 | TALE | 19,714 | 0.785 | 975 | 0.0930 |
| 83333 | ATGO | 12,426 | 0.865 | 8,793 | 0.0521 |
| 83333 | PANDA2 | 11,953 | 0.870 | 88 | 0.2200 |
| 83333 | KNN k=10 (weighted_support) | 10,660 | 0.884 | 1,386 | 0.1000 |
| 1111708 | S2F | 3,069 | 0.000 | 2,967 | 0.0022 |
| 1111708 | TALE | 1,331 | 0.566 | 567 | 0.1980 |
| 1111708 | ATGO | 1,029 | 0.665 | 836 | 0.0881 |
| 1111708 | PANDA2 | 994 | 0.676 | 88 | 0.3200 |
| 1111708 | KNN k=10 (weighted_support) | 923 | 0.699 | 209 | 0.1996 |

## Ontology and term-frequency diagnostics

The ontology table removes GO roots. The frequency table is term-centric and bins terms by positive-protein count.

| Organism | Ontology | Method | Per-gene Fmax | Per-gene AUPR |
|---|---|---|---:|---:|
| 83333 | Biological Process | S2F | 0.1994 | 0.1348 |
| 83333 | Biological Process | KNN k=10 (weighted_support) | 0.1568 | 0.0415 |
| 83333 | Molecular Function | S2F | 0.5157 | 0.1843 |
| 83333 | Molecular Function | KNN k=10 (weighted_support) | 0.4848 | 0.0826 |
| 83333 | Cellular Component | S2F | 0.1678 | 0.1161 |
| 83333 | Cellular Component | KNN k=10 (weighted_support) | 0.1193 | 0.0385 |
| 1111708 | Biological Process | S2F | 0.2309 | 0.1886 |
| 1111708 | Biological Process | KNN k=10 (weighted_support) | 0.2200 | 0.1290 |
| 1111708 | Molecular Function | S2F | 0.7009 | 0.3034 |
| 1111708 | Molecular Function | KNN k=10 (weighted_support) | 0.7681 | 0.0879 |
| 1111708 | Cellular Component | S2F | 0.1818 | 0.1614 |
| 1111708 | Cellular Component | KNN k=10 (weighted_support) | 0.1818 | 0.0152 |

| Organism | Term-frequency bin | Method | Per-term Fmax | Per-term AUPR |
|---|---|---|---:|---:|
| 83333 | singleton | S2F | 0.3309 | 0.0271 |
| 83333 | singleton | KNN k=10 (weighted_support) | 0.1820 | 0.0200 |
| 83333 | rare_2_4 | S2F | 0.3505 | 0.1200 |
| 83333 | rare_2_4 | KNN k=10 (weighted_support) | 0.2683 | 0.0672 |
| 83333 | medium_5_9 | S2F | 0.3565 | 0.1698 |
| 83333 | medium_5_9 | KNN k=10 (weighted_support) | 0.3236 | 0.1385 |
| 83333 | common_10_plus | S2F | 0.4832 | 0.3410 |
| 83333 | common_10_plus | KNN k=10 (weighted_support) | 0.4129 | 0.2783 |
| 1111708 | singleton | S2F | 0.6223 | 0.1115 |
| 1111708 | singleton | KNN k=10 (weighted_support) | 0.4844 | 0.1221 |
| 1111708 | rare_2_4 | S2F | 0.6242 | 0.3077 |
| 1111708 | rare_2_4 | KNN k=10 (weighted_support) | 0.6885 | 0.3551 |
| 1111708 | medium_5_9 | S2F | 0.4640 | 0.2696 |
| 1111708 | medium_5_9 | KNN k=10 (weighted_support) | 0.5798 | 0.3437 |
| 1111708 | common_10_plus | S2F | 0.7390 | 0.6056 |
| 1111708 | common_10_plus | KNN k=10 (weighted_support) | 0.7892 | 0.7416 |

For `83333`, S2F is stronger in every root-excluded ontology and every term-frequency bin. For `1111708`, KNN's advantage is concentrated in Molecular Function and terms observed on at least two proteins; Cellular Component remains weak and the organism has only 33 evaluated proteins.

## K and Gaussian KDE sensitivity

| Organism | Method | Per-gene Fmax | Per-gene AUPR | Per-gene AUROC |
|---|---|---:|---:|---:|
| 83333 | KNN k=10 (weighted_support) | 0.4934 | 0.1195 | 0.8452 |
| 83333 | KNN k=3 (weighted_support) | 0.3517 | 0.1013 | 0.7692 |
| 83333 | KNN k=5 (weighted_support) | 0.4090 | 0.1105 | 0.8127 |
| 83333 | KNN k=7 (weighted_support) | 0.4435 | 0.1201 | 0.8251 |
| 83333 | KDE (gaussian_kernel_support) | 0.4309 | 0.1182 | 0.8464 |
| 1111708 | KNN k=10 (weighted_support) | 0.7262 | 0.1936 | 0.9157 |
| 1111708 | KNN k=3 (weighted_support) | 0.6647 | 0.1817 | 0.8776 |
| 1111708 | KNN k=5 (weighted_support) | 0.6744 | 0.2160 | 0.8979 |
| 1111708 | KNN k=7 (weighted_support) | 0.6665 | 0.1947 | 0.8992 |
| 1111708 | KDE (gaussian_kernel_support) | 0.7100 | 0.1962 | 0.9223 |

## Post-blacklist neighbor audit

| Organism | Queries | Unique KNN-10 donors | Donor organisms | Blacklist violations | Benchmark-accession violations | Median top-1 similarity |
|---|---:|---:|---:|---:|---:|---:|
| 83333 | 160 | 1209 | 304 | 0 | 0 | 0.9895 |
| 1111708 | 33 | 268 | 87 | 0 | 0 | 0.9967 |

| Organism | Neighbor feature | Outcome | Spearman rho |
|---|---|---|---:|
| 83333 | nearest similarity | KNN minus best advanced fmax | +0.058 |
| 83333 | nearest similarity | KNN minus best advanced aupr | +0.008 |
| 1111708 | nearest similarity | KNN minus best advanced fmax | -0.130 |
| 1111708 | nearest similarity | KNN minus best advanced aupr | -0.172 |

The audit can establish that the configured blacklist and shared benchmark accessions were removed before transfer. It cannot establish that all remaining donors are phylogenetically distant or that PLM pretraining was free of exposure. High cosine similarity outside the configured blacklist is a hypothesis for follow-up, not proof of leakage.

## Organism-specific interpretation

### `83333`

KNN-10 narrowly exceeds S2F in per-gene Fmax, but S2F has much higher per-gene AUPR and AUROC and lower per-gene Smin. This is the clearest case of a threshold-specific KNN advantage: its best local cutoff is effective, while its full within-protein ranking is weaker. Overall and per-term results must be read separately because they pool a different set of decisions.

### `1111708`

KNN-10 has a large per-gene Fmax advantage and strong per-gene AUROC, but its per-gene AUPR remains below S2F. The historical saved Smin favors KNN, but the harmonized same-IC Smin slightly favors S2F and the paired interval crosses zero. The organism has only 33 proteins, so paired-bootstrap intervals and protein examples are required before treating the ranking as stable.

## Example proteins

The explorer includes five cases in each direction for each organism. The compact table below lists the two largest Fmax differences in each direction; use the explorer buttons to open exact donor neighborhoods and protein–GO outcomes.

| Organism | Direction | Protein | Best advanced method | KNN Fmax | Advanced Fmax | Delta |
|---|---|---|---|---:|---:|---:|
| 83333 | knn_better | P19317 | TALE | 1.0000 | 0.3333 | +0.6667 |
| 83333 | knn_better | P06974 | TALE | 1.0000 | 0.3333 | +0.6667 |
| 83333 | advanced_better | P37766 | S2F | 0.2264 | 0.9091 | -0.6827 |
| 83333 | advanced_better | P68646 | S2F | 0.1333 | 0.8000 | -0.6667 |
| 1111708 | knn_better | Q55169 | TALE | 1.0000 | 0.3333 | +0.6667 |
| 1111708 | knn_better | Q55544 | TALE | 1.0000 | 0.3333 | +0.6667 |
| 1111708 | advanced_better | P72854 | S2F | 0.2500 | 0.8000 | -0.5500 |
| 1111708 | advanced_better | Q05972 | TALE | 0.3158 | 0.5714 | -0.2556 |

## Additional validation experiments

1. Select one global threshold on a disjoint validation organism or historical time split and apply it unchanged to the benchmark.
2. Repeat KNN after removing donors above sequence-identity and taxonomic-clade cutoffs, without using benchmark GO labels to choose those cutoffs.
3. Remove ontology roots and high-frequency ancestors to measure how much Fmax comes from generic transferred terms.
4. Compare cosine-weighted KNN with equal voting and calibrated probabilities to separate neighborhood quality from score calibration.
5. Use bootstrap intervals and repeated organism-level splits rather than interpreting the 33-protein result as a precise population estimate.

## Reproduction

```bash
python scripts/build_knn_competitor_analysis.py --project-root .
```

The exporter records SHA-256 hashes for every prediction-detail and neighbor input in `notebooks/exports/knn_competitor_analysis/analysis_manifest.json`.

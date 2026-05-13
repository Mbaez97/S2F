# 1111708 Seed Weight Grid

These configs scan different seed-weight mixtures for the blacklist-aware
1111708 experiment.

Weight meaning:

- `alpha`: InterPro seed weight
- `beta`: HMMER seed weight
- `gamma`: Foldseek seed weight
- `delta`: PLM seed residual, computed as `1 - alpha - beta - gamma`

Each config reuses the blacklist-aware Foldseek and PLM assignment files from
`1111708_with_blacklist`, and reuses the blacklist-aware graph collection plus
the existing homology graph. `combined_graph = compute` is intentional: S2F
learns graph-combination coefficients from the weighted seed target, so each
weight mixture should get its own combined graph.

| Config | alpha | beta | gamma | delta | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `1111708_with_blacklist.conf` | 0.60 | 0.10 | 0.20 | 0.10 | Current baseline |
| `1111708_w_equal.conf` | 0.25 | 0.25 | 0.25 | 0.25 | Equal four-seed mixture |
| `1111708_w_interpro_heavy.conf` | 0.70 | 0.10 | 0.10 | 0.10 | InterPro-heavy |
| `1111708_w_hmmer_heavy.conf` | 0.20 | 0.50 | 0.20 | 0.10 | HMMER-heavy |
| `1111708_w_foldseek_heavy.conf` | 0.20 | 0.10 | 0.60 | 0.10 | Foldseek-heavy |
| `1111708_w_plm_heavy.conf` | 0.20 | 0.10 | 0.10 | 0.60 | PLM-heavy |
| `1111708_w_no_interpro.conf` | 0.00 | 0.33 | 0.33 | 0.34 | Leave out InterPro |
| `1111708_w_no_hmmer.conf` | 0.33 | 0.00 | 0.33 | 0.34 | Leave out HMMER |
| `1111708_w_no_foldseek.conf` | 0.33 | 0.33 | 0.00 | 0.34 | Leave out Foldseek |
| `1111708_w_no_plm.conf` | 0.34 | 0.33 | 0.33 | 0.00 | Leave out PLM |
| `1111708_w_foldseek_plm.conf` | 0.00 | 0.00 | 0.50 | 0.50 | Foldseek plus PLM only |
| `1111708_w_interpro_hmmer.conf` | 0.90 | 0.10 | 0.00 | 0.00 | Original-style InterPro plus HMMER |
| `1111708_w_interpro_foldseek.conf` | 0.50 | 0.00 | 0.50 | 0.00 | InterPro plus Foldseek only |
| `1111708_w_interpro_plm.conf` | 0.50 | 0.00 | 0.00 | 0.50 | InterPro plus PLM only |
| `1111708_w_hmmer_foldseek.conf` | 0.00 | 0.50 | 0.50 | 0.00 | HMMER plus Foldseek only |
| `1111708_w_hmmer_plm.conf` | 0.00 | 0.50 | 0.00 | 0.50 | HMMER plus PLM only |
| `1111708_w_interpro_only.conf` | 1.00 | 0.00 | 0.00 | 0.00 | InterPro-only baseline |
| `1111708_w_hmmer_only.conf` | 0.00 | 1.00 | 0.00 | 0.00 | HMMER-only baseline |
| `1111708_w_foldseek_only.conf` | 0.00 | 0.00 | 1.00 | 0.00 | Foldseek-only baseline |
| `1111708_w_plm_only.conf` | 0.00 | 0.00 | 0.00 | 1.00 | PLM-only baseline |

Run one config with:

```bash
python S2F.py predict --run-config conf/weight_grid/1111708_w_current.conf
```

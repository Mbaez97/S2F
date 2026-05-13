# 83333 Seed Weight Grid

These configs scan the same seed-weight mixtures used for the 1111708 and 223283 grids,
but for organism 83333.

Weight meaning:

- `alpha`: InterPro seed weight
- `beta`: HMMER seed weight
- `gamma`: Foldseek seed weight
- `delta`: PLM seed residual, computed as `1 - alpha - beta - gamma`

Use `83333.conf` as the current baseline. The grid configs assume that this
baseline has already produced the 83333 graph collection, homology graph,
InterPro seed, HMMER seed, Foldseek assignments, and PLM assignments. Each grid
config reuses those baseline artifacts and keeps `combined_graph = compute` so
S2F learns graph-combination coefficients from each weighted seed target.

| Config | alpha | beta | gamma | delta | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `83333.conf` | 0.60 | 0.10 | 0.20 | 0.10 | Current baseline |
| `83333_w_equal.conf` | 0.25 | 0.25 | 0.25 | 0.25 | Equal four-seed mixture |
| `83333_w_interpro_heavy.conf` | 0.70 | 0.10 | 0.10 | 0.10 | InterPro-heavy |
| `83333_w_hmmer_heavy.conf` | 0.20 | 0.50 | 0.20 | 0.10 | HMMER-heavy |
| `83333_w_foldseek_heavy.conf` | 0.20 | 0.10 | 0.60 | 0.10 | Foldseek-heavy |
| `83333_w_plm_heavy.conf` | 0.20 | 0.10 | 0.10 | 0.60 | PLM-heavy |
| `83333_w_no_interpro.conf` | 0.00 | 0.33 | 0.33 | 0.34 | Leave out InterPro |
| `83333_w_no_hmmer.conf` | 0.33 | 0.00 | 0.33 | 0.34 | Leave out HMMER |
| `83333_w_no_foldseek.conf` | 0.33 | 0.33 | 0.00 | 0.34 | Leave out Foldseek |
| `83333_w_no_plm.conf` | 0.34 | 0.33 | 0.33 | 0.00 | Leave out PLM |
| `83333_w_foldseek_plm.conf` | 0.00 | 0.00 | 0.50 | 0.50 | Foldseek plus PLM only |
| `83333_w_interpro_hmmer.conf` | 0.90 | 0.10 | 0.00 | 0.00 | Original-style InterPro plus HMMER |
| `83333_w_interpro_foldseek.conf` | 0.50 | 0.00 | 0.50 | 0.00 | InterPro plus Foldseek only |
| `83333_w_interpro_plm.conf` | 0.50 | 0.00 | 0.00 | 0.50 | InterPro plus PLM only |
| `83333_w_hmmer_foldseek.conf` | 0.00 | 0.50 | 0.50 | 0.00 | HMMER plus Foldseek only |
| `83333_w_hmmer_plm.conf` | 0.00 | 0.50 | 0.00 | 0.50 | HMMER plus PLM only |
| `83333_w_interpro_only.conf` | 1.00 | 0.00 | 0.00 | 0.00 | InterPro-only baseline |
| `83333_w_hmmer_only.conf` | 0.00 | 1.00 | 0.00 | 0.00 | HMMER-only baseline |
| `83333_w_foldseek_only.conf` | 0.00 | 0.00 | 1.00 | 0.00 | Foldseek-only baseline |
| `83333_w_plm_only.conf` | 0.00 | 0.00 | 0.00 | 1.00 | PLM-only baseline |

Run the baseline first with:

```bash
python S2F.py predict --run-config conf/weight_grid/83333/83333.conf
```

Run one grid config with:

```bash
python S2F.py predict --run-config conf/weight_grid/83333/83333_w_equal.conf
```

# S2F Change Documentation

This folder contains documentation for the Foldseek and PLM seed integrations
and the related pipeline changes introduced for testing.

## Documents

- [Foldseek GO Transfer Overview](./foldseek_go_transfer.md)
  - what changed in `foldseek.py`
  - how GO terms are transferred from SwissProt Foldseek hits
  - output files and scoring modes
  - confirms that InterPro2GO and `topk` are no longer used

- [Predict Pipeline Changes](./predict_three_seed_pipeline.md)
  - how `predict` now handles InterPro, HMMER, Foldseek, and PLM together
  - how the seed weights are applied
  - how Foldseek affects graph combination and final prediction
  - how `foldseek_output = compute` calls the root-level `foldseek.py`

- [Config And CLI Changes](./config_and_cli_changes.md)
  - new `predict` arguments
  - new run-config fields under `[seeds]` and `[foldseek]`
  - role of `s2f.conf` and `filtered_goa`
  - expected defaults and validation behavior
  - `max_seqs`, GPU, CUDA device, and padded DB notes

- [Verification And Testing](./verification_and_testing.md)
  - smoke checks already run
  - manual end-to-end checks still recommended
  - example testing workflow
  - ProstT5/GPU manual check command

- [1111708 Foldseek Full Experiment](./1111708_foldseek_full_experiment.md)
  - full three-seed experiment config
  - exact `S2F.py predict` command
  - recovery config using a precomputed raw Foldseek TSV
  - expected Foldseek artifacts

- [PLM Seed Instructions](./plm_seed_INSTRUCTIONS.md)
  - standalone `plm.py` usage
  - ESM1b dependency and cache behavior
  - KNN transfer and residual seed weighting
  - full four-seed experiment config

## Related Script Documentation

The script-specific usage guide is:

- [Foldseek Script Instructions](./foldseek_INSTRUCTIONS.md)
- [PLM Seed Instructions](./plm_seed_INSTRUCTIONS.md)

# Data manifest and availability

This GitHub repository includes only `dataset_manifest.csv` and small smoke-test examples.

The full processed expression matrices are not committed to GitHub because the complete data directory is approximately 1.8 GB and several individual CSV files exceed GitHub's normal 100 MB file limit.

Raw source data are public GEO datasets. The accession identifiers, cohort roles, sample counts, platform information, and expected filenames are listed in `dataset_manifest.csv`.

Expected local layout for full reproduction:

```text
data/main_benchmark/
data/external_validation/
data/external_validation_v2/
data/canonical/
```

Use `examples/example_cumida_small.csv` only for smoke testing the benchmark runner.

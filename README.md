# clinifs-benchmark

Benchmark source code accompanying the clinifs paper:

*Multi-dimensional benchmarking of feature selection methods for clinical small-panel cancer gene-expression classification*.

## What is included

This repository is intentionally lightweight. It contains:

- benchmark source code in `src/`
- command-line runners `run_*.py`
- locked Python dependencies in `requirements.txt` and `requirements.lock`
- dataset manifest in `data/dataset_manifest.csv`
- a tiny smoke-test dataset in `examples/example_cumida_small.csv`
- figure-generation and manuscript assets in the companion submission package, referenced from the paper

## What is not included

Large processed expression matrices and full intermediate checkpoint outputs are not stored in GitHub because they exceed normal repository size limits. The raw source data are public GEO datasets listed in `data/dataset_manifest.csv`; large processed matrices can be reconstructed from the public accessions using the documented preprocessing and benchmark scripts.

The deployed Streamlit platform provides interactive access to the main summary results and lightweight online feature-selection runs:

https://clinifs-platform.streamlit.app/

The installable clinifs package is released at:

https://github.com/xwdshiwo/clinifs/releases/tag/v0.1.0

Zenodo archive:

https://doi.org/10.5281/zenodo.19914970

## Installation

```bash
conda create -n fs_bench python=3.12 -y
conda activate fs_bench
pip install -r requirements.txt
pip install "git+https://github.com/xwdshiwo/clinifs.git@v0.1.0"
```

## Smoke test

The included example dataset is only for verifying the code path:

```bash
mkdir -p data/main_benchmark
cp examples/example_cumida_small.csv data/main_benchmark/example_cumida_small.csv
python run_all.py --methods anova --datasets example_cumida_small
```

## Full benchmark

To reproduce the full benchmark, prepare the processed datasets according to `data/dataset_manifest.csv` under:

```text
data/main_benchmark/
data/external_validation/
data/external_validation_v2/
```

Then run:

```bash
python run_all.py --methods all
python run_external_validation.py
python run_negative_controls.py
```

The full runs are computationally expensive. The online platform and manuscript figures report the summary-level results used in the paper.

## Repository roles

- `clinifs`: installable Python toolkit
- `clinifs-platform`: Streamlit demonstration platform
- `clinifs-benchmark`: lightweight benchmark source code and reproduction entry points

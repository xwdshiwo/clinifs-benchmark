"""
Main benchmark runner: 11 datasets × 13+2 methods × 5-fold × 5-repeat nested CV.
Usage:
  python run_all.py                    # run all main methods
  python run_all.py --methods anova l1_logistic boruta
  python run_all.py --supp             # run supplementary SFE/MEL
  python run_all.py --methods all      # run all 15 methods
"""
import sys
import os
import json
import time
import argparse
import warnings
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from runner import run_experiment
from methods import MAIN_METHODS, SUPPLEMENTARY_METHODS, ALL_METHODS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "main_benchmark")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output", "main_benchmark")

DATASET_FILES = sorted([
    f for f in os.listdir(DATA_DIR) if f.endswith(".csv")
])


def get_methods(args):
    if "all" in args.methods:
        return ALL_METHODS
    if args.supp:
        return SUPPLEMENTARY_METHODS
    if args.methods:
        return {k: v for k, v in ALL_METHODS.items() if k in args.methods}
    return MAIN_METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=[],
                        help="Which methods to run (default: all 13 main)")
    parser.add_argument("--supp", action="store_true",
                        help="Run supplementary methods (SFE, MEL)")
    parser.add_argument("--datasets", nargs="+", default=[],
                        help="Specific dataset names (e.g., Lung_GSE19804)")
    parser.add_argument("--skip-done", action="store_true", default=True,
                        help="Skip already-completed experiments")
    args = parser.parse_args()

    methods = get_methods(args)
    datasets = DATASET_FILES
    if args.datasets:
        datasets = [d for d in DATASET_FILES
                    if any(ds in d for ds in args.datasets)]

    total = len(methods) * len(datasets)
    print(f"\n{'='*60}")
    print(f"BIB Benchmark Run")
    print(f"Methods ({len(methods)}): {', '.join(methods.keys())}")
    print(f"Datasets ({len(datasets)}): {', '.join(d.replace('.csv','') for d in datasets)}")
    print(f"Total experiments: {total}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_summaries = []
    t_start = time.time()
    done = 0

    for method_name, method_fn in methods.items():
        for ds_file in datasets:
            ds_name = ds_file.replace(".csv", "")
            ds_path = os.path.join(DATA_DIR, ds_file)
            out_dir = os.path.join(OUTPUT_ROOT, method_name, ds_name)

            print(f"[{done+1}/{total}] {method_name} × {ds_name}")
            t0 = time.time()
            summary = run_experiment(ds_path, method_fn, method_name, out_dir)
            elapsed = time.time() - t0
            if summary:
                all_summaries.append(summary)
            done += 1
            elapsed_total = time.time() - t_start
            eta = (elapsed_total / done) * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] elapsed={elapsed_total/60:.1f}min  ETA={eta/60:.1f}min\n")

    # Save aggregated results
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        out_csv = os.path.join(OUTPUT_ROOT, "summary_all.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n{'='*60}")
        print(f"ALL DONE. Summary saved: {out_csv}")
        print(df[["method", "dataset", "auc_mean", "auc_std",
                   "n_features_mean", "stability"]].to_string(index=False))


if __name__ == "__main__":
    main()

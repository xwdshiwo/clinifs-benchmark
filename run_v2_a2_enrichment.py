"""A2: Enrichment analysis with caching and offline fallback.

For each (method, dataset, k=K_TARGET), get consensus features (>= CONSENSUS_THRESHOLD
folds), map probes → gene symbols (GPL annotation cached), then run enrichment via
the 3-layer backend from ``src/enrichment_utils.py``:

    1. Per-gene-set hash cache (``output/v2/cache_a2/gprofiler/<sha1>.json``)
    2. g:Profiler online API (with proxy + retries; HTTPS_PROXY env or --proxy flag)
    3. Offline hypergeometric enrichment over local GMT libraries
       (``refs/enrichment_gmt/*.gmt``, obtainable via tools/download_enrichment_gmt.py)

CLI:
    python run_v2_a2_enrichment.py                 # auto (online → offline)
    python run_v2_a2_enrichment.py --offline       # force offline even if net is up
    python run_v2_a2_enrichment.py --proxy http://127.0.0.1:7897
    python run_v2_a2_enrichment.py --k 10 --consensus 0.6
"""
import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from enrichment_utils import EnrichmentBackend, load_probe_to_gene_cached  # noqa: E402

E2_DIR = os.path.join(ROOT, "output", "v2", "E2_constrained_k")
OUT_DIR = os.path.join(ROOT, "output", "v2", "A2_enrichment")
CACHE_DIR = os.path.join(ROOT, "output", "v2", "cache_a2")
GMT_DIR = os.path.join(ROOT, "refs", "enrichment_gmt")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DATASETS_PANEL = sorted(f.replace(".csv", "") for f in
                        os.listdir(os.path.join(ROOT, "data", "main_benchmark"))
                        if f.endswith(".csv"))
METHODS_TO_ANALYZE = ["anova", "mrmr", "relieff", "mi", "variancethreshold",
                      "l1_logistic", "elasticnet", "linearsvc_l1",
                      "boruta", "extratrees", "rfecv",
                      "ga", "bpso", "mel", "sfe"]
K_TARGET = 20
CONSENSUS_THRESHOLD = 0.5


def _resolve_gpl_dir() -> str:
    cand = []
    env_gpl_dir = os.environ.get("CLINIFS_GPL_DIR")
    if env_gpl_dir:
        cand.append(env_gpl_dir)
    cand.extend([
        os.path.join(os.path.dirname(ROOT), "cumida_data", "gpl_annotations"),
        os.path.join(ROOT, "data", "gpl_annotations"),
    ])
    for d in cand:
        if os.path.isdir(d):
            return d
    return cand[0]


def get_consensus_genes(method, dataset, k, probe_to_gene,
                        threshold=CONSENSUS_THRESHOLD):
    """Return list of gene symbols from consensus probes."""
    # Naming alias: A3 uses "variancethreshold" but E2 dir is "variance".
    e2_method = "variance" if method == "variancethreshold" else method
    sel_path = os.path.join(E2_DIR, e2_method, f"k{k:02d}", dataset,
                              "selected_features.json")
    sum_path = os.path.join(E2_DIR, e2_method, f"k{k:02d}", dataset, "summary.json")
    if not os.path.exists(sel_path) or not os.path.exists(sum_path):
        return None, None
    # Load summary to get feature_frequency_top20
    with open(sum_path) as f:
        summary = json.load(f)
    # The feature names are probe names (column names from the data CSV)
    freq = summary.get("feature_frequency_top20", {})
    n_folds = summary.get("n_folds", 25)

    probes = [name for name, count in freq.items()
              if count / n_folds >= threshold]
    if not probes:
        # Fallback: use top-k from most frequent
        probes = list(freq.keys())[:k]

    # Map to gene symbols
    genes = []
    _invalid = {"nan", "NaN", "", "-", "--", "---", "----", "?"}
    for p in probes:
        g = probe_to_gene.get(p)
        if g and g not in _invalid and not g.startswith("---"):
            genes.append(g)
    # Dedup preserving order
    seen = set()
    genes_unique = [g for g in genes if not (g in seen or seen.add(g))]
    return probes, genes_unique


def _collect_summary_rows(data: dict, method: str, dataset: str,
                          all_rows: list) -> None:
    rows = data.get("result_rows") or []
    backend = data.get("backend", "unknown")
    for r in rows[:5]:
        all_rows.append({
            "method": method, "dataset": dataset,
            "backend": backend,
            "n_genes_queried": data.get("n_genes", 0),
            "source": r.get("source"),
            "term_id": r.get("native"),
            "term_name": r.get("name"),
            "p_value": r.get("p_value"),
            "p_fdr_bh": r.get("p_fdr_bh"),
            "intersection_size": r.get("intersection_size"),
            "query_size": r.get("query_size"),
            "term_size": r.get("term_size"),
        })


def main():
    parser = argparse.ArgumentParser(description="A2 Enrichment Analysis with caching + offline fallback")
    parser.add_argument("--k", type=int, default=K_TARGET,
                        help=f"Panel size for consensus (default {K_TARGET})")
    parser.add_argument("--consensus", type=float, default=CONSENSUS_THRESHOLD,
                        help=f"Minimum fold fraction for consensus (default {CONSENSUS_THRESHOLD})")
    parser.add_argument("--offline", action="store_true",
                        help="Skip online g:Profiler; use local GMT only")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP(S) proxy for g:Profiler (e.g. http://127.0.0.1:7897)")
    parser.add_argument("--rebuild-probe-cache", action="store_true",
                        help="Force re-parse of GPL annotations")
    parser.add_argument("--max-online-failures", type=int, default=5,
                        help="After N consecutive API failures, switch to offline-only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    print("=" * 72)
    print(f"A2 Enrichment  (k={args.k}, consensus={args.consensus}, "
          f"offline={args.offline}, proxy={args.proxy or os.environ.get('HTTPS_PROXY') or 'none'})")
    print("=" * 72)

    # ---- probe → gene (cached) ----
    gpl_dir = _resolve_gpl_dir()
    probe_to_gene = load_probe_to_gene_cached(gpl_dir=gpl_dir,
                                               cache_dir=CACHE_DIR,
                                               force_rebuild=args.rebuild_probe_cache)
    if not probe_to_gene:
        print("[fatal] no probe→gene mapping available; GPL dir inspected:", gpl_dir)
        return
    print(f"[probe2gene] loaded {len(probe_to_gene)} mappings from {gpl_dir}")

    # ---- enrichment backend ----
    backend = EnrichmentBackend(
        cache_dir=CACHE_DIR,
        gmt_dir=GMT_DIR,
        proxy=args.proxy,
        max_online_failures=args.max_online_failures,
        force_offline=args.offline,
    )
    n_libs = len(backend.libraries)
    print(f"[backend] offline GMT libraries: {n_libs} "
          f"(dir={GMT_DIR}; offline_available={backend.offline_available})")
    if args.offline and not backend.offline_available:
        print("[fatal] --offline requested but no GMT files found.\n"
              "        Run: python tools/download_enrichment_gmt.py")
        return

    all_rows: list = []
    stats = {"cache": 0, "online": 0, "offline": 0, "empty": 0, "unavailable": 0, "skip": 0}
    t0 = time.time()

    for m in METHODS_TO_ANALYZE:
        for ds in DATASETS_PANEL:
            out_file = os.path.join(OUT_DIR, f"enrich_{m}_{ds}_k{args.k}.json")

            # Respect prior success (backend value 'online' or 'offline' with result_rows)
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r", encoding="utf-8") as f:
                        data_prev = json.load(f)
                    if data_prev.get("result_rows"):
                        stats["cache"] += 1
                        _collect_summary_rows(data_prev, m, ds, all_rows)
                        continue
                except Exception:
                    pass  # fall through and recompute

            probes, genes = get_consensus_genes(m, ds, args.k, probe_to_gene,
                                                 threshold=args.consensus)
            if not genes:
                stats["skip"] += 1
                print(f"  [skip] {m} × {ds}: no genes mapped")
                continue

            print(f"  [query] {m:<12s} × {ds[:40]:<40s}  "
                  f"{len(genes):3d} genes / {len(probes):3d} probes", end="", flush=True)
            res = backend.enrich(genes)
            src = res.get("backend", "?")
            stats[src] = stats.get(src, 0) + 1

            data = {
                "method": m, "dataset": ds, "k": args.k,
                "probes": probes, "genes": genes,
                "n_probes": len(probes), "n_genes": len(genes),
                "backend": src,
                "gene_hash": res.get("gene_hash"),
                "result_rows": res.get("result_rows") or [],
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, default=str, ensure_ascii=False)

            if data["result_rows"]:
                top = data["result_rows"][0]
                print(f"  → {src:<9s} top=\"{(top.get('name') or '')[:45]}\" "
                      f"p={top.get('p_value', float('nan')):.2e}")
                _collect_summary_rows(data, m, ds, all_rows)
            else:
                print(f"  → {src:<9s} no significant terms")

            # Rate limit only for real online calls
            if src == "online":
                time.sleep(0.3)

    # ---- persist summary ----
    if all_rows:
        out_csv = os.path.join(OUT_DIR, "enrichment_summary.csv")
        pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[saved] {out_csv}  ({len(all_rows)} rows)")

    # ---- stats ----
    wall = time.time() - t0
    print("\n" + "=" * 72)
    print(f"A2 complete in {wall/60:.1f} min  |  backend stats:  "
          + "  ".join(f"{k}={v}" for k, v in stats.items()))
    if stats.get("unavailable"):
        print("[warn] some queries had no backend available — install gprofiler-official\n"
              "       or run tools/download_enrichment_gmt.py to enable offline mode.")
    print("=" * 72)


if __name__ == "__main__":
    main()

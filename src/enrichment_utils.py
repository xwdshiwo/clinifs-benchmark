"""Enrichment utilities with 3-layer fallback: cache → g:Profiler (online) → local GMT (offline).

Design goals:
  - Persistent caching for probe→gene mapping (SOFT files are 100+ MB; parse once).
  - Per-gene-set hash cache for g:Profiler results (same gene list in different
    (method, dataset) pairs reuses a single API call).
  - Proxy-aware online layer (reads HTTP_PROXY / HTTPS_PROXY or explicit arg).
  - Automatic fallback to local GMT-based hypergeometric enrichment when:
      (a) gprofiler-official is not installed, or
      (b) three consecutive API attempts fail, or
      (c) ``force_offline=True`` is passed.
  - Output schema is compatible with g:Profiler's DataFrame records:
      source, native, name, p_value, intersection_size, query_size, term_size
    so downstream code (fig08, enrichment_summary.csv) does not need changes.

Usage:
    from enrichment_utils import enrich_with_fallback, EnrichmentBackend

    backend = EnrichmentBackend(
        cache_dir="output/v2/cache_a2",
        gmt_dir="refs/enrichment_gmt",
        proxy=None,               # or "http://127.0.0.1:7897"
        max_online_failures=5,    # after N failures, stick to offline mode
    )
    result = backend.enrich(genes=["TP53", "BRCA1", "MYC"])
    # result is a list of dicts (same schema as g:Profiler records)

Offline hypergeometric p-value:
    P(X >= k) = scipy.stats.hypergeom.sf(k-1, N, K, n)
    where N=universe, K=pathway size, n=query size, k=intersection.
    Multiple testing is controlled via Benjamini–Hochberg FDR.
"""
from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe → gene caching
# ---------------------------------------------------------------------------

def _parse_soft_gpl_streaming(fp: str) -> dict:
    """Stream-parse GEO SOFT GPL to probe_id → gene_symbol."""
    mapping: dict = {}
    in_table = False
    probe_idx = gene_idx = None
    header_seen = False
    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not in_table:
                if line.startswith("!platform_table_begin"):
                    in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            cols = line.split("\t")
            if not header_seen:
                header_seen = True
                for i, c in enumerate(cols):
                    cl = c.lower().strip()
                    if probe_idx is None and (cl == "id" or cl == "probe_id"
                                              or "probe" in cl or "affy" in cl):
                        probe_idx = i
                    if gene_idx is None and ("gene symbol" in cl
                                             or cl == "gene_symbol"
                                             or cl == "symbol"
                                             or cl == "gene"):
                        gene_idx = i
                if probe_idx is None or gene_idx is None:
                    return mapping
                continue
            if len(cols) <= max(probe_idx, gene_idx):
                continue
            p = cols[probe_idx].strip()
            g = cols[gene_idx].strip()
            if not p or not g or g in ("nan", "--", "NA"):
                continue
            if "///" in g:
                g = g.split("///")[0].strip()
            if g:
                mapping[p] = g
    return mapping


def load_probe_to_gene_cached(gpl_dir: str, cache_dir: str,
                              force_rebuild: bool = False) -> dict:
    """Load probe→gene map with persistent JSON cache.

    Cache key incorporates the sorted list of source file names + mtimes so that
    modification to any GPL source invalidates the cache automatically.
    """
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "probe2gene.json")
    meta_path = os.path.join(cache_dir, "probe2gene.meta.json")

    # Build source signature
    sig_entries = []
    if os.path.isdir(gpl_dir):
        for f in sorted(os.listdir(gpl_dir)):
            fp = os.path.join(gpl_dir, f)
            if os.path.isfile(fp) and f.lower().endswith((".txt", ".csv", ".xlsx")):
                sig_entries.append((f, os.path.getsize(fp), int(os.path.getmtime(fp))))
    sig = hashlib.sha1(json.dumps(sig_entries, sort_keys=True).encode()).hexdigest()

    if (not force_rebuild) and os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("signature") == sig:
                with open(cache_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                logger.info("[cache] loaded probe2gene (%d entries) from %s",
                            len(mapping), cache_path)
                return mapping
        except Exception as e:
            logger.warning("[cache] probe2gene cache read failed: %s", e)

    # Rebuild
    mapping: dict = {}
    if not os.path.isdir(gpl_dir):
        logger.warning("[probe2gene] GPL dir missing: %s", gpl_dir)
        return mapping

    for f in sorted(os.listdir(gpl_dir)):
        fp = os.path.join(gpl_dir, f)
        fl = f.lower()
        try:
            if fl.endswith(".txt"):
                sub = _parse_soft_gpl_streaming(fp)
                if sub:
                    mapping.update(sub)
                    logger.info("  parsed %s: +%d (SOFT)", f, len(sub))
            elif fl.endswith(".xlsx"):
                df = pd.read_excel(fp)
                n0 = len(mapping)
                _df_to_map(df, mapping)
                logger.info("  parsed %s: +%d (xlsx)", f, len(mapping) - n0)
            elif fl.endswith(".csv"):
                df = pd.read_csv(fp)
                n0 = len(mapping)
                _df_to_map(df, mapping)
                logger.info("  parsed %s: +%d (csv)", f, len(mapping) - n0)
        except Exception as e:
            logger.warning("  [skip] failed to parse %s: %s", f, e)

    # Persist
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"signature": sig, "n_entries": len(mapping),
                       "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        logger.info("[cache] saved probe2gene (%d entries) to %s",
                    len(mapping), cache_path)
    except Exception as e:
        logger.warning("[cache] probe2gene cache save failed: %s", e)

    return mapping


def _df_to_map(df, mapping: dict) -> None:
    probe_col = gene_col = None
    for c in df.columns:
        cl = str(c).lower()
        if probe_col is None and ("probe" in cl or cl == "id" or "affy" in cl):
            probe_col = c
        if gene_col is None and ("gene_symbol" in cl or "symbol" in cl
                                 or "gene symbol" in cl or cl == "gene"):
            gene_col = c
    if not (probe_col and gene_col):
        return
    for _, r in df.iterrows():
        p = str(r[probe_col]).strip()
        g = str(r[gene_col]).strip()
        if p and g and p != "nan" and g not in ("nan", "--"):
            if "///" in g:
                g = g.split("///")[0].strip()
            mapping[p] = g


# ---------------------------------------------------------------------------
# Gene-set hashing
# ---------------------------------------------------------------------------

def gene_set_hash(genes: Iterable[str]) -> str:
    """Stable 16-char hash of a gene symbol list (case-insensitive, order-free)."""
    key = ",".join(sorted({g.upper().strip() for g in genes if g}))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# GMT loader (local offline DB)
# ---------------------------------------------------------------------------

def load_gmt_files(gmt_dir: str) -> dict:
    """Load *.gmt files in a directory.

    GMT format: each line is ``term_name<TAB>description<TAB>gene1<TAB>gene2...``

    Returns a dict::
        {
            "KEGG_2021_Human": {
                "term_id_or_name": {"name": ..., "genes": set([...])},
                ...
            },
            ...
        }
    The outer key is derived from the filename (without .gmt suffix).
    """
    libraries: dict = {}
    if not os.path.isdir(gmt_dir):
        return libraries
    for f in sorted(os.listdir(gmt_dir)):
        if not f.lower().endswith(".gmt"):
            continue
        lib_name = os.path.splitext(f)[0]
        fp = os.path.join(gmt_dir, f)
        terms: dict = {}
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 3:
                        continue
                    term_name = parts[0].strip()
                    description = parts[1].strip()
                    genes = {g.strip().upper() for g in parts[2:] if g.strip()}
                    if not term_name or len(genes) < 3:
                        continue
                    terms[term_name] = {"name": term_name,
                                        "description": description,
                                        "genes": genes}
            libraries[lib_name] = terms
            logger.info("[gmt] loaded %s: %d terms", lib_name, len(terms))
        except Exception as e:
            logger.warning("[gmt] failed to load %s: %s", f, e)
    return libraries


def _lib_name_to_source(lib_name: str) -> str:
    """Map GMT file stem to a short source tag (compatible with g:Profiler)."""
    ln = lib_name.upper()
    if "KEGG" in ln:
        return "KEGG"
    if "HALLMARK" in ln or "MSIGDB_HALLMARK" in ln:
        return "HALLMARK"
    if "REACTOME" in ln or "REAC" in ln:
        return "REAC"
    if "GO_BIOLOGICAL_PROCESS" in ln or "GO_BP" in ln:
        return "GO:BP"
    if "GO_MOLECULAR_FUNCTION" in ln or "GO_MF" in ln:
        return "GO:MF"
    if "GO_CELLULAR_COMPONENT" in ln or "GO_CC" in ln:
        return "GO:CC"
    if "WIKIPATHWAYS" in ln:
        return "WP"
    if "HP" in ln and "HUMAN_PHENOTYPE" in ln:
        return "HP"
    return lib_name  # fallback


# ---------------------------------------------------------------------------
# Offline hypergeometric enrichment
# ---------------------------------------------------------------------------

def hypergeom_enrichment(query_genes: Iterable[str],
                          libraries: dict,
                          universe_size: int = 20000,
                          fdr_alpha: float = 0.05,
                          min_overlap: int = 2,
                          max_rows: int = 200) -> list:
    """Hypergeometric enrichment across all terms in the loaded GMT libraries.

    Parameters
    ----------
    query_genes : iterable of str
    libraries   : dict as returned by :func:`load_gmt_files`
    universe_size : total human protein-coding gene count used as population N.
                    20 000 is standard for MSigDB/Enrichr-style analyses.
    fdr_alpha    : BH-FDR threshold for reporting (rows above still returned if
                   ``max_rows`` has capacity, but ``p_value`` is the raw one).
    min_overlap  : drop terms with intersection < this.

    Returns list of dicts (g:Profiler-compatible).
    """
    try:
        from scipy.stats import hypergeom
    except ImportError as e:
        raise RuntimeError("scipy required for offline enrichment") from e

    q = {g.upper().strip() for g in query_genes if g}
    if not q:
        return []
    n = len(q)
    N = max(universe_size, n * 10)

    rows = []
    for lib_name, terms in libraries.items():
        source = _lib_name_to_source(lib_name)
        for term_name, info in terms.items():
            gene_set = info["genes"]
            K = len(gene_set)
            if K < 3:
                continue
            inter = q & gene_set
            k = len(inter)
            if k < min_overlap:
                continue
            # P(X >= k) = sf(k-1, N, K, n)
            pv = float(hypergeom.sf(k - 1, N, K, n))
            rows.append({
                "source": source,
                "native": term_name,
                "name": info.get("description") or term_name,
                "p_value": pv,
                "intersection_size": k,
                "query_size": n,
                "term_size": K,
                "intersections": sorted(inter),
            })

    if not rows:
        return []

    rows.sort(key=lambda r: r["p_value"])
    # Benjamini–Hochberg FDR (attach as p_fdr but keep 'p_value' raw for compat)
    m = len(rows)
    for rank, r in enumerate(rows, start=1):
        r["p_fdr_bh"] = min(1.0, r["p_value"] * m / rank)
    return rows[:max_rows]


# ---------------------------------------------------------------------------
# Online g:Profiler with cache + fallback
# ---------------------------------------------------------------------------

class EnrichmentBackend:
    """Unified cached/online/offline enrichment façade."""

    def __init__(self,
                 cache_dir: str,
                 gmt_dir: str | None = None,
                 proxy: str | None = None,
                 organism: str = "hsapiens",
                 sources: tuple = ("GO:BP", "GO:MF", "KEGG", "REAC", "HP"),
                 max_online_failures: int = 5,
                 universe_size: int = 20000,
                 force_offline: bool = False) -> None:
        self.cache_dir = cache_dir
        self.gprof_cache = os.path.join(cache_dir, "gprofiler")
        os.makedirs(self.gprof_cache, exist_ok=True)
        self.gmt_dir = gmt_dir
        self.proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        self.organism = organism
        self.sources = list(sources)
        self.max_online_failures = max_online_failures
        self.universe_size = universe_size
        self.force_offline = force_offline

        self._consecutive_failures = 0
        self._online_disabled = bool(force_offline)
        self._libraries: dict | None = None  # lazy load

    # ---- accessors ----
    @property
    def libraries(self) -> dict:
        if self._libraries is None:
            self._libraries = (load_gmt_files(self.gmt_dir) if self.gmt_dir else {})
        return self._libraries

    @property
    def offline_available(self) -> bool:
        return bool(self.libraries)

    # ---- cache helpers ----
    def _cache_path(self, gene_hash: str) -> str:
        return os.path.join(self.gprof_cache, f"{gene_hash}.json")

    def _load_cache(self, gene_hash: str):
        fp = self._cache_path(gene_hash)
        if not os.path.exists(fp):
            return None
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, gene_hash: str, payload: dict) -> None:
        fp = self._cache_path(gene_hash)
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
        except Exception as e:
            logger.warning("[cache] failed to write %s: %s", fp, e)

    # ---- online path ----
    def _call_gprofiler(self, genes: list) -> list | None:
        try:
            from gprofiler import GProfiler
        except ImportError:
            logger.warning("[online] gprofiler-official not installed; disabling online path")
            self._online_disabled = True
            return None

        # Proxy: gprofiler-official uses requests; set env locally to ensure it honors proxy
        env_bak = {}
        if self.proxy:
            for k in ("HTTP_PROXY", "HTTPS_PROXY"):
                env_bak[k] = os.environ.get(k)
                os.environ[k] = self.proxy

        try:
            gp = GProfiler(return_dataframe=True)
            df = gp.profile(organism=self.organism, query=genes,
                            sources=self.sources, no_iea=False)
            return df.to_dict("records") if df is not None and len(df) > 0 else []
        except Exception as e:
            logger.warning("  [online] g:Profiler call failed: %s", e)
            return None
        finally:
            if self.proxy:
                for k, v in env_bak.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def _query_online_with_retry(self, genes: list, max_attempts: int = 3) -> list | None:
        for attempt in range(1, max_attempts + 1):
            res = self._call_gprofiler(genes)
            if res is not None:
                self._consecutive_failures = 0
                return res
            if self._online_disabled:  # ImportError path
                return None
            time.sleep(3 * attempt)
        # All attempts failed
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_online_failures:
            logger.warning("[online] %d consecutive failures; disabling online path and "
                           "falling back to offline GMT for remaining queries.",
                           self._consecutive_failures)
            self._online_disabled = True
        return None

    # ---- public entry ----
    def enrich(self, genes: list) -> dict:
        """Return an enrichment record for a given gene list.

        Output dict::
            {
                "gene_hash": "...",
                "n_genes": int,
                "backend": "cache" | "online" | "offline" | "empty",
                "result_rows": [ ... g:Profiler-compatible dicts ... ],
            }
        """
        genes_clean = sorted({g.upper().strip() for g in genes if g})
        if not genes_clean:
            return {"gene_hash": "", "n_genes": 0, "backend": "empty", "result_rows": []}

        h = gene_set_hash(genes_clean)
        cached = self._load_cache(h)
        if cached is not None:
            # Preserve the original backend tag for auditability but mark this
            # particular call as a cache hit so callers can distinguish.
            cached["original_backend"] = cached.get("backend", "unknown")
            cached["backend"] = "cache"
            return cached

        # Try online unless disabled
        rows = None
        if not self._online_disabled:
            rows = self._query_online_with_retry(genes_clean)

        if rows is not None:
            payload = {"gene_hash": h, "n_genes": len(genes_clean),
                       "backend": "online", "result_rows": rows}
            self._save_cache(h, payload)
            return payload

        # Offline fallback
        if self.offline_available:
            rows_off = hypergeom_enrichment(genes_clean, self.libraries,
                                             universe_size=self.universe_size)
            payload = {"gene_hash": h, "n_genes": len(genes_clean),
                       "backend": "offline", "result_rows": rows_off,
                       "offline_note": "Hypergeometric test on local GMT libraries"}
            self._save_cache(h, payload)
            return payload

        # Neither online nor offline available
        logger.error("[enrich] No backend available for query of %d genes; "
                     "returning empty (install gprofiler-official or place GMT "
                     "files in %s).", len(genes_clean), self.gmt_dir)
        return {"gene_hash": h, "n_genes": len(genes_clean),
                "backend": "unavailable", "result_rows": []}


# Convenience functional API ------------------------------------------------

def enrich_with_fallback(genes: list, cache_dir: str,
                          gmt_dir: str | None = None,
                          proxy: str | None = None,
                          force_offline: bool = False) -> dict:
    """One-shot wrapper creating a transient backend (no cross-call reuse of failure counter)."""
    backend = EnrichmentBackend(cache_dir=cache_dir, gmt_dir=gmt_dir,
                                 proxy=proxy, force_offline=force_offline)
    return backend.enrich(genes)

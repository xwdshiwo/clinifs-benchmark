"""Download public GMT gene-set libraries for offline enrichment.

Sources (all **free / public**):
  * Enrichr: https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=<NAME>
      - KEGG_2021_Human
      - GO_Biological_Process_2023
      - GO_Molecular_Function_2023
      - MSigDB_Hallmark_2020
      - Reactome_2022
      - WikiPathways_2024_Human
      - DisGeNET  (disease-gene, optional)
  * WikiPathways direct: https://data.wikipathways.org/current/gmt/

Enrichr returns GMT with ``description`` field containing the term ID URL; we
normalize to the ``name<TAB>description<TAB>gene1<TAB>...<TAB>geneN`` format.

Usage:
    python tools/download_enrichment_gmt.py                      # all defaults
    python tools/download_enrichment_gmt.py --libs KEGG Hallmark # only these
    python tools/download_enrichment_gmt.py --proxy http://127.0.0.1:7897
    python tools/download_enrichment_gmt.py --force              # re-download existing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import urllib.request as urlreq
    import urllib.error as urlerr
except ImportError:
    print("urllib missing (impossible on CPython)", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "refs" / "enrichment_gmt"

ENRICHR_TEMPLATE = ("https://maayanlab.cloud/Enrichr/"
                    "geneSetLibrary?mode=text&libraryName={lib}")

# Canonical library set (name -> Enrichr library identifier)
DEFAULT_LIBS = {
    "KEGG_2021_Human":              "KEGG_2021_Human",
    "MSigDB_Hallmark_2020":         "MSigDB_Hallmark_2020",
    "Reactome_2022":                "Reactome_2022",
    "GO_Biological_Process_2023":   "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023":   "GO_Molecular_Function_2023",
    "WikiPathways_2024_Human":      "WikiPathways_2024_Human",
}

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppData enrichment_gmt_downloader/1.0 (academic use)")


def build_opener(proxy: str | None):
    handlers = []
    if proxy:
        handlers.append(urlreq.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urlreq.build_opener(*handlers)
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def download_one(opener, lib_id: str, dest: Path, timeout: int = 120) -> tuple[bool, int, str]:
    """Return (success, n_lines, error_msg)."""
    url = ENRICHR_TEMPLATE.format(lib=lib_id)
    try:
        req = urlreq.Request(url)
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except (urlerr.URLError, urlerr.HTTPError, TimeoutError) as e:
        return False, 0, f"{type(e).__name__}: {e}"
    except Exception as e:
        return False, 0, f"{type(e).__name__}: {e}"

    # Enrichr returns one term per line: "term_name\tgene1,1.0\tgene2,1.0\t..."
    # We normalize to the canonical GMT 3-column form expected by enrichment_utils:
    #     term_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
    lines_out = []
    for line in raw.splitlines():
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        term = parts[0].strip()
        gene_fields = parts[1:]
        genes = []
        for g in gene_fields:
            g = g.strip()
            if not g:
                continue
            if "," in g:  # "GENE,1.0" form from Enrichr
                g = g.split(",", 1)[0].strip()
            if g:
                genes.append(g)
        if not term or len(genes) < 3:
            continue
        # Description = term (Enrichr doesn't separately expose id; name is primary key)
        lines_out.append(term + "\t" + term + "\t" + "\t".join(genes))

    if not lines_out:
        return False, 0, "empty response or failed parsing"

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines_out) + "\n")
    return True, len(lines_out), ""


def main():
    ap = argparse.ArgumentParser(description="Download public GMT libraries for offline enrichment")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output dir (default {DEFAULT_OUT})")
    ap.add_argument("--libs", nargs="*", default=None,
                    help=f"Subset of library names to fetch. Available: {list(DEFAULT_LIBS)}")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
                    help="HTTP(S) proxy URL (e.g. http://127.0.0.1:7897). Defaults to env.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if file already exists")
    ap.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    libs = args.libs or list(DEFAULT_LIBS)
    unknown = [l for l in libs if l not in DEFAULT_LIBS]
    if unknown:
        print(f"[warn] Unknown libraries (will be passed through as-is): {unknown}")

    proxy = args.proxy
    opener = build_opener(proxy)
    print(f"Target dir : {out_dir}")
    print(f"Proxy      : {proxy or 'none'}")
    print(f"Libraries  : {libs}")
    print("=" * 72)

    ok = skipped = failed = 0
    for name in libs:
        lib_id = DEFAULT_LIBS.get(name, name)
        dest = out_dir / f"{name}.gmt"
        if dest.exists() and not args.force:
            print(f"[skip] {name}  ({dest.stat().st_size/1024:.0f} KB exists; --force to overwrite)")
            skipped += 1
            continue
        print(f"[get ] {name}  <- {lib_id}")
        t0 = time.time()
        success, n_terms, err = download_one(opener, lib_id, dest, timeout=args.timeout)
        dt = time.time() - t0
        if success:
            print(f"       ok  {n_terms} terms  {dest.stat().st_size/1024:.0f} KB  in {dt:.1f}s")
            ok += 1
        else:
            print(f"       FAILED  {err}")
            failed += 1

    print("=" * 72)
    print(f"Done. ok={ok}  skipped={skipped}  failed={failed}")
    if failed:
        print("Tips:")
        print("  * If network fails, retry with --proxy http://127.0.0.1:7897")
        print("  * Enrichr occasionally rate-limits; re-run after a minute")
        print("  * Alternative: manually place any .gmt files into", out_dir)
        sys.exit(2)


if __name__ == "__main__":
    main()

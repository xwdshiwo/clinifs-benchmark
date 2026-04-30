# `refs/enrichment_gmt/` — offline enrichment gene-set libraries

This directory stores gene-set libraries in `*.gmt` format. They are used by
`src/enrichment_utils.py` for local hypergeometric enrichment analysis when the
online g:Profiler API is unavailable.

## 1. Quick download

Run the downloader from the project root:

```powershell
python tools/download_enrichment_gmt.py
# use a proxy if required
python tools/download_enrichment_gmt.py --proxy http://127.0.0.1:7897
# download only the two core libraries
python tools/download_enrichment_gmt.py --libs KEGG_2021_Human MSigDB_Hallmark_2020
```

The downloader retrieves the following default libraries from the
[Enrichr Gene Set Library API](https://maayanlab.cloud/Enrichr/#libraries),
which is free and does not require registration:

| Library | Source | Typical term count | Use case |
|---|---|---:|---|
| `KEGG_2021_Human.gmt` | KEGG via Enrichr | ~320 | Canonical pathway enrichment |
| `MSigDB_Hallmark_2020.gmt` | Broad MSigDB Hallmark via Enrichr | 50 | Cancer hallmark pathways |
| `Reactome_2022.gmt` | Reactome via Enrichr | ~1600 | Molecular reaction hierarchy |
| `GO_Biological_Process_2023.gmt` | GO via Enrichr | ~6000 | General biological processes |
| `GO_Molecular_Function_2023.gmt` | GO via Enrichr | ~1700 | Molecular functions |
| `WikiPathways_2024_Human.gmt` | WikiPathways via Enrichr | ~800 | Community-curated pathways |

The downloader normalizes Enrichr's `term<TAB>gene,1.0<TAB>...` format into
standard three-column GMT records:
`term<TAB>description<TAB>gene1<TAB>gene2<TAB>...`.

## 2. Manual download

### Direct Enrichr download

Raw URL pattern:

```
https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human
```

Other library names are listed at <https://maayanlab.cloud/Enrichr/#libraries>.
Save downloaded files as `<LibraryName>.gmt` in this directory.

### WikiPathways mirror

```
https://data.wikipathways.org/current/gmt/wikipathways-<DATE>-gmt-Homo_sapiens.gmt
```

### MSigDB

MSigDB is available at <https://www.gsea-msigdb.org/gsea/msigdb/>. Academic use
is free but registration is required. Downloaded files can be placed in this
directory; the parser identifies compatible files from content.

## 3. File format

Standard GMT (Gene Matrix Transposed) stores one term per line:

```
Hallmark: E2F Targets<TAB>HALLMARK_E2F_TARGETS<TAB>TP53<TAB>MYC<TAB>CDK2<TAB>...
```

`src/enrichment_utils.py:load_gmt_files()`:

- Scans all `*.gmt` files in this directory.
- Infers a `source` tag from file names (`KEGG`, `HALLMARK`, `REAC`, `GO:BP`,
  `GO:MF`, `WP`, etc.), keeping output compatible with g:Profiler-style source
  labels.
- Drops terms with fewer than three genes to avoid unstable tiny gene sets.

## 4. Offline enrichment model

For each term gene set $S$ of size $K = |S|$ and query gene list $Q$ of size
$n$, let $k = |Q \cap S|$. The hypergeometric p-value is:

$$P(X \ge k) = \sum_{i=k}^{\min(K,n)} \frac{\binom{K}{i} \binom{N-K}{n-i}}{\binom{N}{n}}$$

The implementation uses `scipy.stats.hypergeom.sf(k-1, N, K, n)`, where `N`
defaults to 20,000 human protein-coding genes. Benjamini-Hochberg FDR
correction is then applied within each library, returning `p_value` (raw) and
`p_fdr_bh`.

## 5. Licenses and citations

| Data source | License / access | Citation |
|---|---|---|
| KEGG | Free for non-commercial academic use | Kanehisa M, Goto S. *Nucleic Acids Res.* 2000;28:27-30. |
| MSigDB Hallmark | Broad Institute EULA; free for academic use | Liberzon A, et al. *Cell Systems* 2015;1:417-425. |
| Reactome | CC0 | Fabregat A, et al. *Nucleic Acids Res.* 2018;46:D649. |
| Gene Ontology | CC BY 4.0 | Ashburner M, et al. *Nat Genet.* 2000;25:25-29. |
| WikiPathways | CC0 | Martens M, et al. *Nucleic Acids Res.* 2021;49:D613. |
| Enrichr distribution | Free for academic use | Kuleshov MV, et al. *Nucleic Acids Res.* 2016;44:W90-W97. |

When these libraries are used for enrichment analysis, cite the corresponding
primary databases. `refs/bibliography.bib` already includes keys such as
`kanehisa2000kegg`, `liberzon2015msigdb`, `ashburner2000go`, and
`kuleshov2016enrichr`.

## 6. `.gitignore` recommendation

Some GMT files can be large (for example, GO Biological Process can be around
10 MB). To avoid accidentally committing large local offline databases, add:

```
refs/enrichment_gmt/*.gmt
!refs/enrichment_gmt/README.md
```

For publication release, this README plus `tools/download_enrichment_gmt.py`
is sufficient to reconstruct the offline enrichment resources.

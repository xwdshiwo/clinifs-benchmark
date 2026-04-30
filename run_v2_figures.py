"""
Generate all V2 paper figures from experiment outputs.
Figures: 1-10 as specified in the plan.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.join(ROOT, "output", "v2")
FIG_DIR = os.path.join(V2_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Global style
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# warm_to_cool palette
WARM_COOL = ["#D77367","#E86254","#EC7E4A","#EC9453","#E8A76B","#EBB884",
             "#DBBD97","#C1BDA6","#A7BCB6","#8DBCC5","#73BCD5","#67A8C2",
             "#5B95AE","#4C809E","#3D6B8F","#2D5680"]

# Method family colors
FAMILY_COLORS = {
    "filter":   "#D77367",
    "embedded": "#EBB884",
    "wrapper":  "#73BCD5",
    "ea":       "#3D6B8F",
    "pipeline": "#EC7E4A",
}
METHOD_FAMILY = {
    "variance":"filter","anova":"filter","mi":"filter","mrmr":"filter","relieff":"filter",
    "l1_logistic":"embedded","elasticnet":"embedded","linearsvc_l1":"embedded",
    "boruta":"embedded","extratrees":"embedded",
    "rfecv":"wrapper","ga":"ea","bpso":"ea","sfe":"ea","mel":"ea",
    "P1_anova_ga":"pipeline","P2_relieff_ga":"pipeline",
    "P3_fusion_bpso":"pipeline","P4_anova_bpso":"pipeline",
}


def load_e2_summaries():
    """Return long-format DataFrame of all E2 summary.json files."""
    rows = []
    base = os.path.join(V2_ROOT, "E2_constrained_k")
    if not os.path.exists(base):
        return pd.DataFrame()
    for method in os.listdir(base):
        m_dir = os.path.join(base, method)
        if not os.path.isdir(m_dir):
            continue
        for k_tag in os.listdir(m_dir):
            k = int(k_tag.replace("k", ""))
            k_dir = os.path.join(m_dir, k_tag)
            for ds in os.listdir(k_dir):
                sp = os.path.join(k_dir, ds, "summary.json")
                if not os.path.exists(sp):
                    continue
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "experiment": "E2",
                    "method": method,
                    "family": METHOD_FAMILY.get(method, "unknown"),
                    "k": k,
                    "dataset": ds,
                    "auc_mean": s.get("auc_mean"),
                    "auc_std":  s.get("auc_std"),
                    "bacc_mean": s.get("bacc_mean"),
                    "f1_mean": s.get("f1_mean"),
                    "mcc_mean": s.get("mcc_mean"),
                    "n_features_mean": s.get("n_features_mean"),
                    "stability_nogueira": s.get("stability_nogueira"),
                    "stability_jaccard":  s.get("stability_jaccard"),
                    "runtime": s.get("runtime_fs_mean_sec"),
                })
    return pd.DataFrame(rows)


def load_e1_summaries():
    rows = []
    base = os.path.join(V2_ROOT, "E1_ea_sparsity")
    if not os.path.exists(base):
        return pd.DataFrame()
    for method in os.listdir(base):
        m_dir = os.path.join(base, method)
        if not os.path.isdir(m_dir):
            continue
        for cfg_tag in os.listdir(m_dir):
            c_dir = os.path.join(m_dir, cfg_tag)
            for ds in os.listdir(c_dir):
                sp = os.path.join(c_dir, ds, "summary.json")
                if not os.path.exists(sp):
                    continue
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "experiment":"E1","method": method,"config": cfg_tag,
                    "dataset": ds,
                    "auc_mean":  s.get("auc_mean"),
                    "n_features_mean": s.get("n_features_mean"),
                    "stability_nogueira": s.get("stability_nogueira"),
                    "alpha": s.get("config",{}).get("alpha"),
                    "beta":  s.get("config",{}).get("beta"),
                })
    return pd.DataFrame(rows)


def load_e3_summaries():
    rows = []
    base = os.path.join(V2_ROOT, "E3_pipelines")
    if not os.path.exists(base):
        return pd.DataFrame()
    for pipeline in os.listdir(base):
        p_dir = os.path.join(base, pipeline)
        if not os.path.isdir(p_dir):
            continue
        for k_tag in os.listdir(p_dir):
            k = int(k_tag.replace("k", ""))
            k_dir = os.path.join(p_dir, k_tag)
            for ds in os.listdir(k_dir):
                sp = os.path.join(k_dir, ds, "summary.json")
                if not os.path.exists(sp):
                    continue
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "experiment":"E3","method": pipeline, "family":"pipeline",
                    "k": k, "dataset": ds,
                    "auc_mean": s.get("auc_mean"),
                    "n_features_mean": s.get("n_features_mean"),
                    "stability_nogueira": s.get("stability_nogueira"),
                })
    return pd.DataFrame(rows)


def load_e4_summaries():
    rows = []
    base = os.path.join(V2_ROOT, "E4_consensus")
    if not os.path.exists(base):
        return pd.DataFrame()
    for method in os.listdir(base):
        m_dir = os.path.join(base, method)
        if not os.path.isdir(m_dir):
            continue
        for seed_tag in os.listdir(m_dir):
            s_dir = os.path.join(m_dir, seed_tag)
            seed = int(seed_tag.replace("seed", ""))
            for ds in os.listdir(s_dir):
                sp = os.path.join(s_dir, ds, "summary.json")
                if not os.path.exists(sp):
                    continue
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "experiment":"E4","method": method, "seed": seed,
                    "dataset": ds,
                    "auc_mean": s.get("auc_mean"),
                    "n_features_mean": s.get("n_features_mean"),
                    "stability_nogueira": s.get("stability_nogueira"),
                })
    return pd.DataFrame(rows)


# ═══════════ Figure generators ═══════════

def fig_k_auc_curves(df, save_path):
    """Fig 3: k vs AUC curves for each method across datasets."""
    if df.empty:
        print("[skip] fig_k_auc: no data")
        return
    datasets = sorted(df["dataset"].unique())
    methods  = ["anova","mrmr","relieff","l1_logistic","elasticnet",
                "boruta","extratrees","rfecv","ga","bpso","mel"]
    methods = [m for m in methods if m in df["method"].unique()]

    n_ds = len(datasets)
    cols = 4; rows = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.8*rows),
                              sharex=True)
    axes = axes.flatten() if n_ds > 1 else [axes]
    cmap = plt.cm.get_cmap("tab20", len(methods))

    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, m in enumerate(methods):
            sub = df[(df["dataset"]==ds) & (df["method"]==m)].sort_values("k")
            if sub.empty:
                continue
            ax.plot(sub["k"], sub["auc_mean"], "-o",
                    label=m, color=cmap(j), markersize=4, linewidth=1.5)
        ax.set_title(ds.replace("_","\n", 1), fontsize=9)
        ax.set_xlabel("k (features)")
        ax.set_ylabel("AUC")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.02)
    for j in range(n_ds, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    plt.suptitle("Figure 3. k vs AUC across methods and datasets", fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_k_stability_curves(df, save_path):
    """Fig 4: k vs Nogueira stability for each method."""
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique())
    methods  = ["anova","mrmr","relieff","l1_logistic","elasticnet",
                "boruta","extratrees","rfecv","ga","bpso","mel"]
    methods = [m for m in methods if m in df["method"].unique()]

    n_ds = len(datasets)
    cols = 4; rows = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.8*rows),
                              sharex=True)
    axes = axes.flatten() if n_ds > 1 else [axes]
    cmap = plt.cm.get_cmap("tab20", len(methods))

    for i, ds in enumerate(datasets):
        ax = axes[i]
        for j, m in enumerate(methods):
            sub = df[(df["dataset"]==ds) & (df["method"]==m)].sort_values("k")
            if sub.empty:
                continue
            ax.plot(sub["k"], sub["stability_nogueira"], "-o",
                    label=m, color=cmap(j), markersize=4, linewidth=1.5)
        ax.set_title(ds.replace("_","\n", 1), fontsize=9)
        ax.set_xlabel("k (features)")
        ax.set_ylabel("Nogueira stability")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    for j in range(n_ds, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    plt.suptitle("Figure 4. k vs Nogueira stability", fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_method_comparison_k10(df, save_path):
    """Fig 5: Method comparison boxplot at k=10 (AUC + stability)."""
    if df.empty:
        return
    sub = df[df["k"] == 10].copy()
    methods = sub.groupby("method")["auc_mean"].mean().sort_values().index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    data_auc = [sub[sub["method"]==m]["auc_mean"].dropna().values for m in methods]
    bp = ax1.boxplot(data_auc, labels=methods, patch_artist=True,
                      medianprops={"color":"red","linewidth":2})
    for patch, m in zip(bp["boxes"], methods):
        fam = METHOD_FAMILY.get(m, "filter")
        patch.set_facecolor(FAMILY_COLORS.get(fam,"#CCCCCC"))
        patch.set_alpha(0.7)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylabel("AUC at k=10")
    ax1.set_title("AUC distribution at k=10")
    ax1.grid(True, alpha=0.3, axis="y")

    data_stab = [sub[sub["method"]==m]["stability_nogueira"].dropna().values
                 for m in methods]
    bp2 = ax2.boxplot(data_stab, labels=methods, patch_artist=True,
                       medianprops={"color":"red","linewidth":2})
    for patch, m in zip(bp2["boxes"], methods):
        fam = METHOD_FAMILY.get(m, "filter")
        patch.set_facecolor(FAMILY_COLORS.get(fam,"#CCCCCC"))
        patch.set_alpha(0.7)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylabel("Nogueira stability at k=10")
    ax2.set_title("Stability distribution at k=10")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)

    fig.suptitle("Figure 5. Method comparison at k=10 (clinical biomarker scenario)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_pipeline_vs_single(df_e2, df_e3, save_path):
    """Fig 6: Two-stage pipelines vs single EA methods."""
    if df_e2.empty or df_e3.empty:
        return
    # Compare P1/P2/P3/P4 vs ga/bpso at each k
    pipelines = df_e3["method"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUC comparison
    ax = axes[0]
    single_ga = df_e2[df_e2["method"] == "ga"].groupby("k")["auc_mean"].mean()
    single_bpso = df_e2[df_e2["method"] == "bpso"].groupby("k")["auc_mean"].mean()
    ax.plot(single_ga.index, single_ga.values, "-o", label="GA (single)",
            color="#3D6B8F", linewidth=2, markersize=7)
    ax.plot(single_bpso.index, single_bpso.values, "-s", label="BPSO (single)",
            color="#5B95AE", linewidth=2, markersize=7)
    colors_p = ["#D77367","#EC7E4A","#E8A76B","#EBB884"]
    for c, p in zip(colors_p, pipelines):
        pipe = df_e3[df_e3["method"] == p].groupby("k")["auc_mean"].mean()
        ax.plot(pipe.index, pipe.values, "--^", label=p, color=c,
                linewidth=1.5, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("k (target features)")
    ax.set_ylabel("Mean AUC (all datasets)")
    ax.set_title("Pipeline vs single EA — AUC")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Stability comparison
    ax = axes[1]
    single_ga_s = df_e2[df_e2["method"] == "ga"].groupby("k")["stability_nogueira"].mean()
    single_bpso_s = df_e2[df_e2["method"] == "bpso"].groupby("k")["stability_nogueira"].mean()
    ax.plot(single_ga_s.index, single_ga_s.values, "-o", label="GA (single)",
            color="#3D6B8F", linewidth=2, markersize=7)
    ax.plot(single_bpso_s.index, single_bpso_s.values, "-s", label="BPSO (single)",
            color="#5B95AE", linewidth=2, markersize=7)
    for c, p in zip(colors_p, pipelines):
        pipe = df_e3[df_e3["method"] == p].groupby("k")["stability_nogueira"].mean()
        ax.plot(pipe.index, pipe.values, "--^", label=p, color=c,
                linewidth=1.5, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Nogueira stability")
    ax.set_title("Pipeline vs single EA — Stability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 6. Two-stage pipelines vs single EA methods", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_ea_sparsity_tradeoff(df, save_path):
    """Fig 7: EA sparsity tuning — alpha/beta scan."""
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, m in zip(axes, ["ga", "bpso"]):
        sub = df[df["method"] == m].copy()
        if sub.empty:
            continue
        ax.scatter(sub["n_features_mean"], sub["auc_mean"],
                    c=sub["stability_nogueira"], cmap="viridis",
                    s=80, alpha=0.7, edgecolors="black", linewidth=0.5)
        for _, r in sub.groupby("config").first().iterrows():
            ax.annotate(r.name, (r["n_features_mean"], r["auc_mean"]),
                        fontsize=8, alpha=0.8)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Nogueira stability")
        ax.set_xlabel("n_features (mean)")
        ax.set_ylabel("AUC (mean)")
        ax.set_title(f"{m.upper()}: α/β tradeoff")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 7. EA sparsity tuning — AUC × n_features × stability",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_consensus_gain(df_e2, df_e4, save_path):
    """Fig 8: EA consensus (multi-seed) stability gain vs single run."""
    if df_e2.empty or df_e4.empty:
        return
    # Compare stability at k=10 between E2 (single GA/BPSO) and E4 (consensus of 10 seeds)
    # For E4, consensus stability: for each dataset + method, treat per-seed
    # selected features across folds as the "folds" list.
    single_k10 = df_e2[(df_e2["k"] == 10) & (df_e2["method"].isin(["ga","bpso"]))]
    single_stab = single_k10.groupby(["method","dataset"])["stability_nogueira"].mean().reset_index()
    single_stab.columns = ["method","dataset","single_stab"]

    # For E4, compute mean of per-seed stability as baseline
    e4_stab = df_e4.groupby(["method","dataset"])["stability_nogueira"].mean().reset_index()
    e4_stab.columns = ["method","dataset","consensus_per_seed_stab"]
    merged = single_stab.merge(e4_stab, on=["method","dataset"], how="inner")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(["ga","bpso"]):
        sub = merged[merged["method"] == m]
        ax.scatter(sub["single_stab"], sub["consensus_per_seed_stab"],
                    label=m, s=100, alpha=0.7,
                    color=WARM_COOL[i*5])
    lim = [min(merged[["single_stab","consensus_per_seed_stab"]].min()) * 0.9,
           max(merged[["single_stab","consensus_per_seed_stab"]].max()) * 1.1]
    ax.plot(lim, lim, "--", color="gray", linewidth=1, label="y=x")
    ax.set_xlabel("Single-run stability (E2 k=10)")
    ax.set_ylabel("Mean per-seed stability (E4 10 seeds)")
    ax.set_title("Figure 8. EA multi-run consensus stability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_pareto_3d(df, save_path):
    """Fig 9: 3D Pareto — AUC × n_features × stability at k=20."""
    if df.empty:
        return
    from mpl_toolkits.mplot3d import Axes3D
    sub = df[df["k"] == 20].copy()
    method_means = sub.groupby("method").agg(
        auc=("auc_mean","mean"),
        nfeat=("n_features_mean","mean"),
        stab=("stability_nogueira","mean")
    ).reset_index()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for _, r in method_means.iterrows():
        fam = METHOD_FAMILY.get(r["method"], "filter")
        color = FAMILY_COLORS.get(fam, "#666666")
        ax.scatter(r["nfeat"], r["stab"], r["auc"], s=180, c=color,
                    edgecolors="black", linewidth=0.8, alpha=0.85)
        ax.text(r["nfeat"], r["stab"], r["auc"] + 0.005,
                r["method"], fontsize=8)
    ax.set_xlabel("n_features")
    ax.set_ylabel("Nogueira stability")
    ax.set_zlabel("AUC")
    ax.set_title("Figure 9. 3D Pareto frontier at k=20")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def fig_recommendation_matrix(df, save_path):
    """Fig 10: Recommendation matrix by (scenario × k) with best method per cell."""
    if df.empty:
        return
    scenarios = {
        "Max AUC":              ("auc_mean", "max"),
        "Max Stability":        ("stability_nogueira", "max"),
        "AUC × Stability":      ("auc_mean", "combo"),
        "Hard datasets (Bladder+Prostate)": ("auc_mean", "hard"),
    }
    k_values = sorted(df["k"].unique())
    matrix = np.full((len(scenarios), len(k_values)), "", dtype="U20")

    for si, (sc_name, (metric, how)) in enumerate(scenarios.items()):
        for ki, k in enumerate(k_values):
            sub = df[df["k"] == k]
            if how == "combo":
                g = sub.groupby("method").agg(
                    auc=("auc_mean","mean"),
                    stab=("stability_nogueira","mean"))
                g["score"] = g["auc"] * (g["stab"].clip(lower=0) + 0.1)
                best = g["score"].idxmax()
            elif how == "hard":
                hard = sub[sub["dataset"].isin(["Bladder_GSE31189","Prostate_GSE6919_U95Av2"])]
                if hard.empty:
                    best = "-"
                else:
                    best = hard.groupby("method")["auc_mean"].mean().idxmax()
            else:
                best = sub.groupby("method")[metric].mean().idxmax()
            matrix[si, ki] = best

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(np.ones_like(matrix, dtype=float), cmap="Greys", alpha=0.1)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(list(scenarios.keys()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i,j], ha="center", va="center",
                    fontsize=10, fontweight="bold")
    ax.set_title("Figure 10. Recommendation matrix: best method per (scenario × k)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("V2 Figure Generation")
    print("=" * 60)
    df_e2 = load_e2_summaries()
    df_e1 = load_e1_summaries()
    df_e3 = load_e3_summaries()
    df_e4 = load_e4_summaries()
    print(f"E1 tasks loaded: {len(df_e1)}")
    print(f"E2 tasks loaded: {len(df_e2)}")
    print(f"E3 tasks loaded: {len(df_e3)}")
    print(f"E4 tasks loaded: {len(df_e4)}")

    # Master table
    master = pd.concat([df_e2, df_e3], ignore_index=True)
    master.to_csv(os.path.join(V2_ROOT, "master_summary.csv"), index=False)
    print(f"Saved master_summary.csv: {len(master)} rows")

    # Figures
    fig_k_auc_curves(df_e2,
                     os.path.join(FIG_DIR, "fig3_k_auc_curves.png"))
    fig_k_stability_curves(df_e2,
                            os.path.join(FIG_DIR, "fig4_k_stability_curves.png"))
    fig_method_comparison_k10(df_e2,
                               os.path.join(FIG_DIR, "fig5_method_comparison_k10.png"))
    fig_pipeline_vs_single(df_e2, df_e3,
                            os.path.join(FIG_DIR, "fig6_pipeline_vs_single.png"))
    fig_ea_sparsity_tradeoff(df_e1,
                              os.path.join(FIG_DIR, "fig7_ea_sparsity.png"))
    fig_consensus_gain(df_e2, df_e4,
                        os.path.join(FIG_DIR, "fig8_consensus_gain.png"))
    fig_pareto_3d(df_e2,
                  os.path.join(FIG_DIR, "fig9_pareto_3d.png"))
    fig_recommendation_matrix(df_e2,
                               os.path.join(FIG_DIR, "fig10_recommendation.png"))


if __name__ == "__main__":
    main()

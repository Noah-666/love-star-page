from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Facmate\Documents\Codex\2026-08-27\plugin-computer-use-openai-bundled")
sys.path.insert(0, str(ROOT / "work" / "pydeps"))
os.environ["MPLCONFIGDIR"] = str(ROOT / "work" / "mplconfig")
os.environ["USERPROFILE"] = str(ROOT / "work")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gseapy as gp

DATA = ROOT / "work" / "data" / "GSE115618_normalized_counts_and_comparisons.xlsx"
OUT = ROOT / "outputs" / "arterial_endothelial_regeneration"
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in (OUT, FIG, TAB, ROOT / "work" / "mplconfig"):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "figure.dpi": 120,
})
sns.set_theme(style="white", context="paper", font_scale=1.0)
COLORS = {"Control": "#4C78A8", "Early injury": "#F2A541", "Regeneration 48h": "#D1495B"}


def save(fig, name, w=None, h=None):
    if w and h:
        fig.set_size_inches(w, h)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def read_de(sheet: str) -> pd.DataFrame:
    x = pd.read_excel(DATA, sheet_name=sheet)
    x.columns = ["gene", "description", "baseMean", "log2FC", "lfcSE", "stat", "pvalue", "padj"]
    x["padj"] = pd.to_numeric(x["padj"], errors="coerce")
    x["log2FC"] = pd.to_numeric(x["log2FC"], errors="coerce")
    x["stat"] = pd.to_numeric(x["stat"], errors="coerce")
    return x


contrast_sheets = {
    "Early injury vs Control": "Inj00hr_vs_Ctl48hr",
    "Regeneration 48h vs Control": "Inj48hr_vs_Ctl48hr",
    "Regeneration 48h vs Early injury": "Inj48hr_vs_Inj00hr",
}
de = {name: read_de(sheet) for name, sheet in contrast_sheets.items()}
for name, x in de.items():
    x.to_csv(TAB / f"DE_{name.replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")

# Expression matrix and sample metadata.
rlog = pd.read_excel(DATA, sheet_name="regularizedLogCounts")
rlog = rlog[rlog["geneName"].notna()].copy()
rlog["geneName"] = rlog["geneName"].astype(str)
rlog = rlog.drop_duplicates("geneName").set_index("geneName")
expr = rlog.drop(columns=["geneDesc"])
meta = []
for s in expr.columns:
    if s.startswith("Ctl48hr"):
        group = "Control"
    elif s.startswith("Inj00hr"):
        group = "Early injury"
    elif s.startswith("Inj48hr"):
        group = "Regeneration 48h"
    else:
        group = "Flushed aorta"
    batch = s.split("_")[-1][0:2] if "_" in s else "NA"
    meta.append((s, group, batch))
meta = pd.DataFrame(meta, columns=["sample", "group", "batch"]).set_index("sample")
analysis_samples = meta.index[meta.group != "Flushed aorta"]
expr_main = expr[analysis_samples]
meta_main = meta.loc[analysis_samples]
meta_main.to_csv(TAB / "sample_metadata.csv", encoding="utf-8-sig")

# Figure 1A: PCA using top 3,000 variable genes.
v = expr_main.var(axis=1).sort_values(ascending=False).head(3000).index
X = expr_main.loc[v].T
pca = PCA(n_components=2, random_state=1)
coords = pca.fit_transform(X.to_numpy(dtype=float))
pca_df = pd.DataFrame(coords, index=X.index, columns=["PC1", "PC2"]).join(meta_main)
fig, ax = plt.subplots()
for g, dd in pca_df.groupby("group", sort=False):
    ax.scatter(dd.PC1, dd.PC2, s=52, color=COLORS[g], label=f"{g} (n={len(dd)})", edgecolor="white", linewidth=0.6)
ax.axhline(0, color="#DDDDDD", lw=0.5); ax.axvline(0, color="#DDDDDD", lw=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(frameon=False, loc="best")
ax.set_title("GSE115618 arterial endothelial regeneration")
sns.despine(ax=ax)
save(fig, "Fig1A_PCA", 5.2, 4.3)
pca_df.to_csv(TAB / "PCA_coordinates.csv", encoding="utf-8-sig")

# Figure 1B: sample correlation.
corr = expr_main.corr(method="pearson")
group_lut = pd.Series(COLORS)
col_colors = meta_main.group.map(group_lut)
cg = sns.clustermap(corr, cmap="vlag", center=0.9, col_colors=col_colors,
                    xticklabels=False, yticklabels=False, figsize=(7.2, 6.5),
                    cbar_kws={"label": "Pearson r"})
cg.fig.savefig(FIG / "Fig1B_sample_correlation.pdf", bbox_inches="tight")
cg.fig.savefig(FIG / "Fig1B_sample_correlation.png", bbox_inches="tight", dpi=600)
plt.close(cg.fig)

# Figure 2: volcano plots.
labels = ["Ptger4", "Atf3", "Gja1", "Hbegf", "Rap1b", "Sphk1", "Cxcl12", "Dll4", "Pcolce2"]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
summary_rows = []
for ax, (name, x) in zip(axes, de.items()):
    q = x.copy()
    q["minuslog10"] = -np.log10(q.padj.clip(lower=1e-300))
    q["class"] = "NS"
    q.loc[(q.padj < 0.05) & (q.log2FC >= 1), "class"] = "Up"
    q.loc[(q.padj < 0.05) & (q.log2FC <= -1), "class"] = "Down"
    palette = {"NS": "#C7C7C7", "Down": "#3B82B4", "Up": "#D64B4B"}
    for cls in ["NS", "Down", "Up"]:
        z = q[q["class"] == cls]
        ax.scatter(z.log2FC, z.minuslog10, s=5 if cls == "NS" else 7, c=palette[cls], alpha=0.55 if cls == "NS" else 0.75, linewidths=0)
    ax.axvline(-1, ls="--", lw=0.7, c="#555555"); ax.axvline(1, ls="--", lw=0.7, c="#555555")
    ax.axhline(-np.log10(0.05), ls="--", lw=0.7, c="#555555")
    texts = []
    for _, row in q[q.gene.isin(labels)].iterrows():
        texts.append(ax.text(row.log2FC, row.minuslog10, row.gene, fontsize=7))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#777777", lw=0.4), force_text=0.5)
    nup = int((q["class"] == "Up").sum()); ndown = int((q["class"] == "Down").sum())
    summary_rows.append((name, nup, ndown))
    ax.set_title(f"{name}\nUp {nup} | Down {ndown}", fontsize=9)
    ax.set_xlabel("log2 fold change"); ax.set_ylabel("−log10 adjusted P")
    sns.despine(ax=ax)
save(fig, "Fig2_volcano_three_contrasts", 13.2, 4.1)
pd.DataFrame(summary_rows, columns=["contrast", "up", "down"]).to_csv(TAB / "DEG_counts.csv", index=False, encoding="utf-8-sig")

# Sustained and stage-specific response sets.
e = de["Early injury vs Control"].set_index("gene")
r = de["Regeneration 48h vs Control"].set_index("gene")
t = de["Regeneration 48h vs Early injury"].set_index("gene")
joint = e[["log2FC", "padj", "stat"]].rename(columns=lambda c: f"early_{c}").join(
    r[["log2FC", "padj", "stat"]].rename(columns=lambda c: f"regen_{c}"), how="inner").join(
    t[["log2FC", "padj", "stat"]].rename(columns=lambda c: f"transition_{c}"), how="left")
joint["same_direction"] = np.sign(joint.early_log2FC) == np.sign(joint.regen_log2FC)
joint["sustained_sig"] = (joint.early_padj < 0.05) & (joint.regen_padj < 0.05) & joint.same_direction
joint["effect_min"] = np.minimum(np.abs(joint.early_log2FC), np.abs(joint.regen_log2FC))
joint["evidence"] = -np.log10(np.maximum(joint.early_padj, joint.regen_padj).clip(lower=1e-300))
joint["core_score"] = joint.effect_min * np.sqrt(joint.evidence)
joint.sort_values("core_score", ascending=False).to_csv(TAB / "candidate_ranking_all_genes.csv", encoding="utf-8-sig")

# Figure 3A: overlap of significant DE genes (simple two-set Venn-like counts).
sig_e = set(e.index[(e.padj < 0.05) & (e.log2FC.abs() >= 1)])
sig_r = set(r.index[(r.padj < 0.05) & (r.log2FC.abs() >= 1)])
only_e, both, only_r = len(sig_e - sig_r), len(sig_e & sig_r), len(sig_r - sig_e)
fig, ax = plt.subplots()
from matplotlib.patches import Circle
ax.add_patch(Circle((0.43, 0.5), 0.29, color=COLORS["Early injury"], alpha=0.55))
ax.add_patch(Circle((0.57, 0.5), 0.29, color=COLORS["Regeneration 48h"], alpha=0.55))
ax.text(0.30, 0.5, str(only_e), ha="center", va="center", fontsize=16)
ax.text(0.50, 0.5, str(both), ha="center", va="center", fontsize=16, fontweight="bold")
ax.text(0.70, 0.5, str(only_r), ha="center", va="center", fontsize=16)
ax.text(0.27, 0.16, "Early injury", ha="center"); ax.text(0.73, 0.16, "Regeneration 48h", ha="center")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); ax.set_title("DEG overlap (FDR < 0.05, |log2FC| ≥ 1)")
save(fig, "Fig3A_DEG_overlap", 5.2, 4.1)

# Figure 3B: trajectory heatmap of top sustained and transition genes.
top_sustained = joint[joint.sustained_sig].sort_values("core_score", ascending=False).head(35).index.tolist()
top_transition = joint[(joint.transition_padj < 0.05) & (joint.transition_log2FC.abs() >= 1)].sort_values("transition_padj").head(25).index.tolist()
focus_genes = list(dict.fromkeys(top_sustained + top_transition + labels))
focus_genes = [g for g in focus_genes if g in expr_main.index]
zmat = expr_main.loc[focus_genes]
zmat = zmat.sub(zmat.mean(axis=1), axis=0).div(zmat.std(axis=1).replace(0, np.nan), axis=0)
order = meta_main.sort_values("group", key=lambda s: s.map({"Control":0,"Early injury":1,"Regeneration 48h":2})).index
cg = sns.clustermap(zmat[order], cmap="vlag", center=0, col_cluster=False, row_cluster=True,
                    col_colors=meta_main.loc[order, "group"].map(group_lut), xticklabels=False,
                    yticklabels=True, figsize=(8.0, 11.5), vmin=-2.5, vmax=2.5,
                    cbar_kws={"label":"row Z-score"})
cg.ax_heatmap.tick_params(axis="y", labelsize=5)
cg.fig.savefig(FIG / "Fig3B_regeneration_trajectory_heatmap.pdf", bbox_inches="tight")
cg.fig.savefig(FIG / "Fig3B_regeneration_trajectory_heatmap.png", bbox_inches="tight", dpi=600)
plt.close(cg.fig)

# Figure 4: expression trajectories for mechanistic candidates and controls.
panel_genes = [g for g in ["Ptger4", "Rap1b", "Hbegf", "Atf3", "Gja1", "Sphk1", "Cxcl12", "Dll4", "Pcolce2"] if g in expr_main.index]
long = expr_main.loc[panel_genes].T.join(meta_main).reset_index(names="sample").melt(
    id_vars=["sample", "group", "batch"], var_name="gene", value_name="rlog")
order_groups = ["Control", "Early injury", "Regeneration 48h"]
fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.1))
for ax, gene in zip(axes.flat, panel_genes):
    dd = long[long.gene == gene]
    sns.stripplot(data=dd, x="group", y="rlog", order=order_groups, hue="group", palette=COLORS,
                  size=4.3, jitter=0.14, alpha=0.75, ax=ax, legend=False)
    sns.pointplot(data=dd, x="group", y="rlog", order=order_groups, color="#222222", errorbar="se",
                  markers="D", markersize=4, linewidth=1, ax=ax)
    ax.set_title(gene, fontstyle="italic", fontsize=10); ax.set_xlabel(""); ax.set_ylabel("rlog expression")
    ax.tick_params(axis="x", rotation=28, labelsize=7)
    sns.despine(ax=ax)
save(fig, "Fig4_candidate_expression_trajectories", 10.2, 9.1)
long.to_csv(TAB / "candidate_expression_long.csv", index=False, encoding="utf-8-sig")

# Figure 5: candidate-gene correlations across the injury trajectory.
cand = [g for g in ["Ptger4", "Rap1b", "Hbegf", "Egfr", "Mapk1", "Mapk3", "Atf3", "Gja1", "Sphk1", "Cxcl12", "Dll4", "Kdr", "Pecam1", "Pcolce2"] if g in expr_main.index]
cand_corr = expr_main.loc[cand].T.corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.0, 6.1))
sns.heatmap(cand_corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, annot=True, fmt=".2f",
            annot_kws={"size":6}, cbar_kws={"label":"Spearman ρ"}, ax=ax)
ax.set_title("Candidate-axis co-expression across injury and regeneration")
save(fig, "Fig5_candidate_correlation", 7.0, 6.1)
cand_corr.to_csv(TAB / "candidate_spearman_correlations.csv", encoding="utf-8-sig")

# GO/KEGG over-representation and preranked GSEA.
enrich_summary = []
gsea_summary = []
for name, x in de.items():
    safe = name.replace(" ", "_")
    for direction, mask in {
        "Up": (x.padj < 0.05) & (x.log2FC >= 1),
        "Down": (x.padj < 0.05) & (x.log2FC <= -1),
    }.items():
        genes = x.loc[mask, "gene"].dropna().astype(str).tolist()
        if len(genes) < 10:
            continue
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=["GO_Biological_Process_2025", "KEGG_2019_Mouse"],
                             organism="mouse", outdir=None, cutoff=1.0, verbose=False).results
            enr["contrast"] = name; enr["direction"] = direction
            enr.to_csv(TAB / f"ORA_{safe}_{direction}.csv", index=False, encoding="utf-8-sig")
            enrich_summary.append(enr)
        except Exception as exc:
            (TAB / f"ORA_{safe}_{direction}_ERROR.txt").write_text(str(exc), encoding="utf-8")
    ranks = x[["gene", "stat", "log2FC"]].dropna().drop_duplicates("gene")
    ranks["gene"] = ranks["gene"].astype(str)
    # Genes with exactly zero Wald statistic carry no ranking information.
    ranks = ranks[ranks["stat"] != 0].copy()
    # Deterministic tiny tie-breaker leaves statistics unchanged at report precision.
    ranks = ranks.sort_values("gene")
    ranks["stat"] = ranks["stat"] + np.arange(len(ranks)) * 1e-12
    ranks = ranks[["gene", "stat"]].sort_values("stat", ascending=False)
    for lib in ["GO_Biological_Process_2025", "KEGG_2019_Mouse"]:
        try:
            gene_sets = gp.get_library(name=lib, organism="mouse")
            pre = gp.prerank(rnk=ranks, gene_sets=gene_sets, min_size=10, max_size=500,
                             permutation_num=1000, threads=4, seed=20260827, outdir=None, verbose=False)
            rr = pre.res2d.copy(); rr["contrast"] = name; rr["library"] = lib
            rr.to_csv(TAB / f"GSEA_{safe}_{lib}.csv", index=False, encoding="utf-8-sig")
            gsea_summary.append(rr)
        except Exception as exc:
            (TAB / f"GSEA_{safe}_{lib}_ERROR.txt").write_text(str(exc), encoding="utf-8")

if enrich_summary:
    enr_all = pd.concat(enrich_summary, ignore_index=True)
    enr_all.to_csv(TAB / "ORA_all_results.csv", index=False, encoding="utf-8-sig")
    top = enr_all[(enr_all["Adjusted P-value"] < 0.05)].copy()
    top["minuslog10"] = -np.log10(top["Adjusted P-value"].clip(lower=1e-300))
    top["label"] = top["Term"].str.replace(r"\s*\(GO:\d+\)$", "", regex=True).str.slice(0, 62)
    chosen = []
    for keys, dd in top.groupby(["contrast", "direction", "Gene_set"], sort=False):
        chosen.append(dd.sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False]).head(5))
    plot_enr = pd.concat(chosen, ignore_index=True) if chosen else top.head(0)
    if len(plot_enr):
        plot_enr["facet"] = plot_enr["contrast"] + " | " + plot_enr["direction"] + " | " + plot_enr["Gene_set"].str.replace("_", " ")
        facets = plot_enr.facet.unique()
        fig, axes = plt.subplots(len(facets), 1, figsize=(9.2, max(5, 2.0*len(facets))), squeeze=False)
        for ax, fac in zip(axes.flat, facets):
            dd = plot_enr[plot_enr.facet == fac].sort_values("minuslog10")
            ax.barh(dd.label, dd.minuslog10, color="#5B8E7D")
            ax.set_title(fac, fontsize=8, loc="left"); ax.set_xlabel("−log10 FDR"); ax.tick_params(axis="y", labelsize=6)
            sns.despine(ax=ax)
        fig.tight_layout()
        save(fig, "Fig6_GO_KEGG_ORA", 9.2, max(5, 2.0*len(facets)))

if gsea_summary:
    gs_all = pd.concat(gsea_summary, ignore_index=True)
    gs_all.to_csv(TAB / "GSEA_all_results.csv", index=False, encoding="utf-8-sig")
    # Accept standard gseapy column spellings.
    fdr_col = next(c for c in gs_all.columns if c.lower().replace(" ", "") in {"fdrq-val", "fdrqval"})
    nes_col = next(c for c in gs_all.columns if c.upper() == "NES")
    term_col = "Term"
    sig = gs_all[pd.to_numeric(gs_all[fdr_col], errors="coerce") < 0.05].copy()
    sig[nes_col] = pd.to_numeric(sig[nes_col], errors="coerce")
    selected = []
    for keys, dd in sig.groupby(["contrast", "library"], sort=False):
        selected.append(pd.concat([dd.nlargest(5, nes_col), dd.nsmallest(5, nes_col)]).drop_duplicates(term_col))
    plot_gs = pd.concat(selected, ignore_index=True) if selected else sig.head(0)
    if len(plot_gs):
        plot_gs["label"] = plot_gs[term_col].astype(str).str.replace("_", " ").str.slice(0, 58)
        plot_gs["facet"] = plot_gs["contrast"] + " | " + plot_gs["library"].str.replace("_", " ")
        facets = plot_gs.facet.unique()
        fig, axes = plt.subplots(len(facets), 1, figsize=(9.4, max(6, 2.7*len(facets))), squeeze=False)
        for ax, fac in zip(axes.flat, facets):
            dd = plot_gs[plot_gs.facet == fac].sort_values(nes_col)
            ax.barh(dd.label, dd[nes_col], color=np.where(dd[nes_col] > 0, "#D1495B", "#4C78A8"))
            ax.axvline(0, color="#333333", lw=0.6); ax.set_title(fac, fontsize=8, loc="left"); ax.set_xlabel("Normalized enrichment score")
            ax.tick_params(axis="y", labelsize=6); sns.despine(ax=ax)
        fig.tight_layout()
        save(fig, "Fig7_GSEA_summary", 9.4, max(6, 2.7*len(facets)))

# Compact numerical audit used in the written report.
audit = {
    "samples_control": int((meta_main.group == "Control").sum()),
    "samples_early": int((meta_main.group == "Early injury").sum()),
    "samples_regeneration": int((meta_main.group == "Regeneration 48h").sum()),
    "genes_tested": int(len(joint)),
    "early_deg": int(len(sig_e)),
    "regeneration_deg": int(len(sig_r)),
    "overlap_deg": int(len(sig_e & sig_r)),
    "sustained_same_direction_fdr": int(joint.sustained_sig.sum()),
}
pd.Series(audit).to_csv(TAB / "analysis_audit.csv", header=["value"], encoding="utf-8-sig")
print(audit)
print("Top sustained candidates:")
print(joint[joint.sustained_sig].sort_values("core_score", ascending=False).head(30)[
    ["early_log2FC", "early_padj", "regen_log2FC", "regen_padj", "transition_log2FC", "transition_padj", "core_score"]])

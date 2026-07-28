"""
08_visualize_extras.py
======================
Compare extras experiments vs. the main results.

Reads:
  results/training_emissions.csv       - main results (fixed test set)
  results/extras_training_emissions.csv - extras experiments

Produces:
  results/11_extras_pareto.svg         - Leverage Pareto sweep (-10..-30%)
  results/12_extras_cluster_compare.svg - Cluster vs Cluster-Outlier comparison
  results/13_extras_temporal_coverage.svg - Global-temp vs Global-temp+MinK

Usage
-----
    python src/08_visualize_extras.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import apply_style, configure_stdout
from utils.results_io import load_training_results

configure_stdout()
apply_style()

ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

main,   ENERGY_UNIT = load_training_results(RESULTS_DIR / "training_emissions.csv")
extras, _           = load_training_results(RESULTS_DIR / "extras_training_emissions.csv")

MODELS = ["BPR", "MultiDAE", "EASE"]
MODEL_COLORS = {"BPR": "#4C72B0", "MultiDAE": "#DD8452", "EASE": "#55A868"}

def short(s):
    return (s.replace("CS-Leverage", "CS-Lev")
             .replace("CS-Cluster-Outlier", "CS-Clust-Out")
             .replace("CS-Cluster", "CS-Clust")
             .replace("Global-temp+MinK3", "G-temp+MinK")
             .replace("Global-temporal", "G-temporal")
             .replace(" [ref]", "")
             .replace("Coreset-Leverage", "CS-Lev")
             .replace("Coreset-Cluster",  "CS-Clust"))

# ── Plot 1: Leverage Pareto sweep - Recall@10 vs fraction dropped ────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey='row')
fig.suptitle(f"Leverage Coreset - Pareto Sweep: Recall@10 & Energy ({ENERGY_UNIT}) vs. data dropped",
             fontsize=13, fontweight="bold")

leverage_strats_main   = [s for s in main["Strategy"].unique()   if "Leverage" in s or "Coreset-Leverage" in s]
leverage_strats_extras = [s for s in extras["Strategy"].unique() if "CS-Leverage" in s and "[ref]" not in s]

fraction_map = {
    "Coreset-Leverage (-20%)":  20,
    "Coreset-Leverage (-30%)":  30,
    "CS-Leverage (-10%)":       10,
    "CS-Leverage (-15%)":       15,
    "CS-Leverage (-20%) [ref]": 20,
}
baseline_recall = main[main.Strategy == "Baseline"].set_index("Model")["Recall10"]
baseline_energy = main[main.Strategy == "Baseline"].set_index("Model")["Total_Energy"]

for col_idx, model in enumerate(MODELS):
    ax_rec    = axes[0, col_idx]
    ax_energy = axes[1, col_idx]

    x_vals, y_rec, y_energy = [], [], []

    for strat in leverage_strats_main:
        row = main[(main.Strategy == strat) & (main.Model == model)]
        if len(row):
            frac = fraction_map.get(strat)
            if frac:
                x_vals.append(frac)
                y_rec.append(row["Recall10"].values[0])
                y_energy.append(row["Total_Energy"].values[0])

    for strat in leverage_strats_extras:
        row = extras[(extras.Strategy == strat) & (extras.Model == model)]
        if len(row):
            frac = fraction_map.get(strat)
            if frac and frac not in x_vals:
                x_vals.append(frac)
                y_rec.append(row["Recall10"].values[0])
                y_energy.append(row["Total_Energy"].values[0])

    b_rec    = baseline_recall.get(model, np.nan)
    b_energy = baseline_energy.get(model, np.nan)

    combined = sorted(zip(x_vals, y_rec, y_energy))
    xs, ys_r, ys_e = zip(*combined) if combined else ([], [], [])

    ax_rec.plot([0] + list(xs), [b_rec] + list(ys_r),
            color=MODEL_COLORS[model], marker="o", linewidth=2, markersize=8)
    ax_rec.axhline(b_rec, color="gray", linewidth=1, linestyle=":", label="Baseline Recall")
    ax_rec.fill_between([0] + list(xs), b_rec, [b_rec] + list(ys_r),
                    alpha=0.12, color="green" if all(y >= b_rec for y in ys_r) else "orange")

    ax_rec.set_ylabel("Recall@10", fontsize=11)
    ax_rec.set_title(model, fontsize=12, fontweight="bold")
    ax_rec.set_xlim(-1, 33)
    ax_rec.legend(fontsize=9)

    ax_energy.plot([0] + list(xs), [b_energy] + list(ys_e),
            color="#E0527A", marker="s", linewidth=2, markersize=8)
    ax_energy.axhline(b_energy, color="gray", linewidth=1, linestyle=":", label="Baseline Energy")
    ax_energy.fill_between([0] + list(xs), b_energy, [b_energy] + list(ys_e),
                    alpha=0.12, color="green" if all(y <= b_energy for y in ys_e) else "red")

    ax_energy.set_xlabel("Data dropped (%)", fontsize=11)
    ax_energy.set_ylabel(f"Total Train Energy ({ENERGY_UNIT})", fontsize=11)
    ax_energy.set_xlim(-1, 33)
    ax_energy.legend(fontsize=9)

plt.tight_layout()
out = RESULTS_DIR / "11_extras_pareto.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


# ── Plot 2: Cluster vs Cluster-Outlier - grouped bar comparison ──────────────
fracs   = ["-20%", "-30%"]
orig_keys    = {"-20%": "Coreset-Cluster (-20%)",     "-30%": "Coreset-Cluster (-30%)"}
outlier_keys = {"-20%": "CS-Cluster-Outlier (-20%)", "-30%": "CS-Cluster-Outlier (-30%)"}

n_m   = len(MODELS)
bar_w = 0.18
x     = np.arange(len(fracs))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Cluster-Centroid vs. Cluster-Outlier: Recall@10, Coverage, & Energy ({ENERGY_UNIT})",
             fontsize=13, fontweight="bold")

metrics = [("Recall10", "Recall@10"), ("Coverage", "Coverage"), ("Total_Energy", f"Total Train Energy ({ENERGY_UNIT})")]

for ax_idx, (metric, ylabel) in enumerate(metrics):
    ax = axes[ax_idx]
    for mi, model in enumerate(MODELS):
        orig_vals    = []
        outlier_vals = []
        for frac in fracs:
            ok = orig_keys[frac]
            r_orig = main[(main.Strategy == ok) & (main.Model == model)]
            orig_vals.append(r_orig[metric].values[0] if len(r_orig) else np.nan)

            ok2 = outlier_keys[frac]
            r_out = extras[(extras.Strategy == ok2) & (extras.Model == model)]
            outlier_vals.append(r_out[metric].values[0] if len(r_out) else np.nan)

        offset_orig    = (mi - n_m/2 + 0.5) * bar_w - bar_w * 0.35
        offset_outlier = (mi - n_m/2 + 0.5) * bar_w + bar_w * 0.35

        ax.bar(x + offset_orig,    orig_vals,    bar_w * 0.65,
               color=MODEL_COLORS[model], alpha=0.55, edgecolor="white",
               label=f"{model} (centroid)" if ax_idx == 0 else "_")
        ax.bar(x + offset_outlier, outlier_vals, bar_w * 0.65,
               color=MODEL_COLORS[model], alpha=0.95, edgecolor="white",
               label=f"{model} (outlier)" if ax_idx == 0 else "_")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {f}" for f in fracs], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel}", fontsize=11)

handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, fontsize=8, ncol=1,
           loc="center right", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(rect=[0, 0, 0.88, 1])
out = RESULTS_DIR / "12_extras_cluster_compare.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


# ── Plot 3: Global-temporal vs Global-temporal+MinK - Recall & Coverage ──────
orig_strat  = "Global-temporal (Sep 2000)"
mink_strat  = "Global-temp+MinK3 (Sep 2000)"

orig_rows = main[main.Strategy == orig_strat]
mink_rows = extras[extras.Strategy == mink_strat]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Global-temporal (Sep 2000): Pure vs. + Min-K=3 Guarantee",
             fontsize=13, fontweight="bold")

bar_w = 0.3
x     = np.arange(n_m)
labels = MODELS

metrics = [("Recall10", "Recall@10"), ("Coverage", "Coverage"), ("Total_Energy", f"Total Train Energy ({ENERGY_UNIT})")]

for ax_idx, (metric, ylabel) in enumerate(metrics):
    ax = axes[ax_idx]
    orig_vals = [orig_rows[orig_rows.Model == m][metric].values[0]
                 if len(orig_rows[orig_rows.Model == m]) else np.nan
                 for m in MODELS]
    mink_vals = [mink_rows[mink_rows.Model == m][metric].values[0]
                 if len(mink_rows[mink_rows.Model == m]) else np.nan
                 for m in MODELS]

    bars1 = ax.bar(x - bar_w/2, orig_vals, bar_w,
                   color="#6C63FF", alpha=0.65, edgecolor="white", label="Pure Global-temporal" if ax_idx == 0 else "_")
    bars2 = ax.bar(x + bar_w/2, mink_vals, bar_w,
                   color="#43BF95", alpha=0.9,  edgecolor="white", label="+ Min-K=3 guarantee" if ax_idx == 0 else "_")

    for bar, v in zip(list(bars1) + list(bars2), orig_vals + mink_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (v*0.01),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12)
    if ax_idx == 0:
        ax.legend(fontsize=9)
    if metric == "Coverage":
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

plt.tight_layout()
out = RESULTS_DIR / "13_extras_temporal_coverage.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")

# ── Summary table ─────────────────────────────────────────────────────────────
_recall_std_col = "Recall@10_std" if "Recall@10_std" in extras.columns else None
_pval_col       = "Recall@10_wilcoxon_p_vs_baseline" if "Recall@10_wilcoxon_p_vs_baseline" in extras.columns else None

print("\n── Extras summary ───────────────────────────────────────────────────────")
for model in MODELS:
    b = main[(main.Model == model) & (main.Strategy == "Baseline")]["Recall10"].values
    b = b[0] if len(b) else np.nan
    print(f"\n  {model}  (Baseline Recall@10 = {b:.4f})")
    for strat, _, _ in [
        ("CS-Leverage (-10%)",           extras, "extras"),
        ("CS-Leverage (-15%)",           extras, "extras"),
        ("CS-Cluster-Outlier (-20%)",    extras, "extras"),
        ("CS-Cluster-Outlier (-30%)",    extras, "extras"),
        ("Global-temp+MinK3 (Sep 2000)", extras, "extras"),
    ]:
        df_ = extras
        row = df_[(df_.Strategy == strat) & (df_.Model == model)]
        if len(row):
            r   = row["Recall10"].values[0]
            cov = row["Coverage"].values[0]
            std_txt = f"±{row[_recall_std_col].values[0]:.4f}" if _recall_std_col else ""
            p = row[_pval_col].values[0] if _pval_col else float("nan")
            sig_txt = f"  [p={p:.3f}]" if pd.notna(p) else ""
            print(f"    {strat:<38}  Recall={r:.4f}{std_txt} ({(r-b)/b*100:+.1f}%)  Coverage={cov:.3f}{sig_txt}")

print("\n[DONE] All extra plots saved to results/")

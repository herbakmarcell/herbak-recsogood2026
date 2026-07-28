"""
src/07_visualize_results.py
===========================
Visualise training results: energy consumption, Recall@10, and the green tradeoff.

Produces:
  results/05_co2_vs_recall.svg        – scatter: total energy vs Recall@10 per job
  results/06_recall_by_strategy.svg   – grouped bar: Recall@10 per strategy & model
  results/07_co2_by_strategy.svg      – grouped bar: total energy per strategy & model
  results/07b_coverage.svg            – grouped bar: catalog coverage per strategy
  results/08_green_tradeoff.svg       – relative energy saved vs relative recall lost (vs Baseline)
  results/09_efficiency.svg           – Recall@10 per kWh total energy (efficiency frontier)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import (apply_style, model_color, configure_stdout,
                              short_label as short, strategy_markers,
                              set_log_ticks, robust_limits, spread_labels)
from utils.results_io import load_training_results

configure_stdout()
apply_style()

ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CSV_PATH    = RESULTS_DIR / "training_emissions.csv"

# Total energy = preprocessing + training (per job); load_training_results
# handles both current kWh columns and legacy CO2-gram CSVs.
df, ENERGY_UNIT = load_training_results(CSV_PATH)

MODELS     = df["Model"].unique().tolist()
STRATEGIES = df["Strategy"].unique().tolist()

_extra = 0
MODEL_COLORS = {}
for m in MODELS:
    MODEL_COLORS[m] = model_color(m, _extra)
    if m not in {"BPR", "MultiDAE", "EASE", "ELSA"}:
        _extra += 1

STRAT_MARKERS = strategy_markers(STRATEGIES)

n_s   = len(STRATEGIES)
n_m   = len(MODELS)
bar_w = max(0.10, 0.72 / n_m)
x     = np.arange(n_s)
fig_w = max(14, n_s * 1.6)

# ── 1. Total energy vs Recall@10 scatter ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for _, row in df.iterrows():
    ax.scatter(
        row["Total_Energy"], row["Recall10"],
        color=MODEL_COLORS[row["Model"]],
        marker=STRAT_MARKERS[row["Strategy"]],
        s=120, zorder=3, edgecolors="white", linewidths=0.6,
    )

sorted_df   = df.sort_values("Total_Energy")
best_recall = -np.inf
pareto = []
for _, row in sorted_df.iterrows():
    if row["Recall10"] > best_recall:
        best_recall = row["Recall10"]
        pareto.append(row)
pareto_df = pd.DataFrame(pareto)
ax.plot(pareto_df["Total_Energy"], pareto_df["Recall10"],
        color="#C9A227", linewidth=1.8, linestyle="--", zorder=2,
        label="Efficiency frontier")

ax.set_xscale("log")
set_log_ticks(ax, "x")
ax.set_xlabel(f"Total energy  (preprocessing + training,  {ENERGY_UNIT},  log scale)", fontsize=11)
ax.set_ylabel("Recall@10  (100 sampled negatives, global fixed test)", fontsize=11)
ax.set_title("Recommendation quality vs. total energy cost", fontsize=13, fontweight="bold")

# Both legends parked below the point cloud, where models bunch near the top
# of the y-range, so neither overlaps the markers.
model_patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
strat_lines   = [Line2D([0],[0], marker=STRAT_MARKERS[s], color="grey",
                         linestyle="None", markersize=8, label=short(s).replace("\n",""))
                 for s in STRATEGIES]
leg1 = ax.legend(handles=model_patches, title="Model",    loc="lower right",
                 fontsize=8, framealpha=0.9)
leg2 = ax.legend(handles=strat_lines,   title="Strategy", loc="lower left",
                 fontsize=7, ncol=2, framealpha=0.9)
ax.add_artist(leg1)

plt.tight_layout()
out = RESULTS_DIR / "05_energy_vs_recall.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


# ── 2. Recall@10 by strategy (grouped bar, * = significant vs Baseline, ─────
#      paired Wilcoxon signed-rank test, see utils/eval_utils.py) ────────────
_recall_std_col = "Recall@10_std" if "Recall@10_std" in df.columns else None
_pval_col       = "Recall@10_wilcoxon_p_vs_baseline" if "Recall@10_wilcoxon_p_vs_baseline" in df.columns else None

fig, ax = plt.subplots(figsize=(fig_w, 5))

for i, model in enumerate(MODELS):
    sub  = df[df.Model == model]
    vals = [sub[sub.Strategy == s]["Recall10"].values[0] for s in STRATEGIES]
    errs = ([sub[sub.Strategy == s][_recall_std_col].values[0] for s in STRATEGIES]
            if _recall_std_col else None)
    offset = (i - n_m/2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, bar_w, yerr=errs, capsize=2,
                  label=model, color=MODEL_COLORS[model], alpha=0.9, edgecolor="white")

    if _pval_col:
        for xi, s, bar, v, e in zip(x, STRATEGIES, bars, vals,
                                     errs if errs else [0]*len(STRATEGIES)):
            row = sub[sub.Strategy == s]
            p = row[_pval_col].values[0] if len(row) else float("nan")
            if pd.notna(p) and p < 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, v + e + 0.006,
                        "*", ha="center", va="bottom", fontsize=13,
                        color=MODEL_COLORS[model], fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([short(s) for s in STRATEGIES], fontsize=8)
ax.set_ylabel("Recall@10  (global fixed test set)", fontsize=11)
ax.set_title("Recommendation quality per reduction strategy", fontsize=13, fontweight="bold")
ax.legend(title="Model", fontsize=9)
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
if _pval_col:
    ax.text(0.01, 0.98, "* p < 0.05 vs Baseline (paired Wilcoxon, same model)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#5a5a6e")

baseline_avg = df[df.Strategy == "Baseline"]["Recall10"].mean()
ax.axhline(baseline_avg, color="gray", linewidth=1.2, linestyle=":", zorder=0)
ax.text(n_s - 0.5, baseline_avg + 0.003, "Baseline avg", ha="right", fontsize=8, color="gray")

coreset_idxs = [i for i, s in enumerate(STRATEGIES) if "Coreset" in s]
for ci in coreset_idxs:
    ax.axvspan(ci - 0.5, ci + 0.5, alpha=0.06, color="purple", zorder=0)
if coreset_idxs:
    ax.text(np.mean(coreset_idxs), ax.get_ylim()[1]*0.99,
            "◀ Coreset strategies ▶", ha="center", va="top", fontsize=8,
            color="purple", alpha=0.7)

plt.tight_layout()
out = RESULTS_DIR / "06_recall_by_strategy.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


# ── 3. Total energy by strategy (grouped bar, log scale) ─────────────────────
fig, ax = plt.subplots(figsize=(fig_w, 5))

for i, model in enumerate(MODELS):
    vals = [df[(df.Model == model) & (df.Strategy == s)]["Total_Energy"].values[0]
            for s in STRATEGIES]
    ax.bar(x + (i - n_m/2 + 0.5)*bar_w, vals, bar_w,
           label=model, color=MODEL_COLORS[model], alpha=0.9, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([short(s) for s in STRATEGIES], fontsize=8)
ax.set_ylabel(f"Total energy  (preprocessing + training,  {ENERGY_UNIT})", fontsize=11)
ax.set_yscale("log")
set_log_ticks(ax, "y")
ax.set_title("Total energy cost per reduction strategy  (preprocessing + training)", fontsize=13, fontweight="bold")
ax.legend(title="Model", fontsize=9)

plt.tight_layout()
out = RESULTS_DIR / "07_energy_by_strategy.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


# ── 3b. Coverage: fraction of the item catalog recommended per strategy ──────
fig, ax = plt.subplots(figsize=(fig_w, 5))

cov_by_strategy = df.groupby("Strategy")["Coverage"].mean().reindex(STRATEGIES)
bar_colors = ["#4C63FF" if "Coreset" not in s else "#E08A2E" for s in STRATEGIES]
bars = ax.bar(x, cov_by_strategy.values * 100, 0.6,
              color=bar_colors, alpha=0.9, edgecolor="white")

for bar, val in zip(bars, cov_by_strategy.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([short(s) for s in STRATEGIES], fontsize=8)
ax.set_ylabel("Catalog coverage  (% of item catalog recommended)", fontsize=11)
ax.set_ylim(0, 108)
ax.set_title("Catalog coverage per strategy\n(fraction of the train-vocabulary items that appear in any user's top-10 list)", fontsize=12, fontweight="bold")

legend_patches = [
    mpatches.Patch(color="#4C63FF", label="Non-Coreset strategies"),
    mpatches.Patch(color="#E08A2E", label="Coreset strategies"),
]
ax.legend(handles=legend_patches, fontsize=9)

plt.tight_layout()
out = RESULTS_DIR / "07b_coverage.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


# ── 4. Green tradeoff: % energy saved vs % Recall change (vs Baseline) ───────
fig, ax = plt.subplots(figsize=(10, 6))

baseline = df[df.Strategy == "Baseline"].set_index("Model")

points = []   # (energy_saved, recall_chg, model, strategy)
for _, row in df[df.Strategy != "Baseline"].iterrows():
    base_energy = baseline.loc[row["Model"], "Total_Energy"]
    base_recall = baseline.loc[row["Model"], "Recall10"]
    energy_saved = (base_energy - row["Total_Energy"]) / base_energy * 100
    recall_chg   = (row["Recall10"] - base_recall) / base_recall * 100
    points.append((energy_saved, recall_chg, row["Model"], row["Strategy"]))

# A few strategies *increase* energy several-fold (energy_saved down to ~-1150%),
# which, drawn to scale, would collapse every other point into a sliver at x=0.
# Frame the axis on the bulk of the points and pin each off-scale point to the
# left edge with its true value annotated instead.
x_lo, x_hi = robust_limits([p[0] for p in points], pad=0.06)
ax.set_xlim(x_lo, x_hi)

offscale = []   # (true_value, y) for points pinned to the left edge
for energy_saved, recall_chg, model, strat in points:
    off      = energy_saved < x_lo
    x_plot   = x_lo if off else energy_saved
    base_edge = "gold" if "Coreset-Leverage" in strat else "white"
    ax.scatter(x_plot, recall_chg,
               color=MODEL_COLORS[model], marker=STRAT_MARKERS[strat],
               s=140, zorder=3, edgecolors=("crimson" if off else base_edge),
               linewidths=1.2)
    if off:
        offscale.append((energy_saved, recall_chg))

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)

if offscale:
    y0, y1  = ax.get_ylim()
    x_text  = x_lo + (x_hi - x_lo) * 0.012
    label_y = spread_labels([p[1] for p in offscale], (y1 - y0) * 0.045)
    for (val, _y), yl in zip(offscale, label_y):
        ax.annotate(f"◀ {val:.0f}%", xy=(x_text, yl), va="center", ha="left",
                    fontsize=6.5, color="crimson", fontweight="bold",
                    annotation_clip=False)

xl, yl = ax.get_xlim(), ax.get_ylim()
ax.fill_between([max(0, xl[0]), xl[1]], 0, max(yl[1], 1),
                alpha=0.07, color="green", zorder=0)
ax.fill_between([max(0, xl[0]), xl[1]], min(yl[0],-1), 0,
                alpha=0.05, color="red",   zorder=0)

ax.text(xl[1]*0.97, max(yl[1],1)*0.92,
        "Energy saved &\nQuality improved ✓",
        ha="right", va="top", fontsize=9, color="green", fontweight="bold")
ax.text(xl[1]*0.97, min(yl[0],-0.5)*0.6,
        "Energy saved,\nsome quality lost",
        ha="right", va="top", fontsize=9, color="firebrick", alpha=0.8)

ax.set_xlabel("Total energy reduction vs Baseline  (%)", fontsize=11)
ax.set_ylabel("Recall@10 change vs Baseline  (%: positive = better)", fontsize=11)
ax.set_title("Green tradeoff: energy saved vs quality change\n(all on same global fixed test set)", fontsize=13, fontweight="bold")

model_patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
strat_lines   = [Line2D([0],[0], marker=STRAT_MARKERS[s], color="grey",
                         linestyle="None", markersize=8,
                         label=short(s).replace("\n",""))
                 for s in STRATEGIES if s != "Baseline"]
leg1 = ax.legend(handles=model_patches, title="Model",    loc="upper left",  fontsize=8)
leg2 = ax.legend(handles=strat_lines,   title="Strategy", loc="center left", fontsize=7)
ax.add_artist(leg1)

plt.tight_layout()
out = RESULTS_DIR / "08_green_tradeoff.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


# ── 5. Efficiency: Recall@10 per kWh total energy ─────────────────────────────
df["Efficiency"] = df["Recall10"] / df["Total_Energy"]

fig, ax = plt.subplots(figsize=(fig_w, 5))

for i, model in enumerate(MODELS):
    vals = [df[(df.Model == model) & (df.Strategy == s)]["Efficiency"].values[0]
            for s in STRATEGIES]
    ax.bar(x + (i - n_m/2 + 0.5)*bar_w, vals, bar_w,
           label=model, color=MODEL_COLORS[model], alpha=0.9, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([short(s) for s in STRATEGIES], fontsize=8)
ax.set_ylabel(f"Recall@10 per {ENERGY_UNIT} total energy  (higher = greener)", fontsize=11)
ax.set_yscale("log")
set_log_ticks(ax, "y")
ax.set_title(f"Energy efficiency: quality delivered per {ENERGY_UNIT} of total energy", fontsize=13, fontweight="bold")
ax.legend(title="Model", fontsize=9)

plt.tight_layout()
out = RESULTS_DIR / "09_efficiency.svg"
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print(f"[OK] {out.name}")


print("\n[DONE] All plots saved to results/")

print("\n── Key findings ─────────────────────────────────────────────────────────")
_has_evaluable  = "Recall@10_evaluable" in df.columns
_has_user_rate  = "Eval_User_Rate" in df.columns
_has_common_sub = "Recall@10_common_subset" in df.columns

for model in MODELS:
    b = df[(df.Model == model) & (df.Strategy == "Baseline")]["Recall10"].values
    if len(b) == 0:
        continue
    b = b[0]
    print(f"\n  {model}  (Baseline Recall@10 = {b:.4f})")
    for s in STRATEGIES:
        if s == "Baseline":
            continue
        row = df[(df.Model == model) & (df.Strategy == s)]
        if len(row) == 0:
            continue
        r = row["Recall10"].values[0]
        std = row[_recall_std_col].values[0] if _recall_std_col else float("nan")
        cov = row["Coverage"].values[0]
        p = row[_pval_col].values[0] if _pval_col else float("nan")
        sig = "  [p={:.3f}]".format(p) if pd.notna(p) else ""
        line = f"    {s:<35}  Recall={r:.4f}"
        if pd.notna(std):
            line += f"±{std:.4f}"
        line += f" ({(r-b)/b*100:+.1f}%)  Coverage={cov:.3f}{sig}"
        print(line)
        if _has_evaluable or _has_user_rate or _has_common_sub:
            extra = []
            if _has_evaluable:
                extra.append(f"Recall_evaluable={row['Recall@10_evaluable'].values[0]:.4f}")
            if _has_user_rate:
                extra.append(f"Eval_User_Rate={row['Eval_User_Rate'].values[0]:.3f}")
            if _has_common_sub:
                extra.append(f"Recall_common_subset={row['Recall@10_common_subset'].values[0]:.4f}")
            print(f"      ({'  '.join(extra)})")

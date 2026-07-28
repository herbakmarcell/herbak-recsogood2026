"""
12_plot_all_methods.py
=======================
Combined overview of every (strategy, model) result from the main and extras
sweeps on a single pair of axes:

  results/16_all_methods_energy_vs_recall.svg  – scatter: Recall@10 vs total
      energy (log scale) across all strategies incl. the extras sweep.
  results/17_all_methods_green_tradeoff.svg    – energy saved vs quality change
      vs Baseline, per (strategy, model).

Distinct from 07_visualize_results.py (main sweep only): this one also folds in
the extras strategies so every method appears together for comparison.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import (apply_style, get_cmap, configure_stdout,
                              short_label as short, strategy_markers,
                              set_log_ticks, robust_limits, spread_labels)
from utils.results_io import load_training_results

configure_stdout()
apply_style()

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "training_emissions.csv"

extras_path = RESULTS_DIR / "extras_training_emissions.csv"
df, ENERGY_UNIT = load_training_results(CSV_PATH, extras_path)

MODELS = df["Model"].unique().tolist()
STRATEGIES = df["Strategy"].unique().tolist()

_MODEL_COLOR_MAP = {
    "BPR":      "#4C72B0",
    "MultiDAE": "#DD8452",
    "EASE":     "#55A868",
}
_tab10 = get_cmap("tab10")
MODEL_COLORS = {m: _MODEL_COLOR_MAP.get(m, _tab10(i)) for i, m in enumerate(MODELS)}

STRAT_MARKERS = strategy_markers(STRATEGIES)

# ── 1. Scatter plot: emissions vs quality ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for _, row in df.iterrows():
    ax.scatter(
        row["Total_Energy"], row["Recall10"],
        color=MODEL_COLORS[row["Model"]],
        marker=STRAT_MARKERS[row["Strategy"]],
        s=120, zorder=3, edgecolors="white", linewidths=0.6,
    )

ax.set_xscale("log")
set_log_ticks(ax, "x")
ax.set_xlabel(f"Total energy  (preprocessing + training,  {ENERGY_UNIT},  log scale)", fontsize=11)
ax.set_ylabel("Recall@10", fontsize=11)
ax.set_title("Overall Comparison: Quality vs. Total Energy Cost (all strategies)", fontsize=13, fontweight="bold")

model_patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
strat_lines   = [Line2D([0],[0], marker=STRAT_MARKERS[s], color="grey", linestyle="None", markersize=8, label=short(s).replace("\n","")) for s in STRATEGIES]
leg1 = ax.legend(handles=model_patches, title="Model", loc="lower right", fontsize=8)
leg2 = ax.legend(handles=strat_lines, title="Strategy", loc="upper left", fontsize=7, ncol=2)
ax.add_artist(leg1)

plt.tight_layout()
out1 = RESULTS_DIR / "16_all_methods_energy_vs_recall.svg"
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)


# ── 2. Green tradeoff plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
# Both the main and extras CSVs carry a Baseline row per model; keep one
# (the main-CSV reference) so each model maps to a single baseline value.
baseline = (df[df.Strategy == "Baseline"]
            .drop_duplicates(subset="Model", keep="first")
            .set_index("Model"))

points = []   # (energy_saved, recall_chg, model, strategy)
for _, row in df[df.Strategy != "Baseline"].iterrows():
    base_energy   = baseline.loc[row["Model"], "Total_Energy"]
    base_recall   = baseline.loc[row["Model"], "Recall10"]
    energy_saved  = (base_energy - row["Total_Energy"]) / base_energy * 100
    recall_chg    = (row["Recall10"] - base_recall) / base_recall * 100
    points.append((energy_saved, recall_chg, row["Model"], row["Strategy"]))

# Frame the x-axis on the bulk of the points; strategies that *increase* energy
# several-fold are pinned to the left edge with their true value annotated so
# they don't squash everything else into x=0.
x_lo, x_hi = robust_limits([p[0] for p in points], pad=0.06)
ax.set_xlim(x_lo, x_hi)

offscale = []   # (true_value, y) for points pinned to the left edge
for energy_saved, recall_chg, model, strat in points:
    off       = energy_saved < x_lo
    x_plot    = x_lo if off else energy_saved
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
ax.fill_between([max(0, xl[0]), xl[1]], 0, max(yl[1], 1), alpha=0.07, color="green", zorder=0)
ax.fill_between([max(0, xl[0]), xl[1]], min(yl[0],-1), 0, alpha=0.05, color="red", zorder=0)

ax.set_xlabel("Total energy reduction vs Baseline  (%)", fontsize=11)
ax.set_ylabel("Recall@10 change vs Baseline  (%: positive = better)", fontsize=11)
ax.set_title("Green Tradeoff: Energy Saved vs Quality Change (all strategies)", fontsize=13, fontweight="bold")

# Model legend in the (empty) lower-left corner, clear of both the pinned
# off-scale labels along the left edge and the point cloud on the right.
leg1 = ax.legend(handles=model_patches, title="Model", loc="lower left", fontsize=8)
strat_lines_no_base = [Line2D([0],[0], marker=STRAT_MARKERS[s], color="grey", linestyle="None", markersize=8, label=short(s).replace("\n","")) for s in STRATEGIES if s != "Baseline"]
leg2 = ax.legend(handles=strat_lines_no_base, title="Strategy", fontsize=7,
                 loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.add_artist(leg1)

plt.tight_layout()
out2 = RESULTS_DIR / "17_all_methods_green_tradeoff.svg"
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)

print("Plots successfully saved!")

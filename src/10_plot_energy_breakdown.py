"""
10_plot_energy_breakdown.py
===========================
Per-strategy energy-composition figures. Merged from the former
20_plot_all_emissions_bar.py and 21_plot_full_pipeline_stacked.py — both broke
the training energy of each reduction strategy into its components, so they now
live together as two functions of one script (output filenames unchanged):

  results/20_all_emissions_comparison_bar.svg  – horizontal bar per strategy:
      reduced-load cost vs reduction-operation cost vs net saving vs baseline
      (all strategies incl. extras).
  results/22_training_pipeline_stacked.svg     – stacked data-processing +
      training energy per strategy, one panel per model.

Usage
-----
    python src/10_plot_energy_breakdown.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import apply_style
from utils.results_io import load_training_results

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def plot_all_emissions_bar():
    apply_style()

    df, _ = load_training_results(RESULTS_DIR / "training_emissions.csv",
                                  RESULTS_DIR / "extras_training_emissions.csv")

    name_map = {
        "User-based (−20%)": "User-based (−20%)",
        "Item-based (−20%)": "Item-based (−20%)",
        "User-temporal (last 300)": "User-temporal (last 300)",
        "Global-temporal (Sep 2000)": "Global-temporal (Sep 2000)",
        "Global-temp+MinK3 (Sep 2000)": "Global-temporal min-k(Sep 2000)",
        "Coreset-Leverage (−20%)": "Coreset-Leverage (−20%)",
        "Coreset-Leverage (−30%)": "Coreset-Leverage (−30%)",
        "Coreset-Cluster (−20%)": "Coreset-Cluster (−20%)",
        "Coreset-Cluster (−30%)": "Coreset-Cluster (−30%)",
        "CS-Cluster-Outlier (−20%)": "Coreset-Cluster-outlier (−20%)",
        "CS-Cluster-Outlier (−30%)": "Coreset-Cluster-outlier (−30%)"
    }

    desired_order = [
        "Coreset-Cluster-outlier (−30%)",
        "Coreset-Cluster-outlier (−20%)",
        "Coreset-Cluster (−30%)",
        "Coreset-Cluster (−20%)",
        "Coreset-Leverage (−30%)",
        "Coreset-Leverage (−20%)",
        "Global-temporal min-k(Sep 2000)",
        "Global-temporal (Sep 2000)",
        "User-temporal (last 300)",
        "Item-based (−20%)",
        "User-based (−20%)"
    ]

    df_strat = df.drop_duplicates(subset=["Strategy"]).copy()

    baseline_row = df_strat[df_strat["Strategy"] == "Baseline"]
    if len(baseline_row) == 0:
        print("Error: Baseline strategy not found.")
        return

    baseline_load = df[df["Strategy"] == "Baseline"]["Energy_kWh_mean"].mean()
    baseline_ratings = baseline_row["Train_size"].values[0]

    df_strat["Plot_Name"] = df_strat["Strategy"].map(name_map)
    df_strat = df_strat[df_strat["Plot_Name"].notna()].copy()

    df_strat["Order"] = df_strat["Plot_Name"].apply(lambda x: desired_order.index(x))
    df_strat = df_strat.sort_values("Order").reset_index(drop=True)

    labels = df_strat["Plot_Name"].tolist()
    red_op = df_strat["Preprocess_kWh_mean"].tolist()
    reduced_load = [baseline_load * (r / baseline_ratings) for r in df_strat["Train_size"].tolist()]

    n = len(labels)
    h = 0.45
    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.7)))

    net_savings = [max(baseline_load - rl - ro, 0) for rl, ro in zip(reduced_load, red_op)]

    ax.barh(labels, reduced_load, height=h, color="#4C63FF", edgecolor="none", label="Reduced load cost")
    ax.barh(labels, red_op, left=reduced_load, height=h, color="#E0527A", edgecolor="none", label="Reduction operation cost")

    left_for_green = [rl + ro for rl, ro in zip(reduced_load, red_op)]
    ax.barh(labels, net_savings, left=left_for_green, height=h, color="#2FA787", edgecolor="none", label="Net energy saved vs baseline")

    def fmt_val(v):
        return f"${v * 1e4:.2f} \\times 10^{{-4}}$"

    ax.axvline(baseline_load, color="#8a8aa0", linewidth=1.3, linestyle="--", alpha=0.7,
               label=f"Baseline load ({fmt_val(baseline_load)} kWh)")

    max_total = max(rl + ro + sv for rl, ro, sv in zip(reduced_load, red_op, net_savings))
    max_x = max(max_total, baseline_load) * 1.25
    pad = max_x * 0.012

    for i, (rl, ro, sv) in enumerate(zip(reduced_load, red_op, net_savings)):
        total = rl + ro + sv
        net = baseline_load - rl - ro

        if net >= 0:
            pct_txt = f"saves {net / baseline_load * 100:.1f}%"
            txt_col = "#1a1a2e"
        else:
            pct_txt = f"overhead +{-net / baseline_load * 100:.1f}%"
            txt_col = "#E0527A"

        ax.text(total + pad, i, pct_txt, va="center", ha="left", color=txt_col, fontsize=8)

    ax.set_xlabel("Energy consumed (kWh)", fontsize=11, labelpad=10)
    ax.set_title("Energy Consumption per Reduction Strategy", fontsize=13, fontweight="bold", color="#1a1a2e", pad=18)

    linthresh = max(baseline_load, 1e-12) * 0.1
    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_xlim(0, max_x)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x * 1e4:g} \\times 10^{{-4}}$" if x != 0 else "0"))
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    legend_elements = [
        Patch(facecolor="#4C63FF", label="Reduced load cost"),
        Patch(facecolor="#E0527A", label="Reduction operation cost"),
        Patch(facecolor="#2FA787", label="Net energy saved vs baseline"),
        plt.Line2D([0], [0], color="#8a8aa0", linewidth=1.3, linestyle="--", alpha=0.8, label="Baseline load boundary"),
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.05, 0.5),
              facecolor="#f7f7fa", edgecolor="#d8d8e0", labelcolor="#1a1a2e", fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out = RESULTS_DIR / "20_all_emissions_comparison_bar.svg"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")


def plot_pipeline_stacked():
    train_df = pd.read_csv(RESULTS_DIR / "training_emissions.csv")
    inf_df = pd.read_csv(RESULTS_DIR / "inference_emissions.csv")

    extras_path = RESULTS_DIR / "extras_training_emissions.csv"
    if extras_path.exists():
        extras_df = pd.read_csv(extras_path)
        train_df = pd.concat([train_df, extras_df], ignore_index=True)

    df = pd.merge(train_df, inf_df, on=["Strategy", "Model"], how="left", suffixes=("_train", "_inf"))

    apply_style()

    MODELS = ["BPR", "MultiDAE", "EASE"]

    target_strategies = [
        "Baseline",
        "User-based (−20%)",
        "Item-based (−20%)",
        "User-temporal (last 300)",
        "Global-temporal (Sep 2000)",
        "Global-temp+MinK3 (Sep 2000)",
        "Coreset-Leverage (−20%)",
        "Coreset-Leverage (−30%)",
        "Coreset-Cluster (−20%)",
        "Coreset-Cluster (−30%)",
        "CS-Cluster-Outlier (−20%)",
        "CS-Cluster-Outlier (−30%)"
    ]

    def normalize_name(name):
        name = str(name).replace("−", "-").strip()
        if "MinK3" in name: return "Global-temporal min-k(Sep 2000)"
        if "Outlier" in name and "20" in name: return "Coreset-Cluster-outlier (-20%)"
        if "Outlier" in name and "30" in name: return "Coreset-Cluster-outlier (-30%)"
        return name

    df["Strategy_norm"] = df["Strategy"].apply(normalize_name)
    target_norm = [normalize_name(s) for s in target_strategies]

    df = df.drop_duplicates(subset=["Model", "Strategy_norm"], keep="last")
    df = df[df["Strategy_norm"].isin(target_norm)]

    df["Data_Processing"] = df["Preprocess_kWh_mean_train"].fillna(0)
    df["Training"] = df["Energy_kWh_mean_train"].fillna(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    def short_label(s):
        if "Baseline" in s: return "Baseline"
        if "User-based" in s: return "User\n-based (-20%)"
        if "Item-based" in s: return "Item\n-based (-20%)"
        if "User-temporal" in s: return "User\n-temp. (last 300)"
        if "min-k" in s: return "Global\n-temp. min-k"
        if "Global-temporal" in s: return "Global\n-temp. (Sep 2000)"
        if "Coreset-Leverage" in s:
            if "20" in s: return "CS-Lev. (-20%)"
            if "30" in s: return "CS-Lev. (-30%)"
        if "outlier" in s:
            if "20" in s: return "CS-Clust.\nOutlier (-20%)"
            if "30" in s: return "CS-Clust.\nOutlier (-30%)"
        if "Coreset-Cluster" in s:
            if "20" in s: return "CS-Clust. (-20%)"
            if "30" in s: return "CS-Clust. (-30%)"
        return s

    for i, model in enumerate(MODELS):
        ax = axes[i]
        model_df = df[df["Model"] == model].copy()

        model_df["sort_key"] = pd.Categorical(model_df["Strategy_norm"], categories=target_norm, ordered=True)
        model_df = model_df.sort_values("sort_key")

        x_labels = [short_label(s) for s in model_df["Strategy_norm"]]
        x = np.arange(len(x_labels))

        dp = model_df["Data_Processing"].values
        tr = model_df["Training"].values

        w = 0.6

        c_dp = "#7678ed"
        c_tr = "#f7b267"

        bar1 = ax.bar(x, dp, w, color=c_dp, label="Data processing (load + reduction)", edgecolor='black', linewidth=0.5)
        bar2 = ax.bar(x, tr, w, bottom=dp, color=c_tr, label="Training", edgecolor='black', linewidth=0.5)

        for j in range(len(x)):
            total = dp[j] + tr[j]
            lbl = f"${total * 1e4:.2f} \\times 10^{{-4}}$" if total > 0 else "0"
            ax.text(x[j], total + (max(tr)*0.03), lbl, ha="center", va="bottom", fontsize=7, color="#1a1a2e", fontweight="bold")

        ax.set_title(f"Model: {model}", fontsize=12, fontweight="bold", color="#1a1a2e")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Total energy (kWh)", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        if i == 0:
            ax.legend(fontsize=9, loc="upper left")

        max_energy = (df["Data_Processing"] + df["Training"]).max()
        ax.set_ylim(0, max_energy * 1.15)

    axes[3].set_visible(False)

    fig.suptitle("Training Pipeline Carbon Cost per Reduction Strategy (By Model)", fontsize=16, fontweight="bold", color="#1a1a2e", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = RESULTS_DIR / "22_training_pipeline_stacked.svg"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_all_emissions_bar()
    plot_pipeline_stacked()

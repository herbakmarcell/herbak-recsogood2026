import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import apply_style, model_color

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

def plot_inference_latency():
    df = pd.read_csv(RESULTS_DIR / "inference_emissions.csv")
    df["Duration_ms"] = df["Duration_s"] * 1000

    models = df["Model"].unique().tolist()
    strategies = df["Strategy"].unique().tolist()

    apply_style()

    model_colors = {m: model_color(m, i) for i, m in enumerate(models)}

    n_strats  = len(strategies)
    n_models  = len(models)
    bar_w     = max(0.10, 0.72 / n_models)
    x         = np.arange(n_strats)

    fig, ax = plt.subplots(figsize=(14, 6))

    for mi, model_name in enumerate(models):
        sub = df[df["Model"] == model_name]
        vals = []
        for s in strategies:
            matches = sub.loc[sub["Strategy"] == s, "Duration_ms"].values
            vals.append(matches[0] if len(matches) > 0 else 0)

        offset = (mi - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w,
                      color=model_colors.get(model_name, "#aaaacc"),
                      edgecolor="none", label=model_name, alpha=0.9)

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7, color="#1a1a2e", rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Inference Time per call (milliseconds)", fontsize=11, labelpad=10)
    ax.set_title(
        "Inference Latency by Model & Strategy (Time to score users)",
        fontsize=14, fontweight="bold", color="#1a1a2e", pad=18,
    )

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[m], label=m) for m in models]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="#f7f7fa", edgecolor="#d8d8e0", labelcolor="#1a1a2e",
        fontsize=10,
    )

    plt.tight_layout()
    out_path = RESULTS_DIR / "14_inference_time.svg"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved inference time plot to {out_path}")

if __name__ == "__main__":
    plot_inference_latency()

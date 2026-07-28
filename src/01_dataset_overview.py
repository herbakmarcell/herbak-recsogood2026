"""
01_dataset_overview.py
======================
1. Downloads MovieLens-1M (if not already present)
2. Plots the movie genre distribution
3. Shows dataset size (users / items / ratings) before and after
   each of the four data reduction strategies from the paper
"""

import io
import sys
import zipfile
import urllib.request
import warnings
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.data_utils import coreset_leverage_reduction, coreset_cluster_reduction, build_global_split
from utils.emission_utils import measure_repeated
from utils.plot_style import apply_style, PALETTE

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "data" / "ml-1m"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

apply_style()


def download_ml1m():
    movies_path = DATA_DIR / "movies.dat"
    ratings_path = DATA_DIR / "ratings.dat"
    if movies_path.exists() and ratings_path.exists():
        print("[OK] MovieLens-1M already downloaded.")
        return
    print(f"[>>] Downloading MovieLens-1M from {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(DATA_DIR.parent)
    print("[OK] Download complete.")


def load_data():
    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return movies, ratings


def plot_genre_distribution(movies: pd.DataFrame):
    all_genres = []
    for genres_str in movies["genres"]:
        all_genres.extend(genres_str.split("|"))

    genre_counts = Counter(all_genres)
    genre_df = (
        pd.DataFrame.from_dict(genre_counts, orient="index", columns=["count"])
        .sort_values("count", ascending=True)
        .reset_index()
        .rename(columns={"index": "genre"})
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(
        genre_df["genre"],
        genre_df["count"],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(genre_df))],
        edgecolor="none",
        height=0.65,
    )

    for bar, count in zip(bars, genre_df["count"]):
        ax.text(
            bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", ha="left",
            color="#1a1a2e", fontsize=8.5,
        )

    ax.set_xlabel("Number of Movies", fontsize=11, labelpad=10)
    ax.set_title(
        "MovieLens-1M - Genre Distribution",
        fontsize=15, fontweight="bold", color="#1a1a2e", pad=18,
    )
    ax.set_xlim(0, genre_df["count"].max() * 1.08)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    total_movies = len(movies)
    unique_genres = len(genre_counts)
    fig.text(
        0.5, 0.01,
        f"{total_movies:,} movies · {unique_genres} genres  (multi-genre movies counted once per genre)",
        ha="center", fontsize=8.5, color="#5a5a6e",
        transform=fig.transFigure,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = RESULTS_DIR / "01_genre_distribution.svg"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Genre distribution plot saved -> {out}")
    return genre_df


def user_based_reduction(df: pd.DataFrame, k: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Remove k% of ratings randomly per user (stratified)."""
    rng = np.random.default_rng(seed)
    keep_indices = []
    for _, group in df.groupby("user_id"):
        n_remove = int(len(group) * k)
        remove_idx = rng.choice(group.index, size=n_remove, replace=False)
        keep_indices.extend(set(group.index) - set(remove_idx))
    return df.loc[keep_indices].reset_index(drop=True)


def item_based_reduction(df: pd.DataFrame, k: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Remove k% of ratings randomly per item (stratified)."""
    rng = np.random.default_rng(seed)
    keep_indices = []
    for _, group in df.groupby("movie_id"):
        n_remove = int(len(group) * k)
        remove_idx = rng.choice(group.index, size=n_remove, replace=False)
        keep_indices.extend(set(group.index) - set(remove_idx))
    return df.loc[keep_indices].reset_index(drop=True)


def user_temporal_reduction(df: pd.DataFrame, n: int = 250) -> pd.DataFrame:
    """For each user keep only their last n interactions (sorted by timestamp)."""
    return (
        df.sort_values("timestamp")
        .groupby("user_id")
        .tail(n)
        .reset_index(drop=True)
    )


def global_temporal_reduction(df: pd.DataFrame, cutoff: str = "2000-08-01") -> pd.DataFrame:
    """Keep only ratings recorded after cutoff date."""
    cutoff_dt = pd.to_datetime(cutoff)
    return df[df["timestamp"] >= cutoff_dt].reset_index(drop=True)


def dataset_stats(df: pd.DataFrame, label: str) -> dict:
    return {
        "Strategy": label,
        "Users": df["user_id"].nunique(),
        "Items": df["movie_id"].nunique(),
        "Ratings": len(df),
    }


def compute_size_table(ratings: pd.DataFrame) -> pd.DataFrame:
    baseline = dataset_stats(ratings, "Baseline (full)")

    reduced = [
        baseline,
        dataset_stats(user_based_reduction(ratings, k=0.20),    "User-based (-20%)"),
        dataset_stats(item_based_reduction(ratings, k=0.20),     "Item-based (-20%)"),
        dataset_stats(user_temporal_reduction(ratings, n=300),   "User-temporal (last 300)"),
        dataset_stats(user_temporal_reduction(ratings, n=250),   "User-temporal (last 250)"),
        dataset_stats(user_temporal_reduction(ratings, n=200),   "User-temporal (last 200)"),
        dataset_stats(global_temporal_reduction(ratings, "2000-09-01"), "Global-temporal (Sep 2000)"),
        dataset_stats(global_temporal_reduction(ratings, "2000-08-01"), "Global-temporal (Aug 2000)"),
        dataset_stats(global_temporal_reduction(ratings, "2000-07-01"), "Global-temporal (Jul 2000)"),
    ]

    df_stats = pd.DataFrame(reduced)
    baseline_ratings = baseline["Ratings"]
    df_stats["Δ Ratings"] = df_stats["Ratings"].apply(
        lambda x: f"{(x - baseline_ratings) / baseline_ratings * 100:+.1f}%"
    )
    return df_stats


def plot_size_comparison(df_stats: pd.DataFrame):
    strategies = df_stats["Strategy"].tolist()
    ratings = df_stats["Ratings"].tolist()
    baseline_val = ratings[0]

    colors = []
    for i, s in enumerate(strategies):
        if i == 0:
            colors.append("#6C63FF")
        elif "User-based" in s:
            colors.append("#FF6584")
        elif "Item-based" in s:
            colors.append("#43BF95")
        elif "User-temporal" in s:
            colors.append("#F4A261")
        elif "Coreset-Leverage" in s:
            colors.append("#E040FB")
        elif "Coreset-Cluster" in s:
            colors.append("#FF7043")
        else:
            colors.append("#2196F3")

    fig, ax = plt.subplots(figsize=(13, max(6, len(strategies) * 0.55)))

    bars = ax.barh(strategies, ratings, color=colors, edgecolor="none", height=0.62)

    ax.axvline(baseline_val, color="#4C63FF", linewidth=1.2, linestyle="--", alpha=0.5)

    for bar, val, s in zip(bars, ratings, strategies):
        pct = (val - baseline_val) / baseline_val * 100
        label = f"{val:,.0f}" if s == "Baseline (full)" else f"{val:,.0f}  ({pct:+.1f}%)"
        ax.text(
            bar.get_width() + 3_000, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", color="#1a1a2e", fontsize=8.5,
        )

    ax.set_xlabel("Number of Ratings", fontsize=11, labelpad=10)
    ax.set_title(
        "MovieLens-1M - Dataset Size Before & After Data Reduction",
        fontsize=14, fontweight="bold", color="#1a1a2e", pad=16,
    )
    ax.set_xlim(0, max(ratings) * 1.18)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1_000)}K"))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#6C63FF", label="Baseline"),
        Patch(facecolor="#FF6584", label="User-based (-20%)"),
        Patch(facecolor="#43BF95", label="Item-based (-20%)"),
        Patch(facecolor="#F4A261", label="User-temporal"),
        Patch(facecolor="#2196F3", label="Global-temporal"),
        Patch(facecolor="#E040FB", label="Coreset-Leverage"),
        Patch(facecolor="#FF7043", label="Coreset-Cluster"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0,
        facecolor="#f7f7fa", edgecolor="#d8d8e0", labelcolor="#1a1a2e",
        fontsize=8.5,
    )

    plt.tight_layout()
    out = RESULTS_DIR / "02_dataset_size_reduction.svg"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Dataset size comparison plot saved -> {out}")


def plot_emissions(em_df: pd.DataFrame, size_table: pd.DataFrame):
    """
    Stacked horizontal bar chart with 3 segments per reduction strategy:

      [purple] Reduced load cost  = baseline_load * (reduced_ratings / full_ratings)
               (proportional estimate: a smaller dataset loads proportionally faster)
      [red   ] Reduction op cost  = actual measured cost of applying the reduction
      [green ] Net energy saved vs the full baseline load

    A dashed vertical line marks the baseline full-load boundary.
    """
    from matplotlib.patches import Patch

    baseline_load_kwh = em_df.loc[em_df["Operation"] == "Load dataset", "Energy_kWh_mean"].values[0]
    baseline_ratings  = size_table.loc[size_table["Strategy"] == "Baseline (full)", "Ratings"].values[0]
    red_em = em_df[em_df["Operation"] != "Load dataset"].copy().reset_index(drop=True)

    red_sizes = size_table[size_table["Strategy"] != "Baseline (full)"].reset_index(drop=True)

    labels           = red_em["Operation"].tolist()
    red_op_kwh       = red_em["Energy_kWh_mean"].tolist()
    reduced_load_kwh = [
        baseline_load_kwh * (r / baseline_ratings)
        for r in red_sizes["Ratings"].tolist()
    ]

    n = len(labels)
    h = 0.45
    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.7)))

    net_savings_kwh = [
        max(baseline_load_kwh - rl - ro, 0)
        for rl, ro in zip(reduced_load_kwh, red_op_kwh)
    ]

    ax.barh(labels, reduced_load_kwh,
            height=h, color="#4C63FF", edgecolor="none",
            label="Reduced load cost (purple)")

    ax.barh(labels, red_op_kwh, left=reduced_load_kwh,
            height=h, color="#E0527A", edgecolor="none",
            label="Reduction operation cost (red)")

    left_for_green = [rl + ro for rl, ro in zip(reduced_load_kwh, red_op_kwh)]
    ax.barh(labels, net_savings_kwh, left=left_for_green,
            height=h, color="#2FA787", edgecolor="none",
            label="Net energy saved vs baseline (green)")

    ax.axvline(baseline_load_kwh, color="#8a8aa0", linewidth=1.3,
               linestyle="--", alpha=0.7,
               label=f"Baseline load ({baseline_load_kwh:.8f} kWh)")

    max_total = max(rl + ro + sv for rl, ro, sv in zip(reduced_load_kwh, red_op_kwh, net_savings_kwh))
    max_x = max(max_total, baseline_load_kwh) * 1.25
    pad   = max_x * 0.012

    for i, (rl, ro, sv) in enumerate(zip(reduced_load_kwh, red_op_kwh, net_savings_kwh)):
        total = rl + ro + sv

        if rl > max_x * 0.06:
            ax.text(rl / 2, i, f"Load\n{rl:.8f} kWh",
                    va="center", ha="center",
                    color="#ffffff", fontsize=6.5, fontweight="bold")

        if ro > max_x * 0.015:
            ax.text(rl + ro / 2, i, f"Op\n{ro:.8f} kWh",
                    va="center", ha="center",
                    color="#ffffff", fontsize=6.0, fontweight="bold")

        if sv > max_x * 0.04:
            ax.text(rl + ro + sv / 2, i, f"Saved\n{sv:.8f} kWh",
                    va="center", ha="center",
                    color="#0f0f1a", fontsize=6.5, fontweight="bold")

        net_kwh = baseline_load_kwh - rl - ro
        if net_kwh >= 0:
            pct_txt = f"saves {net_kwh / baseline_load_kwh * 100:.1f}%"
            txt_col = "#1a1a2e"
        else:
            pct_txt = f"overhead +{-net_kwh / baseline_load_kwh * 100:.1f}%"
            txt_col = "#E0527A"
        ax.text(total + pad, i, pct_txt,
                va="center", ha="left", color=txt_col, fontsize=7.5)

    ax.set_xlabel("Energy consumed (kWh)", fontsize=11, labelpad=10)
    ax.set_title(
        "Energy Consumption per Reduction Strategy  (MovieLens-1M)",
        fontsize=13, fontweight="bold", color="#1a1a2e", pad=18,
    )
    ax.set_xlim(0, max_x)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.6f}"))

    legend_elements = [
        Patch(facecolor="#4C63FF", label="Reduced load cost"),
        Patch(facecolor="#E0527A", label="Reduction operation cost"),
        Patch(facecolor="#2FA787", label="Net energy saved vs baseline"),
        plt.Line2D([0], [0], color="#8a8aa0", linewidth=1.3,
                   linestyle="--", alpha=0.8, label="Baseline load boundary"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        borderaxespad=0,
        facecolor="#f7f7fa", edgecolor="#d8d8e0", labelcolor="#1a1a2e",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    out = RESULTS_DIR / "03_emissions.svg"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Emissions plot saved -> {out}")


if __name__ == "__main__":
    download_ml1m()

    print("\n[ENERGY] Measuring energy consumption for each operation ...")
    (movies, ratings), em_kwh, em_std, dur_load, _ = measure_repeated(
        "Load dataset", load_data,
        results_dir=RESULTS_DIR, warmup=1, repeats=3, runs=3,
    )
    print(f"\n[DATA] Loaded {len(movies):,} movies and {len(ratings):,} ratings")
    print(f"   Users: {ratings['user_id'].nunique():,}  |  "
          f"Items: {ratings['movie_id'].nunique():,}  |  "
          f"Date range: {ratings['timestamp'].min().date()} -> {ratings['timestamp'].max().date()}")

    print("\n[PLOT] Plotting genre distribution ...")
    genre_df = plot_genre_distribution(movies)

    print("\n[REDUCTION] Applying reductions and measuring energy ...")
    train_raw_df, _, _ = build_global_split(ratings)
    print(f"[DATA] Operating on train split: {len(train_raw_df):,} ratings")

    reduction_tasks = [
        ("User-based (-20%)",              user_based_reduction,        dict(df=train_raw_df, k=0.20)),
        ("Item-based (-20%)",              item_based_reduction,        dict(df=train_raw_df, k=0.20)),
        ("User-temporal (last 300)",        user_temporal_reduction,     dict(df=train_raw_df, n=300)),
        ("User-temporal (last 250)",        user_temporal_reduction,     dict(df=train_raw_df, n=250)),
        ("User-temporal (last 200)",        user_temporal_reduction,     dict(df=train_raw_df, n=200)),
        ("Global-temporal (Sep 2000)",      global_temporal_reduction,   dict(df=train_raw_df, cutoff="2000-09-01")),
        ("Global-temporal (Aug 2000)",      global_temporal_reduction,   dict(df=train_raw_df, cutoff="2000-08-01")),
        ("Global-temporal (Jul 2000)",      global_temporal_reduction,   dict(df=train_raw_df, cutoff="2000-07-01")),
        ("Coreset-Leverage (-20%)",         coreset_leverage_reduction,  dict(df=train_raw_df, fraction=0.80)),
        ("Coreset-Leverage (-30%)",         coreset_leverage_reduction,  dict(df=train_raw_df, fraction=0.70)),
        ("Coreset-Cluster (-20%)",          coreset_cluster_reduction,   dict(df=train_raw_df, fraction=0.80)),
        ("Coreset-Cluster (-30%)",          coreset_cluster_reduction,   dict(df=train_raw_df, fraction=0.70)),
    ]

    emission_records = [{
        "Operation": "Load dataset", "Energy_kWh_mean": em_kwh, "Energy_kWh_std": em_std,
        "Duration_ms": dur_load * 1000,
    }]
    size_records = [dataset_stats(train_raw_df, "Baseline (full)")]

    for label, fn, kwargs in reduction_tasks:
        reduced_df, kwh_mean, kwh_std, dur_s, _ = measure_repeated(
            label, fn, results_dir=RESULTS_DIR, warmup=1, repeats=5, runs=5, **kwargs,
        )
        emission_records.append({
            "Operation": label, "Energy_kWh_mean": kwh_mean, "Energy_kWh_std": kwh_std,
            "Duration_ms": dur_s * 1000,
        })
        stats = dataset_stats(reduced_df, label)
        size_records.append(stats)

    em_df = pd.DataFrame(emission_records)
    size_table = pd.DataFrame(size_records)
    baseline_ratings = size_records[0]["Ratings"]
    size_table["Delta Ratings"] = size_table["Ratings"].apply(
        lambda x: f"{(x - baseline_ratings) / baseline_ratings * 100:+.1f}%"
    )

    print("\n" + "=" * 72)
    print("DATASET SIZES:")
    print(size_table.to_string(index=False))
    print("\nENERGY CONSUMPTION (kWh, mean ± std over repeated measurements):")
    print(em_df.to_string(index=False))
    print("=" * 72)

    size_table.to_csv(RESULTS_DIR / "dataset_sizes.csv", index=False)
    em_df.to_csv(RESULTS_DIR / "emissions.csv", index=False)
    print(f"\n[OK] CSVs saved -> {RESULTS_DIR}")

    print("\n[PLOT] Plotting dataset size comparison ...")
    plot_size_comparison(size_table)

    print("[PLOT] Plotting energy consumption ...")
    plot_emissions(em_df, size_table)

    print("\n[DONE] All plots and CSVs saved to 'results/'")

"""
02_train_models.py
==================
Stage 2: Train BPR, MultiDAE, and EASE on every dataset variant
(baseline + 8 reduction strategies) while tracking energy consumption.

Key design: the train/val/test split is performed ONCE on the raw data
(build_global_split), so every strategy is evaluated on the same fixed
test set.  Reduction functions operate only on the training portion to
prevent any data leakage.

Seeded runs (statistical validity)
-----------------------------------
Every (strategy, model) is trained/evaluated across `eval_utils.DEFAULT_SEEDS`
(5 seeds) rather than once. Each seed varies BOTH the reduction-strategy
randomness (where applicable) and the model's own training randomness, so a
single seed loop yields mean +/- std for energy AND accuracy from the same
computational budget - no separate "repeat for energy noise" pass is needed
on top of it. A paired Wilcoxon signed-rank test (vs Baseline, per model) is
then computed from the raw per-seed Recall@K arrays, so "+X% Recall" claims
come with a significance figure instead of being a single noisy point
estimate.

Outputs
-------
models/<strategy_slug>/<model_name>/   - saved cornac model (canonical seed only)
results/training_emissions.csv         - one row per (strategy, model), mean+-std across seeds
results/04_training_emissions.svg      - grouped bar chart with error bars

Usage
-----
    python src/02_train_models.py
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.data_utils     import download_ml1m, load_data, REDUCTION_STRATEGIES, \
                                   apply_reduction, build_global_split, build_cornac_dataset, \
                                   slugify, MODELS_DIR
from utils.emission_utils import measure_emissions
from utils.eval_utils     import (
    DEFAULT_SEEDS, evaluate_model, build_common_item_vocabulary,
    build_common_subset_candidates, paired_wilcoxon_pvalue,
    split_users_by_history_length,
)
from utils.plot_style     import apply_style, get_cmap, model_color

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

K          = 64     # embedding / latent dimension
N_EPOCHS   = 50
LR         = 1e-3
BATCH_SIZE = 256
TOPK       = 10     # evaluation cut-off

CANONICAL_SEED = DEFAULT_SEEDS[0]   # "the" seed used for saved models / common-vocab reference

import torch as _torch
USE_GPU = _torch.cuda.is_available()
print(f"[DEVICE] {'GPU: ' + _torch.cuda.get_device_name(0) if USE_GPU else 'CPU (no CUDA)'}")


def make_bpr(seed=CANONICAL_SEED):
    import cornac
    return cornac.models.BPR(
        k=K, max_iter=N_EPOCHS, learning_rate=LR,
        lambda_reg=1e-4, seed=seed, verbose=False,
    )


def make_multidae(seed=CANONICAL_SEED):
    """
    MultiDAE = VAE with beta=0 (pure reconstruction, no KL term).
    Cornac's VAECF with beta=0.0 exactly replicates MultiDAE behaviour
    from Liang et al. 2018.
    """
    import cornac
    return cornac.models.VAECF(
        k=K,
        autoencoder_structure=[600],    # standard hidden layer from paper
        act_fn="tanh",
        likelihood="mult",
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        beta=0.0,                       # 0 = DAE; >0 = VAE
        seed=seed,
        use_gpu=USE_GPU,
        verbose=False,
    )


def make_ease(seed=CANONICAL_SEED):
    """EASE is a deterministic closed-form solution - seed has no effect on
    the model itself; accuracy variance across seeds for EASE comes only
    from the reduction strategy's own randomness (if any)."""
    import cornac
    return cornac.models.EASE(lamb=500, verbose=False)


MODEL_FACTORIES = {
    "BPR":      make_bpr,
    "MultiDAE": make_multidae,
    "EASE":     make_ease,
}


# Only one representative strategy per group is trained, to keep the sweep tractable
ACTIVE_STRATEGIES = {
    "Baseline",
    "User-based (-20%)",
    "Item-based (-20%)",
    "User-temporal (last 300)",
    "Global-temporal (Sep 2000)",
    "Coreset-Leverage (-20%)",
    "Coreset-Leverage (-30%)",
    "Coreset-Cluster (-20%)",
    "Coreset-Cluster (-30%)",
}


def _train_seeded(factory, train_set, seed):
    """Build a brand-new model with *seed* from *factory* and fit it."""
    model = factory(seed=seed)
    model.fit(train_set)
    return model


def run():
    print("[>>] Loading MovieLens-1M ...")
    download_ml1m()
    _, ratings = load_data()
    print(f"[OK] Loaded {len(ratings):,} ratings")

    print("[>>] Building global fixed train/val/test split ...")
    train_raw_df, global_val_df, global_test_df = build_global_split(ratings)
    n_global_test = len(global_test_df)
    print(
        f"[OK] Split: {len(train_raw_df):,} train  |  "
        f"{len(global_val_df):,} val  |  {n_global_test:,} test\n"
    )

    active = [(n, f) for n, f in REDUCTION_STRATEGIES if n in ACTIVE_STRATEGIES]

    # Untracked warm-up done once globally (not per strategy/seed): first-call
    # overhead from imports/cuDNN autotuning doesn't reset between later calls
    # within the same process, so repeating this per job would be redundant cost.
    print("[>>] Warm-up: priming model backends (untracked, discarded) ...")
    warmup_train_set, _, _, _ = build_cornac_dataset(train_raw_df, global_val_df, global_test_df)
    for model_name, factory in MODEL_FACTORIES.items():
        try:
            _train_seeded(factory, warmup_train_set, CANONICAL_SEED)
        except Exception as e:
            print(f"   [WARN] Warm-up failed for {model_name}: {e}")
    del warmup_train_set
    print("[OK] Warm-up complete.\n")

    print("[>>] Computing common item vocabulary across active strategies ...")
    strategy_names = [n for n, _ in active]
    common_items = build_common_item_vocabulary(
        train_raw_df, strategy_names, apply_reduction, seed=CANONICAL_SEED,
    )
    common_subset_candidates = build_common_subset_candidates(
        global_test_df, common_items, n_neg=100,
    )
    print(
        f"[OK] Common item vocabulary: {len(common_items):,} items  |  "
        f"{len(common_subset_candidates):,} test users have their true item in it\n"
    )

    print("[>>] Splitting users by historical interaction length ...")
    user_groups, group_counts = split_users_by_history_length(train_raw_df, global_test_df)
    print(f"[OK] Test user group sizes: Short={group_counts['short']}, Medium={group_counts['medium']}, Long={group_counts['long']}\n")

    records         = []
    raw_recalls     = {}   # (strategy, model) -> list of per-seed Recall@K, for Wilcoxon
    PROGRESS_LOG    = RESULTS_DIR / "training_progress.log"
    total_jobs      = len(active) * len(MODEL_FACTORIES) * len(DEFAULT_SEEDS)
    job_idx         = 0

    PROGRESS_LOG.write_text(
        f"Training started: {datetime.now().strftime('%H:%M:%S')}\n"
        f"Total jobs: {total_jobs}  "
        f"({len(active)} strategies x {len(MODEL_FACTORIES)} models x {len(DEFAULT_SEEDS)} seeds)\n"
        f"Global test users: {n_global_test}\n"
        f"Seeds: {DEFAULT_SEEDS}\n"
        + "=" * 60 + "\n",
        encoding="utf-8",
    )

    for strategy_name, reduce_fn in active:
        slug = slugify(strategy_name)
        print(f"\n{'='*72}")
        print(f"  Strategy: {strategy_name}")
        print(f"{'='*72}")

        _ = reduce_fn(train_raw_df, seed=CANONICAL_SEED)   # per-strategy warm-up, untracked

        per_model = {m: {
            "preprocess_kwh": [], "energy_kwh": [], "duration_s": [],
            "recall": [], "ndcg": [], "recall_evaluable": [], "eval_user_rate": [],
            "coverage": [], "recall_common_subset": [],
            "recall_short": [], "recall_medium": [], "recall_long": [],
            "per_user_recalls": [],
        } for m in MODEL_FACTORIES}
        train_sizes, oov_counts = [], []

        for seed in DEFAULT_SEEDS:
            reduced_train, preprocess_kwh, _ = measure_emissions(
                f"Preprocess | {strategy_name} | seed={seed}",
                reduce_fn, train_raw_df, seed=seed,
                results_dir=RESULTS_DIR,
                output_file="preprocess_codecarbon_raw.csv",
            )
            train_sizes.append(len(reduced_train))

            train_set, val_set, test_set, meta = build_cornac_dataset(
                reduced_train, global_val_df, global_test_df
            )
            oov = meta["n_oov_test"]
            oov_counts.append(oov)

            for model_name, factory in MODEL_FACTORIES.items():
                job_idx += 1
                print(f"  [JOB {job_idx}/{total_jobs}] {model_name} | {strategy_name} | seed={seed}")

                label = f"{model_name} | {strategy_name} | seed={seed}"
                trained_model, energy_kwh, duration_s = measure_emissions(
                    label, _train_seeded, factory, train_set, seed,
                    results_dir=RESULTS_DIR,
                    output_file="training_codecarbon_raw.csv",
                )

                if seed == CANONICAL_SEED:
                    save_dir = MODELS_DIR / slug / model_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        trained_model.save(save_dir=str(save_dir))
                    except Exception as e:
                        print(f"   [WARN] Could not save model: {e}")

                scores = evaluate_model(
                    trained_model, train_set, test_set, meta["item_map"],
                    n_global_test, topk=TOPK, seed=seed,
                    common_subset_candidates=common_subset_candidates,
                    user_groups=user_groups,
                    group_counts=group_counts,
                )

                d = per_model[model_name]
                d["preprocess_kwh"].append(preprocess_kwh)
                d["energy_kwh"].append(energy_kwh)
                d["duration_s"].append(duration_s)
                d["recall"].append(scores[f"Recall@{TOPK}"])
                d["ndcg"].append(scores[f"NDCG@{TOPK}"])
                d["recall_evaluable"].append(scores[f"Recall@{TOPK}_evaluable"])
                d["eval_user_rate"].append(scores["Eval_User_Rate"])
                d["coverage"].append(scores.get("Coverage", float("nan")))
                d["recall_common_subset"].append(scores.get(f"Recall@{TOPK}_common_subset", float("nan")))
                d["recall_short"].append(scores.get(f"Recall@{TOPK}_short", float("nan")))
                d["recall_medium"].append(scores.get(f"Recall@{TOPK}_medium", float("nan")))
                d["recall_long"].append(scores.get(f"Recall@{TOPK}_long", float("nan")))
                d["per_user_recalls"].append(scores["per_user_recalls"])

                print(
                    f"     Recall@{TOPK}={scores[f'Recall@{TOPK}']:.4f}  "
                    f"Coverage={scores['Coverage']:.3f}  "
                    f"Energy={energy_kwh:.8f} kWh"
                )

        for model_name, d in per_model.items():
            rec = {
                "Strategy":                        strategy_name,
                "Model":                            model_name,
                "N_seeds":                          len(DEFAULT_SEEDS),
                # nan-safe: a seed's energy tracking can fail after retries
                # (see emission_utils._run_tracked) without invalidating the
                # rest of that strategy's seeded runs.
                "Preprocess_kWh_mean":               float(np.nanmean(d["preprocess_kwh"])),
                "Preprocess_kWh_std":                float(np.nanstd(d["preprocess_kwh"])),
                "Energy_kWh_mean":                   float(np.nanmean(d["energy_kwh"])),
                "Energy_kWh_std":                    float(np.nanstd(d["energy_kwh"])),
                "Duration_s":                         float(np.nanmean(d["duration_s"])),
                "Duration_s_std":                     float(np.nanstd(d["duration_s"])),
                f"Recall@{TOPK}":                    float(np.mean(d["recall"])),
                f"Recall@{TOPK}_std":                float(np.std(d["recall"])),
                f"NDCG@{TOPK}":                       float(np.mean(d["ndcg"])),
                f"NDCG@{TOPK}_std":                   float(np.std(d["ndcg"])),
                f"Recall@{TOPK}_evaluable":           float(np.mean(d["recall_evaluable"])),
                "Eval_User_Rate":                     float(np.mean(d["eval_user_rate"])),
                "Coverage":                           float(np.mean(d["coverage"])),
                "Coverage_std":                       float(np.std(d["coverage"])),
                f"Recall@{TOPK}_common_subset":       float(np.nanmean(d["recall_common_subset"])),
                f"Recall@{TOPK}_short":               float(np.nanmean(d["recall_short"])),
                f"Recall@{TOPK}_medium":              float(np.nanmean(d["recall_medium"])),
                f"Recall@{TOPK}_long":                float(np.nanmean(d["recall_long"])),
                "Train_size":                         int(round(np.mean(train_sizes))),
                "OOV_test_users":                     float(np.mean(oov_counts)),
            }
            records.append(rec)
            raw_recalls[(strategy_name, model_name)] = d["per_user_recalls"][:]

        with open(PROGRESS_LOG, "a", encoding="utf-8") as pf:
            for model_name, d in per_model.items():
                pf.write(
                    f"{datetime.now().strftime('%H:%M:%S')}  "
                    f"{model_name:<10} | {strategy_name:<30} | "
                    f"Recall@{TOPK}={np.mean(d['recall']):.4f}+-{np.std(d['recall']):.4f}  "
                    f"Energy={np.mean(d['energy_kwh']):.8f} kWh  "
                    f"Coverage={np.mean(d['coverage']):.3f}\n"
                )

    def get_avg_user_scores(scores_list, subset_uids):
        avg_scores = []
        for uid in subset_uids:
            s = np.mean([d.get(uid, 0.0) for d in scores_list])
            avg_scores.append(s)
        return avg_scores

    all_uids = global_test_df["user_id"].unique()
    short_uids = [u for u, g in user_groups.items() if g == "short"] if user_groups else []
    medium_uids = [u for u, g in user_groups.items() if g == "medium"] if user_groups else []
    long_uids = [u for u, g in user_groups.items() if g == "long"] if user_groups else []

    for rec in records:
        if rec["Strategy"] == "Baseline":
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_short"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_medium"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_long"] = float("nan")
            continue

        baseline_recalls = raw_recalls.get(("Baseline", rec["Model"]))
        strat_recalls    = raw_recalls.get((rec["Strategy"], rec["Model"]))

        if baseline_recalls is not None and strat_recalls is not None:
            b_all = get_avg_user_scores(baseline_recalls, all_uids)
            s_all = get_avg_user_scores(strat_recalls, all_uids)
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline"] = paired_wilcoxon_pvalue(s_all, b_all)

            if user_groups:
                b_s = get_avg_user_scores(baseline_recalls, short_uids)
                s_s = get_avg_user_scores(strat_recalls, short_uids)
                rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_short"] = paired_wilcoxon_pvalue(s_s, b_s)

                b_m = get_avg_user_scores(baseline_recalls, medium_uids)
                s_m = get_avg_user_scores(strat_recalls, medium_uids)
                rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_medium"] = paired_wilcoxon_pvalue(s_m, b_m)

                b_l = get_avg_user_scores(baseline_recalls, long_uids)
                s_l = get_avg_user_scores(strat_recalls, long_uids)
                rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_long"] = paired_wilcoxon_pvalue(s_l, b_l)
            else:
                rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_short"] = float("nan")
                rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_medium"] = float("nan")
                rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_long"] = float("nan")
        else:
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_short"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_medium"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_long"] = float("nan")

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "training_emissions_wilcoxon.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Training emissions CSV saved -> {csv_path}")

    print("[DONE] All models trained and saved.\n")


def plot_training_emissions(df: pd.DataFrame):
    apply_style()

    models     = df["Model"].unique().tolist()
    strategies = df["Strategy"].unique().tolist()

    # model_color() falls back to the shared PALETTE (matplotlib>=3.9-safe,
    # see utils/plot_style.py) for any model outside the curated MODEL_COLORS
    _KNOWN = {"BPR", "MultiDAE", "EASE", "ELSA", "LightGCN", "NeuMF"}
    extra_idx = 0
    model_colors = {}
    for m in models:
        model_colors[m] = model_color(m, extra_idx)
        if m not in _KNOWN:
            extra_idx += 1

    energy_col = "Energy_kWh_mean" if "Energy_kWh_mean" in df.columns else "CO2_g"
    std_col    = "Energy_kWh_std" if "Energy_kWh_std" in df.columns else None

    n_strats  = len(strategies)
    n_models  = len(models)
    bar_w     = max(0.10, 0.72 / n_models)
    x         = np.arange(n_strats)

    fig_w = max(14, n_strats * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    for mi, model_name in enumerate(models):
        sub  = df[df["Model"] == model_name]
        vals = [sub.loc[sub["Strategy"] == s, energy_col].values[0]
                if len(sub.loc[sub["Strategy"] == s]) > 0 else 0
                for s in strategies]
        errs = ([sub.loc[sub["Strategy"] == s, std_col].values[0]
                 if len(sub.loc[sub["Strategy"] == s]) > 0 else 0
                 for s in strategies] if std_col else None)

        offset = (mi - n_models / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, width=bar_w, yerr=errs, capsize=2,
                        color=model_colors.get(model_name, "#aaaacc"),
                        edgecolor="none", label=model_name, alpha=0.9)

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.6f}", ha="center", va="bottom",
                    fontsize=6, color="#1a1a2e", rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Energy consumed (kWh)", fontsize=11, labelpad=10)
    ax.set_title(
        "Training Energy Consumption by Model & Reduction Strategy  (MovieLens-1M, mean±std over 5 seeds)",
        fontsize=13, fontweight="bold", color="#1a1a2e", pad=18,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.6f}"))

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[m], label=m) for m in models]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="#f7f7fa", edgecolor="#d8d8e0", labelcolor="#1a1a2e",
        fontsize=9,
    )

    plt.tight_layout()
    out = RESULTS_DIR / "04_training_emissions.svg"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Training energy plot saved -> {out}")


if __name__ == "__main__":
    run()

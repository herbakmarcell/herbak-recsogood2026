"""
03_train_extras.py
==================
Improved / extended reduction strategies - run separately to keep the main
results in training_emissions.csv intact.

Experiments
-----------
1. CS-Leverage Pareto sweep  : -10%, -15%  (find the Recall sweet-spot)
2. CS-Cluster-Outlier         : keep FARTHEST interactions instead of nearest
3. Global-temp + MinK3        : temporal cutoff + per-user min-3 guarantee
4. Baseline / ref strategies  : for apples-to-apples comparison in extras plots

Seeded runs + significance
----------------------------
Mirrors 02_train_models.py: every (strategy, model) is trained/evaluated
across utils.eval_utils.DEFAULT_SEEDS (5 seeds, driving both the reduction's
own randomness and the model's training randomness), so energy AND accuracy
both get a mean +/- std from the same runs, plus a paired Wilcoxon signed-
rank test vs. Baseline per model. See utils/eval_utils.py's module docstring
for what each reported metric means.

Outputs
-------
results/extras_training_emissions.csv
results/11_extras_pareto.svg
results/12_extras_cluster_vs_outlier.svg
results/13_extras_temporal_coverage.svg

Usage
-----
    python src/03_train_extras.py
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

from utils.data_utils import (
    download_ml1m, load_data,
    apply_reduction, build_global_split, build_cornac_dataset,
    EXTRA_STRATEGIES, slugify,
)
from utils.emission_utils import measure_emissions
from utils.eval_utils import (
    DEFAULT_SEEDS, evaluate_model, build_common_item_vocabulary,
    build_common_subset_candidates, paired_wilcoxon_pvalue,
    split_users_by_history_length,
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K          = 64
N_EPOCHS   = 50
LR         = 1e-3
BATCH_SIZE = 256
TOPK       = 10

CANONICAL_SEED = DEFAULT_SEEDS[0]

import torch as _torch
USE_GPU = _torch.cuda.is_available()
print(f"[DEVICE] {'GPU: ' + _torch.cuda.get_device_name(0) if USE_GPU else 'CPU'}")


def make_bpr(seed=CANONICAL_SEED):
    import cornac
    return cornac.models.BPR(
        k=K, max_iter=N_EPOCHS, learning_rate=LR,
        lambda_reg=1e-4, seed=seed, verbose=False,
    )

def make_multidae(seed=CANONICAL_SEED):
    import cornac
    return cornac.models.VAECF(
        k=K, autoencoder_structure=[600], act_fn="tanh",
        likelihood="mult", n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        learning_rate=LR, beta=0.0, seed=seed, use_gpu=USE_GPU, verbose=False,
    )

def make_ease(seed=CANONICAL_SEED):
    """Deterministic closed-form solution - seed has no effect on EASE itself."""
    import cornac
    return cornac.models.EASE(lamb=500, verbose=False)

MODEL_FACTORIES = {
    "BPR":      make_bpr,
    "MultiDAE": make_multidae,
    "EASE":     make_ease,
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

    print("[>>] Building global fixed split ...")
    train_raw_df, global_val_df, global_test_df = build_global_split(ratings)
    n_global_test = len(global_test_df)
    print(f"[OK] {len(train_raw_df):,} train  |  {n_global_test:,} test\n")

    strategy_names = [n for n, _ in EXTRA_STRATEGIES]
    print("[>>] Computing common item vocabulary across extra strategies ...")
    common_items = build_common_item_vocabulary(
        train_raw_df, strategy_names, apply_reduction, seed=CANONICAL_SEED,
    )
    common_subset_candidates = build_common_subset_candidates(
        global_test_df, common_items, n_neg=100,
    )
    print(f"[OK] Common item vocabulary: {len(common_items):,} items\n")

    print("[>>] Splitting users by historical interaction length ...")
    user_groups, group_counts = split_users_by_history_length(train_raw_df, global_test_df)
    print(f"[OK] Test user group sizes: Short={group_counts['short']}, Medium={group_counts['medium']}, Long={group_counts['long']}\n")

    records      = []
    raw_recalls  = {}   # (strategy, model) -> list of per-seed per_user_recalls, for Wilcoxon
    PROGRESS_LOG = RESULTS_DIR / "extras_progress.log"
    total_jobs   = len(EXTRA_STRATEGIES) * len(MODEL_FACTORIES) * len(DEFAULT_SEEDS)
    job_idx      = 0

    PROGRESS_LOG.write_text(
        f"Extras run started: {datetime.now().strftime('%H:%M:%S')}\n"
        f"Total jobs: {total_jobs}  "
        f"({len(EXTRA_STRATEGIES)} strategies x {len(MODEL_FACTORIES)} models x {len(DEFAULT_SEEDS)} seeds)\n"
        + "=" * 60 + "\n",
        encoding="utf-8",
    )

    for strategy_name, reduce_fn in EXTRA_STRATEGIES:
        print(f"\n{'='*72}")
        print(f"  Strategy: {strategy_name}")
        print(f"{'='*72}")

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
                output_file="extras_preprocess_codecarbon_raw.csv",
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

                trained_model, energy_kwh, duration_s = measure_emissions(
                    f"{model_name} | {strategy_name} | seed={seed}",
                    _train_seeded, factory, train_set, seed,
                    results_dir=RESULTS_DIR,
                    output_file="extras_training_codecarbon_raw.csv",
                )

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
                d["coverage"].append(scores["Coverage"])
                d["recall_common_subset"].append(scores.get(f"Recall@{TOPK}_common_subset", float("nan")))
                d["recall_short"].append(scores.get(f"Recall@{TOPK}_short", float("nan")))
                d["recall_medium"].append(scores.get(f"Recall@{TOPK}_medium", float("nan")))
                d["recall_long"].append(scores.get(f"Recall@{TOPK}_long", float("nan")))
                d["per_user_recalls"].append(scores["per_user_recalls"])

                print(
                    f"     Recall@{TOPK}={scores[f'Recall@{TOPK}']:.4f}  "
                    f"Coverage={scores['Coverage']:.3f}  Energy={energy_kwh:.8f} kWh"
                )

        for model_name, d in per_model.items():
            rec = {
                "Strategy":                        strategy_name,
                "Model":                            model_name,
                "N_seeds":                          len(DEFAULT_SEEDS),
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
                pf.write(
                    f"{datetime.now().strftime('%H:%M:%S')}  "
                    f"{model_name:<10} | {strategy_name:<35} | "
                    f"Recall@{TOPK}={np.mean(d['recall']):.4f}+-{np.std(d['recall']):.4f}  "
                    f"Coverage={np.mean(d['coverage']):.3f}\n"
                )

    def get_avg_user_scores(scores_list, subset_uids):
        avg_scores = []
        for uid in subset_uids:
            s = np.mean([d.get(uid, 0.0) for d in scores_list])
            avg_scores.append(s)
        return avg_scores

    all_uids = global_test_df["user_id"].unique()
    short_uids = [u for u, g in user_groups.items() if g == "short"]
    medium_uids = [u for u, g in user_groups.items() if g == "medium"]
    long_uids = [u for u, g in user_groups.items() if g == "long"]

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
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_short"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_medium"] = float("nan")
            rec[f"Recall@{TOPK}_wilcoxon_p_vs_baseline_long"] = float("nan")

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "extras_training_emissions_wilcoxon.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Extras CSV saved -> {csv_path}")
    print("[DONE] All extra jobs complete.\n")


if __name__ == "__main__":
    run()

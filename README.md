# herbak-recsogood2026

Do reduced training datasets make recommender systems greener without wrecking recommendation quality? This project measures the **energy/carbon cost vs. recommendation quality** tradeoff of several dataset-reduction strategies on MovieLens-1M, using **CodeCarbon** for energy measurement and **Cornac** (primary) / **RecBole** (cross-check) as training frameworks.

## Approach

1. Load MovieLens-1M and build one fixed, leave-one-out train/val/test split up front, so every strategy and model is evaluated on exactly the same test users.
2. Apply a reduction strategy to the training portion only (never to val/test), producing a smaller training set.
3. Train BPR, MultiDAE, and EASE on each reduced training set.
4. Track energy consumption (kWh) for data loading, reduction, and training with CodeCarbon, following the warm-up + repeat measurement protocol from Schödl et al. 2025 ("Investigating Carbon Footprint of Recommender Systems Beyond Training Time", RecSys '25). Raw kWh is reported instead of a CO2-equivalent figure, since a fixed carbon-intensity factor adds no information and can be applied later by the reader.
5. Evaluate Recall@10 / NDCG@10 / catalog coverage on the fixed test set, repeated across 5 seeds so both energy and accuracy get a mean ± std from the same runs, plus a paired Wilcoxon signed-rank test against the baseline per model.

### Reduction strategies

| Strategy | Idea |
|---|---|
| User-based / Item-based (-20%) | Randomly drop a fraction of ratings, stratified per user / per item |
| User-temporal (last N) | Keep only each user's N most recent interactions |
| Global-temporal (cutoff date) | Drop all interactions before a fixed date |
| Coreset-Leverage | Sample interactions with probability proportional to their SVD leverage score |
| Coreset-Cluster | Cluster interactions in SVD space and keep those nearest each cluster centroid |
| CS-Cluster-Outlier (extra) | Same as Coreset-Cluster but keeps the farthest points from each centroid |
| Global-temp+MinK3 (extra) | Global-temporal cutoff with a per-user minimum-interactions guarantee, to fix coverage loss |

## Setup

```bash
pip install -r requirements.txt
```

MovieLens-1M is downloaded automatically on first run into `src/data/ml-1m/`.

## Pipeline

Run from the repository root, in order:

| Script | Purpose |
|---|---|
| `src/01_dataset_overview.py` | Downloads ML-1M, plots genre distribution and dataset-size/energy impact of each reduction strategy |
| `src/02_train_models.py` | Trains BPR / MultiDAE / EASE on the baseline + main reduction strategies (Cornac), across 5 seeds; saves models and `results/training_emissions_wilcoxon.csv` |
| `src/03_train_extras.py` | Same as above for the extra/improved strategies (Pareto sweep, cluster-outlier, temporal+min-K); saves `results/extras_training_emissions_wilcoxon.csv` |
| `src/04_recbole_export.py` | Exports every reduced dataset variant to RecBole's `.inter` format |
| `src/05_recbole_train.py` | Re-trains BPR / MultiDAE / EASE with RecBole for a cross-framework check; saves `results/recbole_emissions.csv` |
| `src/06_recbole_preprocessing.py` | Compares Cornac vs. RecBole dataset-building overhead per strategy |
| `src/07_visualize_results.py` | Energy vs. Recall scatter, per-strategy bar charts, green tradeoff, efficiency frontier |
| `src/08_visualize_extras.py` | Same comparisons for the extras sweep (Pareto, cluster-outlier, temporal+min-K) |
| `src/09_plot_inference_time.py` | Inference latency by model/strategy |
| `src/10_plot_energy_breakdown.py` | Per-strategy energy composition (load / reduction / training) and stacked pipeline cost |
| `src/11_recbole_analysis.py` | Statistical analysis and Cornac-vs-RecBole comparison plot |
| `src/12_plot_all_methods.py` | Combined overview of every (strategy, model) result from both sweeps |

`src/utils/` holds the shared building blocks used across scripts: data loading and reduction strategies (`data_utils.py`), the CodeCarbon measurement wrapper (`emission_utils.py`), evaluation metrics and significance testing (`eval_utils.py`), the shared plot theme (`plot_style.py`), CSV loading for plots (`results_io.py`), and pure-PyTorch LightGCN/NGCF/ELSA model wrappers (`graph_models.py`).

Outputs are written to `results/` (CSVs and SVG plots) and `models/` (saved Cornac models, canonical seed only).

## Project status

This repository is the companion code for the RecSys 2026 Workshop paper *"Energy Footprint of Data Reduction Strategies for Recommender Systems"* (Herbák, Lesota, Tommasel). The pipeline has continued to evolve since the paper's results were produced, and now includes work the paper lists as future directions:

- **Multi-seed evaluation is now the default methodology**, not a supplementary robustness check. Previously only a single seed was used for the main results, with 5-seed variance checked separately. Now every (strategy, model) is trained/evaluated across 5 seeds (`utils/eval_utils.DEFAULT_SEEDS`), with mean ± std and a paired Wilcoxon signed-rank test vs. Baseline reported for every metric.
- **A shared cross-strategy negative-sampling pool** (`Recall@10_common_subset` in `utils/eval_utils.py`) now lets Recall/NDCG be compared on an identical candidate-item pool across strategies, instead of each strategy sampling negatives from its own (differently sized) item vocabulary.
- **A RecBole cross-framework comparison** (`04_recbole_export.py`–`06_recbole_preprocessing.py`, `11_recbole_analysis.py`) has been added, re-running the same models/strategies in RecBole and comparing against the Cornac results.
- **A short/medium/long user-history breakdown** (`split_users_by_history_length`) was added to check whether reduction strategies affect light and heavy users differently.

As a result, the current code, its default outputs, and some of the reported CSV columns no longer map one-to-one onto the tables and figures in the paper: treat the paper as the methodology snapshot at submission time, and this repository as the actively developed superset.

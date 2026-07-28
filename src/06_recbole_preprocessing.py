import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.plot_style import apply_style, configure_stdout
configure_stdout()

from codecarbon import EmissionsTracker

import cornac
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import logging
logging.getLogger('recbole').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
RECBOLE_DATA_DIR = ROOT / "src" / "data" / "recbole"
RESULTS_DIR = ROOT / "results"
OUT_FILE = RESULTS_DIR / "preprocessing_emissions.csv"

DATASETS = [
    "baseline",
    "user_based_20",
    "item_based_20",
    "user_temporal_last_300",
    "global_temporal_sep_2000",
    "coreset_leverage_20",
    "coreset_leverage_30",
    "coreset_cluster_20",
    "coreset_cluster_30",
]

ITERATIONS = 3

def measure_cornac_prep(dataset_name: str, iter_idx: int):
    # Load the exported .inter file directly so this measures pure framework
    # overhead, not the reduction logic itself.
    inter_file = RECBOLE_DATA_DIR / dataset_name / f"{dataset_name}.inter"
    df = pd.read_csv(inter_file, sep='\t')
    df.columns = [c.split(':')[0] for c in df.columns]

    uir_tuples = list(df[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))

    tracker = EmissionsTracker(project_name=f"Cornac_{dataset_name}_{iter_idx}", measure_power_secs=0.5, log_level="error")
    tracker.start()
    start_t = time.time()

    dataset = cornac.data.Dataset.build(uir_tuples, global_uid_map={}, global_iid_map={}, seed=42+iter_idx)

    duration = time.time() - start_t
    emissions = tracker.stop()

    return {
        'Framework': 'Cornac',
        'Strategy': dataset_name,
        'Iteration': iter_idx,
        'CO2_g': emissions * 1000,
        'Duration_s': duration,
    }

def measure_recbole_prep(dataset_name: str, iter_idx: int):
    config_dict = {
        'model': 'BPR',  # dummy model, only needed to init the config
        'dataset': dataset_name,
        'data_path': str(RECBOLE_DATA_DIR),
        'eval_args': {'split': {'LS': 'valid_and_test'}, 'order': 'TO', 'mode': 'uni100'},
        'metrics': ['Recall', 'NDCG'],
        'topk': [10],
        'seed': 42 + iter_idx,
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'user_inter_num_interval': '[0,inf)',
        'item_inter_num_interval': '[0,inf)',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'device': 'cpu',  # CPU-only, so the Cornac/RecBole comparison is apples-to-apples
        'show_progress': False,
    }

    tracker = EmissionsTracker(project_name=f"RecBole_{dataset_name}_{iter_idx}", measure_power_secs=0.5, log_level="error")
    tracker.start()
    start_t = time.time()

    config = Config(model='BPR', dataset=dataset_name, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    duration = time.time() - start_t
    emissions = tracker.stop()

    return {
        'Framework': 'RecBole',
        'Strategy': dataset_name,
        'Iteration': iter_idx,
        'CO2_g': emissions * 1000,
        'Duration_s': duration,
    }

def main():
    results = []

    for i in range(ITERATIONS):
        for ds in DATASETS:
            print(f"[{i+1}/{ITERATIONS}] Measuring {ds}...")

            try:
                res_rec = measure_recbole_prep(ds, i)
                results.append(res_rec)
            except Exception as e:
                print(f"RecBole failed on {ds}: {e}")

            try:
                res_cor = measure_cornac_prep(ds, i)
                results.append(res_cor)
            except Exception as e:
                print(f"Cornac failed on {ds}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUT_FILE, index=False)

    agg_df = df.groupby(['Framework', 'Strategy']).agg(
        CO2_mean=('CO2_g', lambda x: x.mean() / 475.0),
        CO2_std=('CO2_g', lambda x: x.std() / 475.0),
        Duration_mean=('Duration_s', 'mean'),
        Duration_std=('Duration_s', 'std')
    ).reset_index()

    print("\n--- Preprocessing Emissions Comparison ---")
    print(agg_df.to_markdown(index=False, floatfmt=".4f"))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Data Preprocessing Overhead: Cornac vs RecBole", fontsize=14, fontweight='bold')

    strategies = [d for d in DATASETS if d in agg_df['Strategy'].values]
    x = np.arange(len(strategies))
    width = 0.35

    cornac_co2 = [agg_df[(agg_df['Framework']=='Cornac') & (agg_df['Strategy']==s)]['CO2_mean'].values[0] for s in strategies]
    recbole_co2 = [agg_df[(agg_df['Framework']=='RecBole') & (agg_df['Strategy']==s)]['CO2_mean'].values[0] for s in strategies]

    cornac_dur = [agg_df[(agg_df['Framework']=='Cornac') & (agg_df['Strategy']==s)]['Duration_mean'].values[0] for s in strategies]
    recbole_dur = [agg_df[(agg_df['Framework']=='RecBole') & (agg_df['Strategy']==s)]['Duration_mean'].values[0] for s in strategies]

    axes[0].bar(x - width/2, cornac_co2, width, label='Cornac', color='#4C72B0')
    axes[0].bar(x + width/2, recbole_co2, width, label='RecBole', color='#DD8452')
    axes[0].set_ylabel('Energy (kWh)')
    axes[0].set_title('Preprocessing Emissions')
    axes[0].set_xticks(x)

    LABEL_MAP = {
        "baseline": "Baseline",
        "user_based_20": "User-based (-20%)",
        "item_based_20": "Item-based (-20%)",
        "user_temporal_last_300": "User-temporal (last 300)",
        "global_temporal_sep_2000": "Global-temporal (Sep 2000)",
        "coreset_leverage_20": "Coreset-Leverage (-20%)",
        "coreset_leverage_30": "Coreset-Leverage (-30%)",
        "coreset_cluster_20": "Coreset-Cluster (-20%)",
        "coreset_cluster_30": "Coreset-Cluster (-30%)"
    }

    axes[0].set_xticklabels([LABEL_MAP.get(s, s) for s in strategies], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x - width/2, cornac_dur, width, label='Cornac', color='#4C72B0')
    axes[1].bar(x + width/2, recbole_dur, width, label='RecBole', color='#DD8452')
    axes[1].set_ylabel('Time (s)')
    axes[1].set_title('Preprocessing Duration')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABEL_MAP.get(s, s) for s in strategies], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "16_preprocessing_comparison.svg"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()

import os
import sys
import time
import pandas as pd
from pathlib import Path
from logging import getLogger
from codecarbon import EmissionsTracker

import torch

# PyTorch 2.6+ defaults torch.load(weights_only=True), which breaks loading
# RecBole's own checkpoint objects - force the old behaviour back on.
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, MultiDAE, EASE
from recbole.trainer import Trainer

import logging
logging.getLogger('recbole').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
RECBOLE_DATA_DIR = ROOT / "src" / "data" / "recbole"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / "recbole_emissions.csv"

MODELS = ["BPR", "MultiDAE", "EASE"]
SEEDS = [42, 43, 44]

def run_recbole(model_name: str, dataset_name: str, seed: int):
    config_dict = {
        'model': model_name,
        'dataset': dataset_name,
        'data_path': str(RECBOLE_DATA_DIR),

        # Chronological leave-one-out, matching the Cornac global split
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'mode': 'uni100'
        },
        'metrics': ['Recall', 'NDCG'],
        'topk': [10],

        'seed': seed,
        'epochs': 50,
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'learning_rate': 1e-3,

        'reg_weight': 500.0,   # EASE-specific

        # Do not filter out any items or users
        'user_inter_num_interval': '[0,inf)',
        'item_inter_num_interval': '[0,inf)',

        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},

        # Disable early stopping by validating only after the max epoch
        'eval_step': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'show_progress': False,
    }

    config = Config(model=model_name, dataset=dataset_name, config_dict=config_dict)

    try:
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
    except Exception as e:
        print(f"Skipping {dataset_name} due to dataset error: {e}")
        return None

    if model_name == 'BPR':
        model = BPR(config, train_data.dataset).to(config['device'])
    elif model_name == 'MultiDAE':
        model = MultiDAE(config, train_data.dataset).to(config['device'])
    elif model_name == 'EASE':
        model = EASE(config, train_data.dataset).to(config['device'])

    trainer = Trainer(config, model)

    tracker = EmissionsTracker(
        project_name=f"{dataset_name}_{model_name}_{seed}",
        measure_power_secs=1.0,
        log_level="error"
    )
    tracker.start()

    start_time = time.time()
    trainer.fit(train_data, show_progress=False)
    test_result = trainer.evaluate(test_data, show_progress=False)
    duration = time.time() - start_time

    emissions = tracker.stop()

    return {
        'Strategy': dataset_name,
        'Model': model_name,
        'Seed': seed,
        'CO2_g': emissions * 1000,
        'Duration_s': duration,
        'Recall@10': test_result.get('recall@10', 0),
        'NDCG@10': test_result.get('ndcg@10', 0)
    }

def main():
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    datasets = [d.name for d in RECBOLE_DATA_DIR.iterdir() if d.is_dir()]

    results = []

    if OUT_FILE.exists():
        existing_df = pd.read_csv(OUT_FILE)
        results = existing_df.to_dict('records')
        print(f"Loaded {len(results)} existing results from {OUT_FILE}")

    def already_run(ds, md, s):
        for r in results:
            if r['Strategy'] == ds and r['Model'] == md and r['Seed'] == s:
                return True
        return False

    total_runs = len(datasets) * len(MODELS) * len(SEEDS)
    current_run = 0

    for dataset_name in datasets:
        for model_name in MODELS:
            for seed in SEEDS:
                current_run += 1
                if already_run(dataset_name, model_name, seed):
                    print(f"[{current_run}/{total_runs}] Skipping {dataset_name} | {model_name} | Seed {seed} (Already run)")
                    continue

                print(f"[{current_run}/{total_runs}] Running {dataset_name} | {model_name} | Seed {seed}...")
                res = run_recbole(model_name, dataset_name, seed)
                if res is not None:
                    results.append(res)
                    pd.DataFrame(results).to_csv(OUT_FILE, index=False)

if __name__ == "__main__":
    main()

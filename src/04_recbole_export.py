import os
import re
import pandas as pd
from pathlib import Path

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent))
from utils.data_utils import load_data, build_global_split, apply_reduction, REDUCTION_STRATEGIES

ROOT = Path(__file__).resolve().parent.parent
RECBOLE_DATA_DIR = ROOT / "src" / "data" / "recbole"
RECBOLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')

def export_to_inter(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path, dataset_name: str):
    # val and test are the most-recent interaction per user, so RecBole's 'LS'
    # (leave-one-out, time-ordered) split peels them off exactly, leaving
    # train_df as the training set.
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    recbole_df = pd.DataFrame({
        "user_id:token": full_df["user_id"],
        "item_id:token": full_df["movie_id"],
        "rating:float": full_df["rating"],
        "timestamp:float": full_df["timestamp"].astype('int64') // 10**9,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}.inter"
    recbole_df.to_csv(out_file, sep="\t", index=False)
    print(f"Exported {dataset_name} to {out_file} (Total interactions: {len(recbole_df)})")

def main():
    print("Loading raw data...")
    movies, ratings = load_data()
    train_pool, val_df, test_df = build_global_split(ratings)

    for strategy_name, _ in REDUCTION_STRATEGIES:
        clean_name = sanitize_name(strategy_name)
        print(f"\nProcessing {strategy_name} -> {clean_name}")

        train_reduced = apply_reduction(train_pool, strategy_name, seed=42)

        dataset_dir = RECBOLE_DATA_DIR / clean_name
        export_to_inter(train_reduced, val_df, test_df, dataset_dir, clean_name)

if __name__ == "__main__":
    main()

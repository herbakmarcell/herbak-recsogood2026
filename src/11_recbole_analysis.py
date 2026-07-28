import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

def load_and_aggregate_recbole():
    df = pd.read_csv(RESULTS_DIR / "recbole_emissions.csv")

    agg_df = df.groupby(['Strategy', 'Model']).agg(
        Recall_mean=('Recall@10', 'mean'),
        Recall_std=('Recall@10', 'std'),
        NDCG_mean=('NDCG@10', 'mean'),
        NDCG_std=('NDCG@10', 'std'),
        CO2_mean=('CO2_g', 'mean'),
        Duration_mean=('Duration_s', 'mean'),
        Count=('Seed', 'count')
    ).reset_index()

    baseline = df[df['Strategy'] == 'baseline']

    p_values = []
    for _, row in agg_df.iterrows():
        strat = row['Strategy']
        mod = row['Model']
        if strat == 'baseline':
            p_values.append(1.0)
            continue

        strat_runs = df[(df['Strategy'] == strat) & (df['Model'] == mod)]['Recall@10']
        base_runs = baseline[baseline['Model'] == mod]['Recall@10']

        if len(strat_runs) > 1 and len(base_runs) > 1:
            _, p = stats.ttest_ind(strat_runs, base_runs, equal_var=False)
            p_values.append(p)
        else:
            p_values.append(np.nan)

    agg_df['P_Value_vs_Baseline'] = p_values
    return agg_df

def plot_framework_comparison(recbole_agg):
    cornac_df = pd.read_csv(RESULTS_DIR / "training_emissions.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Framework Comparison: Cornac vs RecBole (Recall@10 vs Energy)", fontsize=14, fontweight='bold')

    models = ['BPR', 'MultiDAE', 'EASE']
    colors = {'BPR': '#4C72B0', 'MultiDAE': '#DD8452', 'EASE': '#55A868'}

    for ax, model in zip(axes, models):
        c_model = cornac_df[cornac_df['Model'] == model]
        if len(c_model) > 0:
            c_base = c_model[c_model['Strategy'] == 'Baseline']
            c_rest = c_model[c_model['Strategy'] != 'Baseline']

            ax.scatter(c_rest['CO2_g'] / 475.0, c_rest['Recall@10'], marker='o', alpha=0.6,
                       color=colors[model], label='Cornac (Reduction)')
            if len(c_base) > 0:
                ax.scatter(c_base['CO2_g'] / 475.0, c_base['Recall@10'], marker='*', s=150,
                           color='gold', edgecolor='black', label='Cornac (Baseline)')

        r_model = recbole_agg[recbole_agg['Model'] == model]
        if len(r_model) > 0:
            r_base = r_model[r_model['Strategy'] == 'baseline']
            r_rest = r_model[r_model['Strategy'] != 'baseline']

            ax.scatter(r_rest['CO2_mean'] / 475.0, r_rest['Recall_mean'], marker='s', alpha=0.8,
                       color=colors[model], edgecolor='black', label='RecBole (Reduction)')
            if len(r_base) > 0:
                ax.scatter(r_base['CO2_mean'] / 475.0, r_base['Recall_mean'], marker='*', s=150,
                           color='red', edgecolor='black', label='RecBole (Baseline)')

        ax.set_title(model)
        ax.set_xlabel("Training Energy (kWh)")
        ax.set_ylabel("Recall@10")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "recbole_vs_cornac_pareto.svg"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

def main():
    if not (RESULTS_DIR / "recbole_emissions.csv").exists():
        print("recbole_emissions.csv not found! Run 05_recbole_train.py first.")
        return

    agg_df = load_and_aggregate_recbole()

    print("\n--- RecBole Statistical Analysis ---")
    print(agg_df.to_markdown(index=False, floatfmt=".4f"))

    agg_df.to_csv(RESULTS_DIR / "recbole_statistical_analysis.csv", index=False)

    plot_framework_comparison(agg_df)

if __name__ == "__main__":
    main()

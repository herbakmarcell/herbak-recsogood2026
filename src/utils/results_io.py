"""
utils/results_io.py
===================
Single source of truth for loading the training-results CSVs into the shape the
visualisation scripts expect.

Every plotting script used to repeat the same block: strip column whitespace,
alias ``Recall@10`` -> ``Recall10``, sniff whether the CSV is in the current
kWh columns or the older CO2-gram columns, build a ``Total_Energy`` column
(preprocessing + training), and backfill ``Coverage`` for legacy CSVs. That
logic now lives here so it can't drift between scripts.
"""

from pathlib import Path

import pandas as pd

# kWh (current) vs CO2-gram (legacy) column pairs, and the axis-label unit.
_ENERGY_KWH, _ENERGY_CO2 = "Energy_kWh_mean", "CO2_g"
_PREPROC_KWH, _PREPROC_CO2 = "Preprocess_kWh_mean", "Preprocess_CO2_g"
UNIT_KWH, UNIT_CO2 = "kWh", "g CO₂-eq"


def load_training_results(path, extras_path=None):
    """
    Load a training-emissions CSV (optionally appending an extras CSV) and
    return ``(df, energy_unit)``.

    The returned frame always has ``Recall10`` (and ``NDCG10`` when present),
    a ``Total_Energy`` column (preprocessing + training energy), and a
    ``Coverage`` column (backfilled to 1.0 for legacy CSVs that predate it).
    ``energy_unit`` is ``"kWh"`` for current CSVs, ``"g CO₂-eq"`` for legacy ones.
    """
    df = pd.read_csv(path)
    if extras_path is not None and Path(extras_path).exists():
        df = pd.concat([df, pd.read_csv(extras_path)], ignore_index=True)

    df.columns = df.columns.str.strip()
    if "Recall@10" in df.columns:
        df["Recall10"] = df["Recall@10"]
    if "NDCG@10" in df.columns:
        df["NDCG10"] = df["NDCG@10"]

    energy_col     = _ENERGY_KWH  if _ENERGY_KWH  in df.columns else _ENERGY_CO2
    preprocess_col = _PREPROC_KWH if _PREPROC_KWH in df.columns else _PREPROC_CO2
    unit           = UNIT_KWH if energy_col == _ENERGY_KWH else UNIT_CO2

    prep = df[preprocess_col] if preprocess_col in df.columns else 0
    df["Total_Energy"] = prep + df[energy_col]

    if "Coverage" not in df.columns:
        df["Coverage"] = 1.0

    return df, unit

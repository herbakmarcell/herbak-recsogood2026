"""
utils/data_utils.py
===================
Shared data-loading, reduction strategies, and cornac dataset builder.
All reduction logic is kept here so both 01_dataset_overview.py and
02_train_models.py reference the same single source of truth.
"""

import io
import zipfile
import urllib.request
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "src" / "data" / "ml-1m"
MODELS_DIR = ROOT / "models"

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_ml1m() -> None:
    movies_path  = DATA_DIR / "movies.dat"
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


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (movies_df, ratings_df) from the ml-1m files."""
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


def user_based_reduction(df: pd.DataFrame, k: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Remove k% of ratings randomly per user (stratified)."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in df.groupby("user_id"):
        n_remove = int(len(group) * k)
        remove_idx = rng.choice(group.index, size=n_remove, replace=False)
        keep.extend(set(group.index) - set(remove_idx))
    return df.loc[keep].reset_index(drop=True)


def item_based_reduction(df: pd.DataFrame, k: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Remove k% of ratings randomly per item (stratified)."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in df.groupby("movie_id"):
        n_remove = int(len(group) * k)
        remove_idx = rng.choice(group.index, size=n_remove, replace=False)
        keep.extend(set(group.index) - set(remove_idx))
    return df.loc[keep].reset_index(drop=True)


def user_temporal_reduction(df: pd.DataFrame, n: int = 250) -> pd.DataFrame:
    """For each user keep only their last n interactions."""
    return (
        df.sort_values("timestamp")
        .groupby("user_id")
        .tail(n)
        .reset_index(drop=True)
    )


def global_temporal_reduction(df: pd.DataFrame, cutoff: str = "2000-08-01") -> pd.DataFrame:
    """Keep only ratings recorded on or after *cutoff* date."""
    cutoff_dt = pd.to_datetime(cutoff)
    return df[df["timestamp"] >= cutoff_dt].reset_index(drop=True)


def _build_interaction_matrix(df: pd.DataFrame):
    """Return (sparse_csr_matrix, user_row_idx, item_col_idx) for the raw df."""
    from scipy.sparse import csr_matrix
    users = df["user_id"].values
    items = df["movie_id"].values
    user_ids, user_inv = np.unique(users, return_inverse=True)
    item_ids, item_inv = np.unique(items, return_inverse=True)
    R = csr_matrix(
        (np.ones(len(df)), (user_inv, item_inv)),
        shape=(len(user_ids), len(item_ids)),
    )
    return R, user_inv, item_inv


def _safe_keep(df: pd.DataFrame, keep_mask: np.ndarray, min_per_user: int = 3) -> pd.DataFrame:
    """
    Given a boolean mask of interactions to keep, additionally guarantee every
    user retains at least *min_per_user* interactions so LOO eval is valid.
    """
    selected = set(np.where(keep_mask)[0])
    for uid, group in df.groupby("user_id"):
        idx = group.index.tolist()
        in_sel = [i for i in idx if i in selected]
        if len(in_sel) < min_per_user:
            extra = [i for i in idx if i not in selected]
            for i in extra[: min_per_user - len(in_sel)]:
                selected.add(i)
    return df.loc[sorted(selected)].reset_index(drop=True)


def coreset_leverage_reduction(
    df: pd.DataFrame,
    fraction: float = 0.80,
    svd_rank: int   = 50,
    seed: int       = 42,
) -> pd.DataFrame:
    """
    Leverage-score coreset.

    Each interaction (u, i) is assigned an importance score proportional to
    the squared row-norm of u in the left singular vectors plus the squared
    column-norm of i in the right singular vectors of the top-k SVD of R.
    Interactions are then sampled WITHOUT replacement with probabilities
    proportional to those scores.

    Theoretical guarantee: the sampled sub-matrix approximates the spectral
    norm of R up to a (1+ε) factor with high probability.
    """
    from scipy.sparse.linalg import svds

    R, user_inv, item_inv = _build_interaction_matrix(df)
    k = min(svd_rank, min(R.shape) - 1)
    U, _, Vt = svds(R.astype(np.float32), k=k)

    user_lev = (U ** 2).sum(axis=1)    # (n_users,)
    item_lev = (Vt ** 2).sum(axis=0)   # (n_items,)

    scores = user_lev[user_inv] + item_lev[item_inv]   # (n_interactions,)
    probs  = scores / scores.sum()

    n_keep = int(len(df) * fraction)
    rng    = np.random.default_rng(seed)
    chosen = rng.choice(len(df), size=n_keep, replace=False, p=probs)

    mask = np.zeros(len(df), dtype=bool)
    mask[chosen] = True
    return _safe_keep(df, mask)


def coreset_cluster_reduction(
    df: pd.DataFrame,
    fraction: float = 0.80,
    svd_rank: int   = 50,
    n_clusters: int = 500,
    seed: int       = 42,
) -> pd.DataFrame:
    """
    Cluster-based coreset (k-center approximation).

    Each interaction is embedded as f(u,i) = U[u,:]*Σ + Vt[:,i]*Σ in the
    SVD latent space.  MiniBatchKMeans partitions the interaction space into
    *n_clusters* clusters.  We then select the n_keep interactions nearest
    to their respective cluster centres, maximising geometric diversity.

    This approximates the k-center objective (minimise the maximum distance
    from any unselected point to the coreset) without the O(n²) cost of the
    exact greedy algorithm.
    """
    from scipy.sparse.linalg import svds
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin

    R, user_inv, item_inv = _build_interaction_matrix(df)
    k = min(svd_rank, min(R.shape) - 1)
    U, sigma, Vt = svds(R.astype(np.float32), k=k)

    user_emb = U[user_inv] * sigma      # (n_interactions, k)
    item_emb = (Vt.T)[item_inv] * sigma # (n_interactions, k)
    features = user_emb + item_emb      # additive combination

    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed,
        n_init=5, batch_size=10_000, max_iter=100,
    )
    km.fit(features)

    labels    = km.labels_
    centers   = km.cluster_centers_
    n_keep    = int(len(df) * fraction)

    # Proportional allocation: each cluster contributes proportionally to size
    unique, counts = np.unique(labels, return_counts=True)
    alloc = np.maximum(1, np.round(counts / counts.sum() * n_keep).astype(int))

    chosen = []
    for c, n_c in zip(unique, alloc):
        idxs = np.where(labels == c)[0]
        dists = np.linalg.norm(features[idxs] - centers[c], axis=1)
        top = idxs[np.argsort(dists)[:n_c]]
        chosen.extend(top.tolist())

    chosen = np.array(chosen[:n_keep])
    mask = np.zeros(len(df), dtype=bool)
    mask[chosen] = True
    return _safe_keep(df, mask)


def coreset_cluster_outlier_reduction(
    df: pd.DataFrame,
    fraction: float = 0.80,
    svd_rank: int   = 50,
    n_clusters: int = 500,
    seed: int       = 42,
) -> pd.DataFrame:
    """
    Cluster-based coreset keeping OUTLIERS (interactions FARTHEST from centers).

    Unlike coreset_cluster_reduction which keeps the interactions nearest to
    cluster centres (the "average"), this variant keeps the most unusual
    interactions.  Theory: outlier interactions carry more unique signal per
    rating and act as a natural noise filter for recommendation models.

    Same SVD embedding + K-Means setup, but the sort direction is reversed.
    """
    from scipy.sparse.linalg import svds
    from sklearn.cluster import MiniBatchKMeans

    R, user_inv, item_inv = _build_interaction_matrix(df)
    k = min(svd_rank, min(R.shape) - 1)
    U, sigma, Vt = svds(R.astype(np.float32), k=k)

    user_emb = U[user_inv] * sigma
    item_emb = (Vt.T)[item_inv] * sigma
    features = user_emb + item_emb

    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed,
        n_init=5, batch_size=10_000, max_iter=100,
    )
    km.fit(features)

    labels  = km.labels_
    centers = km.cluster_centers_
    n_keep  = int(len(df) * fraction)

    unique, counts = np.unique(labels, return_counts=True)
    alloc = np.maximum(1, np.round(counts / counts.sum() * n_keep).astype(int))

    chosen = []
    for c, n_c in zip(unique, alloc):
        idxs  = np.where(labels == c)[0]
        dists = np.linalg.norm(features[idxs] - centers[c], axis=1)
        top   = idxs[np.argsort(dists)[::-1][:n_c]]   # farthest, not nearest
        chosen.extend(top.tolist())

    chosen = np.array(chosen[:n_keep])
    mask   = np.zeros(len(df), dtype=bool)
    mask[chosen] = True
    return _safe_keep(df, mask)


def global_temporal_with_mink_reduction(
    df: pd.DataFrame,
    cutoff: str = "2000-09-01",
    min_k: int  = 3,
) -> pd.DataFrame:
    """
    Global temporal cutoff with a per-user minimum interaction guarantee.

    The pure global-temporal strategy drops all pre-cutoff data, leaving
    36% of users with zero training interactions (coverage = 64%).

    This variant patches the gap: users with fewer than *min_k* post-cutoff
    interactions get their most-recent pre-cutoff interactions added back
    until they reach min_k.  Result: near-100% coverage while still strongly
    preferring recent data.
    """
    cutoff_dt = pd.to_datetime(cutoff)

    post  = df[df["timestamp"] >= cutoff_dt].copy()
    pre   = df[df["timestamp"] <  cutoff_dt].copy()

    post_counts = post.groupby("user_id").size()
    all_users   = df["user_id"].unique()

    rows_to_add = []
    for uid in all_users:
        n_post = post_counts.get(uid, 0)
        if n_post < min_k:
            needed = min_k - n_post
            user_pre = pre[pre["user_id"] == uid].sort_values("timestamp", ascending=False)
            rows_to_add.append(user_pre.head(needed))

    if rows_to_add:
        extra = pd.concat(rows_to_add, ignore_index=True)
        result = pd.concat([post, extra], ignore_index=True)
    else:
        result = post

    return result.reset_index(drop=True)


# Ordered list of (strategy_name, callable(df, seed=...) -> reduced_df).
# Every entry accepts a `seed` kwarg for a uniform call signature across
# seeded runs (see eval_utils.DEFAULT_SEEDS); strategies with no randomness
# of their own (temporal cutoffs, baseline copy) simply ignore it.
REDUCTION_STRATEGIES = [
    ("Baseline",                     lambda df, seed=42: df.copy()),
    ("User-based (-20%)",            lambda df, seed=42: user_based_reduction(df, k=0.20, seed=seed)),
    ("Item-based (-20%)",            lambda df, seed=42: item_based_reduction(df, k=0.20, seed=seed)),
    ("User-temporal (last 300)",     lambda df, seed=42: user_temporal_reduction(df, n=300)),
    ("User-temporal (last 250)",     lambda df, seed=42: user_temporal_reduction(df, n=250)),
    ("User-temporal (last 200)",     lambda df, seed=42: user_temporal_reduction(df, n=200)),
    ("Global-temporal (Sep 2000)",   lambda df, seed=42: global_temporal_reduction(df, "2000-09-01")),
    ("Global-temporal (Aug 2000)",   lambda df, seed=42: global_temporal_reduction(df, "2000-08-01")),
    ("Global-temporal (Jul 2000)",   lambda df, seed=42: global_temporal_reduction(df, "2000-07-01")),
    ("Coreset-Leverage (-20%)",      lambda df, seed=42: coreset_leverage_reduction(df, fraction=0.80, seed=seed)),
    ("Coreset-Leverage (-30%)",      lambda df, seed=42: coreset_leverage_reduction(df, fraction=0.70, seed=seed)),
    ("Coreset-Cluster (-20%)",       lambda df, seed=42: coreset_cluster_reduction(df, fraction=0.80, seed=seed)),
    ("Coreset-Cluster (-30%)",       lambda df, seed=42: coreset_cluster_reduction(df, fraction=0.70, seed=seed)),
]

EXTRA_STRATEGIES = [
    ("CS-Leverage (-10%)",           lambda df, seed=42: coreset_leverage_reduction(df, fraction=0.90, seed=seed)),
    ("CS-Leverage (-15%)",           lambda df, seed=42: coreset_leverage_reduction(df, fraction=0.85, seed=seed)),
    ("CS-Cluster-Outlier (-20%)",    lambda df, seed=42: coreset_cluster_outlier_reduction(df, fraction=0.80, seed=seed)),
    ("CS-Cluster-Outlier (-30%)",    lambda df, seed=42: coreset_cluster_outlier_reduction(df, fraction=0.70, seed=seed)),
    ("Global-temp+MinK3 (Sep 2000)", lambda df, seed=42: global_temporal_with_mink_reduction(df, "2000-09-01", min_k=3)),
    ("Baseline",                     lambda df, seed=42: df.copy()),
    ("CS-Leverage (-20%) [ref]",     lambda df, seed=42: coreset_leverage_reduction(df, fraction=0.80, seed=seed)),
    ("CS-Cluster (-20%) [ref]",      lambda df, seed=42: coreset_cluster_reduction(df, fraction=0.80, seed=seed)),
]


def apply_reduction(ratings: pd.DataFrame, strategy_name: str, seed: int = 42) -> pd.DataFrame:
    """Return the reduced ratings DataFrame for the given strategy name and seed."""
    for name, fn in REDUCTION_STRATEGIES + EXTRA_STRATEGIES:
        if name == strategy_name:
            return fn(ratings, seed=seed)
    raise ValueError(f"Unknown strategy: {strategy_name!r}")


def build_global_split(ratings: pd.DataFrame):
    """
    Perform leave-one-out split on the RAW, unfiltered ratings.

    This must be called ONCE before any reduction strategy is applied.
    The returned val_df and test_df are then reused for every strategy so
    that all models and strategies are compared on the SAME test set.

    Split:
      - test  : most-recent interaction per user    (1 row / user)
      - val   : 2nd most-recent interaction per user (1 row / user)
      - train : all remaining interactions

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame  (original columns, no ID remapping)
    """
    df = ratings.sort_values(["user_id", "timestamp"]).copy()
    df["_rank"] = df.groupby("user_id").cumcount(ascending=False)  # 0 = most recent

    test_df  = df[df["_rank"] == 0].drop(columns="_rank").reset_index(drop=True)
    val_df   = df[df["_rank"] == 1].drop(columns="_rank").reset_index(drop=True)
    train_df = df[df["_rank"] >= 2].drop(columns="_rank").reset_index(drop=True)

    return train_df, val_df, test_df


def build_cornac_dataset(
    train_reduced_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Convert pre-split DataFrames into cornac Dataset objects.

    Design guarantees
    -----------------
    * ID maps are built from TRAINING DATA ONLY - the model's item vocabulary
      is exactly what it was trained on.
    * val_df and test_df are the GLOBAL fixed splits (same across all strategies).
    * Items that appear in val/test but NOT in training are out-of-vocabulary
      (OOV). They are excluded from the cornac val/test sets and counted in meta
      so evaluate_model can treat them as misses.

    Reserved-but-inactive: the returned val_set is not currently used by any
    caller for early stopping or model selection - none of cornac's BPR/VAECF/
    EASE are wired up to watch it during `.fit()` in this project. It is kept
    in the split so that wiring up real early stopping later doesn't require
    touching the leakage-free split logic again.

    Parameters
    ----------
    train_reduced_df : ratings kept for training after applying a reduction strategy
    val_df           : global fixed validation interactions (from build_global_split)
    test_df          : global fixed test interactions       (from build_global_split)

    Returns
    -------
    train_set, val_set, test_set : cornac.data.Dataset
    meta : dict
        n_test_users  – total test users (including OOV) - fixed denominator
        n_oov_test    – users whose test item is not in training vocabulary
        user_map      – original user_id → cornac int index
        item_map      – original movie_id → cornac int index (train vocab only)
    """
    import cornac

    all_user_ids = sorted(
        set(train_reduced_df["user_id"]) |
        set(val_df["user_id"])           |
        set(test_df["user_id"])
    )
    user_map = {uid: i for i, uid in enumerate(all_user_ids)}

    train_item_ids  = sorted(train_reduced_df["movie_id"].unique())
    item_map        = {iid: i for i, iid in enumerate(train_item_ids)}
    train_movie_set = set(train_item_ids)

    n_oov_test = sum(
        1 for _, row in test_df.iterrows()
        if row["movie_id"] not in train_movie_set
    )

    def to_uir(df):
        rows = []
        for _, row in df.iterrows():
            u = user_map.get(row["user_id"])
            i = item_map.get(row["movie_id"])   # None for OOV items
            if u is not None and i is not None:
                rows.append((u, i, float(row["rating"])))
        return rows

    train_uir = to_uir(train_reduced_df)
    val_uir   = to_uir(val_df)
    test_uir  = to_uir(test_df)   # OOV test items silently dropped here; counted via meta["n_oov_test"]

    train_set = cornac.data.Dataset.from_uir(train_uir)
    val_set   = cornac.data.Dataset.from_uir(val_uir)  if val_uir  else None
    test_set  = cornac.data.Dataset.from_uir(test_uir) if test_uir else None

    meta = {
        "n_test_users": len(test_df),
        "n_oov_test":   n_oov_test,
        "user_map":     user_map,
        "item_map":     item_map,
    }

    return train_set, val_set, test_set, meta


def slugify(name: str) -> str:
    """Convert a strategy/model name to a safe directory-name string."""
    return (
        name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "minus")
            .replace("%", "pct")
            .replace("/", "_")
    )

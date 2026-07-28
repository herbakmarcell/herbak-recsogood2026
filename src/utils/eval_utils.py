"""
utils/eval_utils.py
====================
Shared model-evaluation logic used by 02_train_models.py and 03_train_extras.py
— single source of truth so evaluation semantics can't silently drift between
scripts, and so the seeded-run / common-subset machinery only has to be
reasoned about once.

Why seeded runs
---------------
A single fixed seed (as this project used until now) means "repeated" energy
measurements are deterministic re-runs, not independent samples — there is no
way to tell whether a strategy's Recall@K difference from baseline is real or
sampling noise. `DEFAULT_SEEDS` below drives BOTH the reduction-strategy
randomness (where applicable) and the model's own training randomness for
each replicate, so a single seed loop gives you variance estimates for
accuracy AND energy from the same computational budget — see
02_train_models.py::run() for how this is wired together.

Metrics reported by evaluate_model()
-------------------------------------
Recall@K / NDCG@K         : headline, sampled-negative, ZERO-PADDED across the
                            full fixed global test-user set (OOV/unevaluable
                            users count as 0 — this is what all previous
                            versions of this project reported as "the"
                            Recall/NDCG).
Recall@K_evaluable        : same numerator, but averaged only over users that
                            WERE evaluable (no zero-padding) — answers "how
                            good is this model for users it can actually see"
                            separately from "how many users can it see."
Eval_User_Rate            : n_evaluated / n_global_test_users — the
                            reachability axis (this is what earlier versions
                            of this project mislabeled "Coverage").
Coverage                  : CATALOG coverage — fraction of the model's full
                            item vocabulary that appears in ANY user's
                            FULL-CATALOG top-K ranking. `model.score(u)`
                            already returns a full-catalog score vector, so
                            this costs nothing extra to compute; it is
                            distinct from (and usually higher granularity
                            than) the small 101-item sampled candidate set
                            used for Recall/NDCG.
Recall@K_common_subset    : Recall@K computed only over test users whose true
                            item survives in a shared cross-strategy item
                            vocabulary, using one FIXED negative-sampling seed
                            so every strategy faces the exact same candidate
                            items per user. This removes the confound where a
                            strategy with a smaller/differently-composed item
                            vocabulary faces an easier or harder sampled-
                            ranking task purely as a side effect of its own
                            vocabulary size (see project README, item B).
                            Simplification: unlike the headline Recall@K, this
                            diagnostic does NOT exclude a user's own training
                            items from the fixed negative pool (doing so would
                            reintroduce a per-strategy-dependent pool, exactly
                            what this metric exists to avoid) — treat it as a
                            controlled relative comparison across strategies,
                            not as an absolute Recall figure.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Shared across 02/03 so "the canonical run" always means the same
# thing (SEEDS[0]) — e.g. which model gets saved to disk for later reuse.
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

# Fixed, independent of DEFAULT_SEEDS: the common-subset candidate pool must
# stay IDENTICAL across every strategy/model/seed, so it uses its own seed.
COMMON_SUBSET_SEED = 20240101


def _build_ground_truth(test_set, train_set) -> Tuple[Dict[int, List[int]], Dict[int, set]]:
    gt, seen = {}, {}
    for u, i, _ in zip(*test_set.uir_tuple):
        gt.setdefault(int(u), []).append(int(i))
    for u, i, _ in zip(*train_set.uir_tuple):
        seen.setdefault(int(u), set()).add(int(i))
    return gt, seen

def split_users_by_history_length(
    train_raw_df: pd.DataFrame,
    global_test_df: pd.DataFrame
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Groups users into Short (bottom 33%), Medium (middle 33%), and Long (top 33%)
    profiles based on their interaction count in the raw training dataset.
    Returns the mapping of user_id -> group, and the exact count of *test users*
    in each group (for accurate zero-padding during evaluation).
    """
    counts = train_raw_df.groupby("user_id").size()
    q33 = counts.quantile(0.3333)
    q67 = counts.quantile(0.6667)

    user_groups = {}
    for uid, count in counts.items():
        if count <= q33:
            user_groups[uid] = "short"
        elif count <= q67:
            user_groups[uid] = "medium"
        else:
            user_groups[uid] = "long"

    group_counts = {"short": 0, "medium": 0, "long": 0, "unknown": 0}
    for uid in global_test_df["user_id"].unique():
        g = user_groups.get(uid, "unknown")
        group_counts[g] += 1

    return user_groups, group_counts


def build_common_item_vocabulary(
    train_raw_df: pd.DataFrame,
    strategy_names: Iterable[str],
    apply_reduction_fn,
    seed: int = DEFAULT_SEEDS[0],
) -> set:
    """
    Intersection of item vocabularies across all named strategies, using one
    reference seed. Every strategy's own vocabulary is guaranteed to be a
    superset of this common set, so sampling negatives from it (see
    build_common_subset_candidates) removes the confound of differently
    sized/composed per-strategy negative-sampling pools.

    *apply_reduction_fn* should be `data_utils.apply_reduction`, passed in
    rather than imported here to avoid a circular import between the two
    utils modules.
    """
    common = None
    for name in strategy_names:
        reduced = apply_reduction_fn(train_raw_df, name, seed=seed)
        vocab = set(reduced["movie_id"].unique())
        common = vocab if common is None else (common & vocab)
    return common or set()


def build_common_subset_candidates(
    global_test_df: pd.DataFrame,
    common_item_ids: set,
    n_neg: int = 100,
    seed: int = COMMON_SUBSET_SEED,
) -> Dict[int, Tuple[int, List[int]]]:
    """
    For every (user, true test movie_id) pair whose true item survives in
    *common_item_ids*, pre-sample a FIXED set of negative movie_ids (also
    drawn from common_item_ids) using one master seed shared across every
    strategy, model, and seeded run.

    Returned in original movie_id space — callers translate through their
    own strategy's item_map (from build_cornac_dataset's meta) at evaluation
    time, since each strategy has its own movie_id -> cornac-index mapping.
    """
    rng = np.random.default_rng(seed)
    common_arr = np.array(sorted(common_item_ids))
    candidates: Dict[int, Tuple[int, List[int]]] = {}

    for row in global_test_df.itertuples(index=False):
        uid, true_item = int(row.user_id), int(row.movie_id)
        if true_item not in common_item_ids:
            continue
        pool = common_arr[common_arr != true_item]
        if len(pool) == 0:
            continue
        negs = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
        candidates[uid] = (true_item, negs.tolist())

    return candidates


def evaluate_model(
    model,
    train_set,
    test_set,
    item_map: dict,
    n_global_test_users: int,
    n_neg: int = 100,
    topk: int = 10,
    seed: int = 42,
    common_subset_candidates: Optional[Dict[int, Tuple[int, List[int]]]] = None,
    user_groups: Optional[Dict[int, str]] = None,
    group_counts: Optional[Dict[str, int]] = None,
) -> dict:
    """
    Sampled-negative evaluation on a FIXED global test set. See module
    docstring for what each returned metric means and why.
    """
    rng     = np.random.default_rng(seed)
    n_items = train_set.num_items   # model's item vocabulary size

    gt, seen = _build_ground_truth(test_set, train_set)

    recalls, ndcgs = [], []
    recalls_short, recalls_medium, recalls_long = [], [], []
    per_user_recalls = {}
    recommended_items_full = set()   # full-catalog coverage tracking

    cornac_to_raw_user = {idx: raw for raw, idx in train_set.uid_map.items()}

    for u, test_items in gt.items():
        try:
            test_set_u = set(test_items)
            exclude    = seen.get(u, set()) | test_set_u

            negs = []
            while len(negs) < n_neg:
                batch = rng.integers(0, n_items, size=n_neg * 2).tolist()
                negs.extend(i for i in batch if i not in exclude)
            negs = negs[:n_neg]

            all_items  = test_items + negs
            all_scores = np.asarray(model.score(u)).flatten()   # full-catalog vector
            ranked     = sorted(all_items, key=lambda i: -all_scores[i])
            top_k      = ranked[:topk]

            hits = len(set(top_k) & test_set_u)
            hit_ratio = hits / min(len(test_items), topk)
            recalls.append(hit_ratio)

            raw_uid = cornac_to_raw_user.get(u)
            if raw_uid is not None:
                per_user_recalls[raw_uid] = hit_ratio

            if user_groups:
                g = user_groups.get(raw_uid, "unknown") if raw_uid is not None else "unknown"
                if g == "short":
                    recalls_short.append(hit_ratio)
                elif g == "medium":
                    recalls_medium.append(hit_ratio)
                elif g == "long":
                    recalls_long.append(hit_ratio)

            dcg   = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top_k) if i in test_set_u)
            ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), topk)))
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

            # all_scores already spans every item in the model's vocabulary,
            # so ranking it fully here costs nothing extra.
            if n_items > topk:
                full_top_k = np.argpartition(-all_scores, topk - 1)[:topk]
            else:
                full_top_k = np.arange(n_items)
            recommended_items_full.update(full_top_k.tolist())
        except Exception:
            pass   # user not in model's user space — genuine errors are not swallowed

    n_evaluated = len(recalls)
    n_missed    = n_global_test_users - n_evaluated

    all_recalls = recalls + [0.0] * n_missed
    all_ndcgs   = ndcgs   + [0.0] * n_missed

    result = {
        f"Recall@{topk}":            float(np.mean(all_recalls)),
        f"NDCG@{topk}":              float(np.mean(all_ndcgs)),
        f"Recall@{topk}_evaluable":  float(np.mean(recalls)) if recalls else 0.0,
        "Eval_User_Rate":            n_evaluated / n_global_test_users if n_global_test_users else 0.0,
        "Coverage":                  len(recommended_items_full) / n_items if n_items else 0.0,
        "per_user_recalls":          per_user_recalls,
    }

    if user_groups and group_counts:
        result[f"Recall@{topk}_short"] = float(np.sum(recalls_short) / group_counts["short"]) if group_counts["short"] else 0.0
        result[f"Recall@{topk}_medium"] = float(np.sum(recalls_medium) / group_counts["medium"]) if group_counts["medium"] else 0.0
        result[f"Recall@{topk}_long"] = float(np.sum(recalls_long) / group_counts["long"]) if group_counts["long"] else 0.0

    if common_subset_candidates:
        cs_hits = []
        for raw_u, (true_movie_id, neg_movie_ids) in common_subset_candidates.items():
            true_idx = item_map.get(true_movie_id)
            if true_idx is None:
                continue   # this strategy's vocab dropped the true item — excluded, not zero-padded
            neg_idxs = [item_map[m] for m in neg_movie_ids if m in item_map]
            if not neg_idxs:
                continue

            cornac_u = train_set.uid_map.get(str(raw_u))
            if cornac_u is None:
                cornac_u = train_set.uid_map.get(int(raw_u))
            if cornac_u is None:
                continue

            try:
                all_scores = np.asarray(model.score(cornac_u)).flatten()
                candidates = [true_idx] + neg_idxs
                ranked     = sorted(candidates, key=lambda i: -all_scores[i])
                cs_hits.append(int(true_idx in ranked[:topk]))
            except (KeyError, IndexError):
                continue

        result[f"Recall@{topk}_common_subset"] = float(np.mean(cs_hits)) if cs_hits else float("nan")
        result["Common_subset_n_users"] = len(cs_hits)

    return result


def paired_wilcoxon_pvalue(sample_a, sample_b) -> float:
    """
    Paired Wilcoxon signed-rank test p-value, e.g. comparing a strategy's
    per-seed Recall@K array against baseline's per-seed Recall@K array for
    the same model. Returns NaN (not 1.0) when the test can't be computed
    (too few paired samples, or all differences exactly zero) — callers
    should treat NaN as "insufficient evidence," not as "no effect."

    Caveat: with few seeds (this project uses 5), Wilcoxon has very low
    power — treat significant results as suggestive, not conclusive, and
    always report the raw per-seed values alongside the p-value.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = a - b
    if np.all(diffs == 0):
        # Some scipy versions return p=1.0 here (with a RuntimeWarning) instead
        # of raising — but "all differences are exactly zero" means the test
        # is degenerate, not that we have strong evidence of no difference.
        return float("nan")
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except ValueError:
        # e.g. all non-zero differences share a value scipy still can't rank
        return float("nan")

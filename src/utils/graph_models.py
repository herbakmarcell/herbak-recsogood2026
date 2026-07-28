"""
utils/graph_models.py
=====================
Pure-PyTorch implementations of LightGCN and NGCF with a cornac-compatible
interface (fit / score / save / load) so they drop into the training loop
without any DGL or PyG dependency.

References
----------
LightGCN: He et al. 2020 – https://arxiv.org/abs/2002.02126
NGCF:     Wang et al. 2019 – https://arxiv.org/abs/1905.08108
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim


def _build_norm_adj(train_set) -> torch.sparse.Tensor:
    """
    Build the symmetrically normalised adjacency matrix  D^{-1/2} A D^{-1/2}
    of the bipartite user-item interaction graph.

    The combined adjacency is:
        A = [ 0   R  ]
            [ R^T 0  ]
    where R is the (n_users × n_items) binary interaction matrix.
    """
    n_users = train_set.num_users
    n_items = train_set.num_items
    N       = n_users + n_items

    rows, cols = [], []
    for u, i, _ in zip(*train_set.uir_tuple):
        rows.append(int(u))
        cols.append(int(i) + n_users)
        rows.append(int(i) + n_users)
        cols.append(int(u))

    data = np.ones(len(rows), dtype=np.float32)
    A    = sp.coo_matrix((data, (rows, cols)), shape=(N, N))

    deg  = np.asarray(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    d_inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm     = D_inv_sqrt @ A @ D_inv_sqrt

    A_coo = A_norm.tocoo().astype(np.float32)
    indices = torch.tensor(
        np.vstack([A_coo.row, A_coo.col]), dtype=torch.long
    )
    values  = torch.tensor(A_coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N, N))


class _BPRSampler:
    """
    Pre-builds user→positive-items mappings once; subsequent calls to
    sample() are cheap O(batch_size) operations with no dict rebuilding.
    """
    def __init__(self, train_set):
        uir = train_set.uir_tuple
        self.n_items = train_set.num_items

        pos_sets:  dict[int, set]  = {}
        pos_lists: dict[int, list] = {}
        for u, i, _ in zip(*uir):
            u, i = int(u), int(i)
            pos_sets.setdefault(u, set()).add(i)
            pos_lists.setdefault(u, []).append(i)

        self._pos_sets  = pos_sets
        self._pos_lists = pos_lists
        self._users     = np.array(list(pos_lists.keys()), dtype=np.int64)

    def sample(self, batch_size: int, rng: np.random.Generator):
        chosen = rng.choice(self._users, size=batch_size, replace=True)

        pos_is = np.array(
            [rng.choice(self._pos_lists[u]) for u in chosen], dtype=np.int64
        )
        neg_is = rng.integers(0, self.n_items, size=batch_size)
        for j in range(batch_size):
            while neg_is[j] in self._pos_sets[chosen[j]]:
                neg_is[j] = rng.integers(0, self.n_items)

        return (
            torch.tensor(chosen, dtype=torch.long),
            torch.tensor(pos_is, dtype=torch.long),
            torch.tensor(neg_is, dtype=torch.long),
        )


class _LightGCNNet(nn.Module):
    def __init__(self, n_users, n_items, emb_size, num_layers):
        super().__init__()
        self.n_users   = n_users
        self.n_items   = n_items
        self.num_layers = num_layers
        self.emb = nn.Embedding(n_users + n_items, emb_size)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, adj):
        e = self.emb.weight
        all_e = [e]
        for _ in range(self.num_layers):
            e = torch.sparse.mm(adj, e)
            all_e.append(e)
        final = torch.stack(all_e, dim=1).mean(dim=1)
        return final[:self.n_users], final[self.n_users:]


class LightGCN:
    """Cornac-compatible LightGCN wrapper (pure PyTorch, no DGL)."""

    name = "LightGCN"

    def __init__(
        self,
        emb_size:             int   = 64,
        num_layers:           int   = 3,
        num_epochs:           int   = 50,
        learning_rate:        float = 1e-3,
        batch_size:           int   = 256,
        lambda_reg:           float = 1e-4,
        max_steps_per_epoch:  int   = 500,
        seed:                 int   = 42,
        use_gpu:              bool  = True,
        verbose:              bool  = False,
    ):
        self.emb_size             = emb_size
        self.num_layers           = num_layers
        self.num_epochs           = num_epochs
        self.learning_rate        = learning_rate
        self.batch_size           = batch_size
        self.lambda_reg           = lambda_reg
        self.max_steps_per_epoch  = max_steps_per_epoch
        self.seed                 = seed
        self.verbose              = verbose
        self.device               = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self._net: Optional[_LightGCNNet] = None
        self.is_fitted = False


    def fit(self, train_set):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        n_users = train_set.num_users
        n_items = train_set.num_items

        adj = _build_norm_adj(train_set).to(self.device)
        net = _LightGCNNet(n_users, n_items, self.emb_size, self.num_layers).to(self.device)
        opt = optim.Adam(net.parameters(), lr=self.learning_rate)

        # Sampler is built once - O(n_interactions) setup, O(batch) per call
        sampler = _BPRSampler(train_set)
        n_interactions = len(train_set.uir_tuple[0])
        steps = min(n_interactions // self.batch_size, self.max_steps_per_epoch)

        # Backward pass runs every 50 mini-batches instead of every one, to
        # avoid building one giant autograd graph across the full epoch.
        chunk_size = 50

        for epoch in range(self.num_epochs):
            net.train()
            total_loss = 0.0
            chunk_start = 0

            while chunk_start < steps:
                chunk_end = min(chunk_start + chunk_size, steps)

                user_emb, item_emb = net(adj)

                opt.zero_grad()
                chunk_loss = torch.tensor(0.0, device=self.device)

                for _ in range(chunk_end - chunk_start):
                    u, pos_i, neg_i = sampler.sample(self.batch_size, rng)
                    u     = u.to(self.device)
                    pos_i = pos_i.to(self.device)
                    neg_i = neg_i.to(self.device)

                    u_e   = user_emb[u]
                    pos_e = item_emb[pos_i]
                    neg_e = item_emb[neg_i]

                    bpr = -torch.log(torch.sigmoid(
                        (u_e * pos_e).sum(-1) - (u_e * neg_e).sum(-1)
                    ) + 1e-8).mean()
                    reg = self.lambda_reg * (
                        u_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)
                    ) / self.batch_size
                    chunk_loss = chunk_loss + bpr + reg

                chunk_loss.backward()
                opt.step()
                total_loss += chunk_loss.item()
                chunk_start = chunk_end

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.num_epochs}  loss={total_loss/steps:.4f}")

        net.eval()
        with torch.no_grad():
            u_e, i_e = net(adj)
        self._user_emb = u_e.cpu().numpy()
        self._item_emb = i_e.cpu().numpy()
        self._net      = net
        self.is_fitted = True
        return self

    def score(self, user_idx: int) -> np.ndarray:
        return self._user_emb[user_idx] @ self._item_emb.T

    def save(self, save_dir: str):
        path = Path(save_dir) / "lightgcn.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {"user_emb": self._user_emb, "item_emb": self._item_emb,
                 "config": {k: v for k, v in self.__dict__.items()
                             if not k.startswith("_") and k != "device"}},
                f,
            )

    @classmethod
    def load(cls, save_dir: str) -> "LightGCN":
        path = next(Path(save_dir).glob("*.pkl"))
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(**data["config"])
        m._user_emb  = data["user_emb"]
        m._item_emb  = data["item_emb"]
        m.is_fitted  = True
        return m


class _NGCFNet(nn.Module):
    def __init__(self, n_users, n_items, emb_size, layer_sizes, dropout_rates):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb = nn.Embedding(n_users + n_items, emb_size)
        nn.init.xavier_uniform_(self.emb.weight)

        self.layers = nn.ModuleList()
        self.drops  = nn.ModuleList()
        in_dim = emb_size
        for out_dim, drop in zip(layer_sizes, dropout_rates):
            self.layers.append(nn.Linear(in_dim, out_dim, bias=False))
            self.drops.append(nn.Dropout(p=drop))
            in_dim = out_dim

        self.out_dim = emb_size + sum(layer_sizes)   # concat of all layers

    def forward(self, adj):
        e = self.emb.weight
        all_e = [e]
        for lin, drop in zip(self.layers, self.drops):
            agg = torch.sparse.mm(adj, e)
            e   = torch.relu(lin(agg) + lin(e))
            e   = drop(e)
            all_e.append(e)

        final = torch.cat(all_e, dim=-1)
        return final[:self.n_users], final[self.n_users:]


class NGCF:
    """Cornac-compatible NGCF wrapper (pure PyTorch, no DGL)."""

    name = "NGCF"

    def __init__(
        self,
        emb_size:      int        = 64,
        layer_sizes:   list       = None,
        dropout_rates: list       = None,
        num_epochs:    int        = 50,
        learning_rate: float      = 1e-3,
        batch_size:    int        = 256,
        lambda_reg:    float      = 1e-4,
        seed:          int        = 42,
        use_gpu:       bool       = True,
        verbose:       bool       = False,
    ):
        self.emb_size      = emb_size
        self.layer_sizes   = layer_sizes   or [64, 64, 64]
        self.dropout_rates = dropout_rates or [0.1, 0.1, 0.1]
        self.num_epochs    = num_epochs
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.lambda_reg    = lambda_reg
        self.seed          = seed
        self.verbose       = verbose
        self.device        = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self._net: Optional[_NGCFNet] = None
        self.is_fitted = False

    def fit(self, train_set):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        n_users = train_set.num_users
        n_items = train_set.num_items

        adj = _build_norm_adj(train_set).to(self.device)
        net = _NGCFNet(
            n_users, n_items, self.emb_size,
            self.layer_sizes, self.dropout_rates,
        ).to(self.device)
        opt = optim.Adam(net.parameters(), lr=self.learning_rate)

        sampler = _BPRSampler(train_set)
        n_interactions = len(train_set.uir_tuple[0])
        steps = max(1, n_interactions // self.batch_size)

        for epoch in range(self.num_epochs):
            net.train()
            total_loss = 0.0
            for _ in range(steps):
                u, pos_i, neg_i = sampler.sample(self.batch_size, rng)
                u    = u.to(self.device)
                pos_i = pos_i.to(self.device)
                neg_i = neg_i.to(self.device)

                user_emb, item_emb = net(adj)
                u_e   = user_emb[u]
                pos_e = item_emb[pos_i]
                neg_e = item_emb[neg_i]

                pos_scores = (u_e * pos_e).sum(-1)
                neg_scores = (u_e * neg_e).sum(-1)
                bpr_loss   = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                reg_loss   = self.lambda_reg * (
                    u_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)
                ) / self.batch_size
                loss = bpr_loss + reg_loss

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.num_epochs}  loss={total_loss/steps:.4f}")

        net.eval()
        with torch.no_grad():
            u_e, i_e = net(adj)
        self._user_emb = u_e.detach().cpu().numpy()
        self._item_emb = i_e.detach().cpu().numpy()
        self._net      = net
        self.is_fitted = True
        return self

    def score(self, user_idx: int) -> np.ndarray:
        return self._user_emb[user_idx] @ self._item_emb.T

    def save(self, save_dir: str):
        path = Path(save_dir) / "ngcf.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {"user_emb": self._user_emb, "item_emb": self._item_emb,
                 "config": {k: v for k, v in self.__dict__.items()
                             if not k.startswith("_") and k != "device"}},
                f,
            )

    @classmethod
    def load(cls, save_dir: str) -> "NGCF":
        path = next(Path(save_dir).glob("*.pkl"))
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(**data["config"])
        m._user_emb  = data["user_emb"]
        m._item_emb  = data["item_emb"]
        m.is_fitted  = True
        return m


class ELSAWrapper:
    """
    Cornac-compatible wrapper for recombee/ELSA.

    ELSA is an input-conditioned model: at inference time it needs the user's
    training interaction row.  We precompute all user scores in a single GPU
    forward pass right after fitting and cache them as a plain numpy matrix so
    that score(u) is an O(1) lookup - identical to BPR / EASE at evaluation.
    """

    name = "ELSA"

    def __init__(
        self,
        n_dims:     int   = 64,
        epochs:     int   = 50,
        batch_size: int   = 512,
        lr:         float = 0.1,
        use_gpu:    bool  = True,
        verbose:    bool  = False,
    ):
        self.n_dims     = n_dims
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.verbose    = verbose
        self.device     = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self._scores: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, train_set) -> "ELSAWrapper":
        import warnings
        from elsa import ELSA as _ELSA
        from scipy.sparse import csr_matrix

        n_users = train_set.num_users
        n_items = train_set.num_items
        u_arr, i_arr, _ = train_set.uir_tuple

        R = csr_matrix(
            (np.ones(len(u_arr)), (u_arr.astype(int), i_arr.astype(int))),
            shape=(n_users, n_items),
        )

        model = _ELSA(n_items=n_items, n_dims=self.n_dims,
                      device=self.device, lr=self.lr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress sparse invariant warning
            model.fit(R, epochs=self.epochs,
                      batch_size=self.batch_size, verbose=self.verbose)

        # Precompute all-user score matrix in one forward pass - (n_users, n_items)
        with torch.no_grad():
            all_preds = model.predict(R, batch_size=self.batch_size)

        self._scores = all_preds.cpu().numpy()   # shape: (n_users, n_items)
        self.is_fitted = True
        return self

    def score(self, user_idx: int) -> np.ndarray:
        """Return predicted scores for all items for the given user index."""
        return self._scores[user_idx]            # shape: (n_items,)

    def save(self, save_dir: str):
        path = Path(save_dir) / "elsa.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "scores": self._scores,
                "config": {k: v for k, v in self.__dict__.items()
                           if not k.startswith("_") and k != "device"},
            }, f)

    @classmethod
    def load(cls, save_dir: str) -> "ELSAWrapper":
        path = Path(save_dir) / "elsa.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(**data["config"])
        obj._scores   = data["scores"]
        obj.is_fitted = True
        return obj

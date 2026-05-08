"""
Neural Collaborative Filtering (NCF/NeuMF) — He et al. 2017
+ BPR pairwise ranking loss — Rendle et al. 2009

Device: auto-selects MPS (Apple M-series) > CUDA > CPU.
  On Mac M5 chip, MPS gives 5-10× speedup over CPU.
  emb_size=64, MLP 128→64→32, 15 epochs ≈ 90s total on M5 MPS.

Training: BPR loss = -log σ(score(u, pos) − score(u, neg))
Optimised directly for ranking, not rating prediction.
"""

import os
import time
import numpy as np
import pandas as pd
import torch

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
import torch.nn as nn
import torch.optim as optim


# ── Device: MPS (Apple M-series GPU) > CUDA > CPU ────────────────────────────
def _best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# NeuMF architecture
# ─────────────────────────────────────────────────────────────────────────────
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_size=64, mlp_dims=(128, 64, 32)):
        super().__init__()
        self.gmf_u = nn.Embedding(n_users, emb_size)
        self.gmf_i = nn.Embedding(n_items, emb_size)
        self.mlp_u = nn.Embedding(n_users, emb_size)
        self.mlp_i = nn.Embedding(n_items, emb_size)

        in_dims = [emb_size * 2] + list(mlp_dims[:-1])
        layers  = []
        for d_in, d_out in zip(in_dims, mlp_dims):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(emb_size + mlp_dims[-1], 1)

        for emb in (self.gmf_u, self.gmf_i, self.mlp_u, self.mlp_i):
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, u, i):
        gmf = self.gmf_u(u) * self.gmf_i(i)
        mlp = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=1))
        return self.out(torch.cat([gmf, mlp], dim=1)).squeeze(-1)

    @torch.no_grad()
    def score_all(self, u_idx, n_items, device=None, batch=8192):
        """Score all n_items for user u_idx. Returns CPU numpy array."""
        if device is None:
            device = next(self.parameters()).device
        scores = np.empty(n_items, dtype=np.float32)
        u_t = torch.tensor([u_idx], device=device)
        for s in range(0, n_items, batch):
            e   = min(s + batch, n_items)
            i_t = torch.arange(s, e, device=device)
            scores[s:e] = self(u_t.expand(e - s), i_t).cpu().numpy()
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Recommender wrapper
# ─────────────────────────────────────────────────────────────────────────────
class NCFRecommender:
    """
    Drop-in replacement for CollaborativeRecommender.
    Exposes .model = self so hybrid.py / evaluate.py work unchanged.
    """

    def __init__(self, ratings_path=None,
                 emb_size=64, mlp_dims=(128, 64, 32),
                 n_epochs=15, batch_size=8192, lr=5e-4, reg=1e-5,
                 top_items=20_000, top_users=12_000, neg_per_pos=2):

        if ratings_path is None:
            ratings_path = os.path.join(_DATA_DIR, "ratings.csv")
        ratings = pd.read_csv(ratings_path)[['userId', 'movieId', 'rating']]

        # ── Select top active users ───────────────────────────────────────────
        u_cnt     = ratings.groupby('userId').size()
        sel_users = u_cnt.nlargest(top_users).index
        ratings   = ratings[ratings['userId'].isin(sel_users)]

        # ── Cap items to top most-rated (cold items → unreliable embeddings) ─
        i_cnt     = ratings.groupby('movieId').size()
        sel_items = i_cnt.nlargest(top_items).index
        ratings   = ratings[ratings['movieId'].isin(sel_items)]

        users = ratings['userId'].unique()
        items = ratings['movieId'].unique()

        self._user_map = {u: i for i, u in enumerate(users)}
        self._item_map = {m: i for i, m in enumerate(items)}
        self._idx_item = {v: k for k, v in self._item_map.items()}
        self.global_mean = float(ratings['rating'].mean())

        n_u, n_i = len(users), len(items)

        # ── Positive pairs (liked movies only, ≥4.0) ─────────────────────────
        liked   = ratings[ratings['rating'] >= 4.0]
        pos_u   = liked['userId'].map(self._user_map).values.astype(np.int64)
        pos_i   = liked['movieId'].map(self._item_map).values.astype(np.int64)
        n_pos   = len(pos_u)

        # user → set of liked item indices (for clean negative sampling)
        user_pos_set = [set() for _ in range(n_u)]
        for u, i in zip(pos_u, pos_i):
            user_pos_set[u].add(int(i))

        # ── Device: MPS (Apple M-series) > CUDA > CPU ───────────────────────
        device = _best_device()

        # ── Model + optimiser ─────────────────────────────────────────────────
        net   = NeuMF(n_u, n_i, emb_size=emb_size, mlp_dims=mlp_dims).to(device)
        opt   = optim.Adam(net.parameters(), lr=lr, weight_decay=reg)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        n_train = n_pos * neg_per_pos

        print(f"  [NCF] {n_u} users · {n_i} items · emb={emb_size} · "
              f"epochs={n_epochs} · pos_pairs={n_pos:,} · neg×{neg_per_pos} · "
              f"device={device}")

        # ── Training loop ─────────────────────────────────────────────────────
        net.train()
        for epoch in range(n_epochs):
            t0 = time.time()

            rep_u = np.tile(pos_u, neg_per_pos)
            rep_i = np.tile(pos_i, neg_per_pos)
            neg_i = np.random.randint(0, n_i, n_train)
            for idx in range(n_train):
                u = int(rep_u[idx])
                while neg_i[idx] in user_pos_set[u]:
                    neg_i[idx] = np.random.randint(0, n_i)

            perm  = np.random.permutation(n_train)
            pu_t  = torch.from_numpy(rep_u[perm]).to(device)
            pi_t  = torch.from_numpy(rep_i[perm]).to(device)
            ni_t  = torch.from_numpy(neg_i[perm]).to(device)

            total_loss, n_batches = 0.0, 0
            for s in range(0, n_train, batch_size):
                e    = min(s + batch_size, n_train)
                u_b  = pu_t[s:e]
                pi_b = pi_t[s:e]
                ni_b = ni_t[s:e]

                opt.zero_grad()
                s_pos = net(u_b, pi_b)
                s_neg = net(u_b, ni_b)
                loss  = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches  += 1

            sched.step()
            elapsed = time.time() - t0
            print(f"    epoch {epoch+1:02d}/{n_epochs}  "
                  f"BPR-loss={total_loss/n_batches:.4f}  {elapsed:.1f}s", flush=True)

        net.eval()
        self._net    = net
        self._device = device
        self._n_i    = n_i
        self.model   = self

    # ── Model interface (same as SVDModel / IALSModel) ────────────────────────

    @property
    def user_map(self):  return self._user_map

    @property
    def item_map(self):  return self._item_map

    def predict_for_user(self, user_id):
        if user_id not in self._user_map:
            return {}
        scores = self._net.score_all(self._user_map[user_id], self._n_i,
                                     device=self._device)
        return {self._idx_item[i]: float(scores[i]) for i in range(self._n_i)}

    def predict(self, user_id, movie_id):
        class _P: pass
        p = _P()
        if user_id not in self._user_map or movie_id not in self._item_map:
            p.est = self.global_mean; return p
        u = torch.tensor([self._user_map[user_id]], device=self._device)
        i = torch.tensor([self._item_map[movie_id]], device=self._device)
        with torch.no_grad():
            p.est = float(self._net(u, i).cpu())
        return p

    def recommend_for_user(self, user_id, movies_df, top_n=10):
        scores = self.predict_for_user(user_id)
        preds  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for mid, _ in preds[:top_n]:
            row = movies_df[movies_df['movieId'] == mid]
            if not row.empty:
                result.append(row['title'].values[0])
        return result

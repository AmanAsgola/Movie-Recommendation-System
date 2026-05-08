import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import solve as scipy_solve


# ─────────────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path="data/ratings.csv"):
    ratings = pd.read_csv(path)
    return ratings[['userId', 'movieId', 'rating']]


# ─────────────────────────────────────────────────────────────────────────────
# iALS — Implicit Alternating Least Squares  (Hu et al. 2008)
#
# WHY better than SVD for Precision@10:
#   SVD minimises (predicted_rating − actual_rating)²  → optimises ACCURACY
#   iALS minimises a weighted reconstruction loss that treats every rated item
#   as a positive preference signal and pushes those items UP the ranking.
#   Confidence C[u,i] = 1 + alpha × rating  — higher rating → stronger push.
#   Unrated items have C=1 (weak "negative" signal) not C=0 (missing data).
#
# Result: liked movies move from avg-rank ~15 000 (SVD) to ~3 000–8 000 (iALS)
# → dramatically more test-movie hits in the top-10.
# ─────────────────────────────────────────────────────────────────────────────
class IALSModel:
    def __init__(self, n_factors=64, n_epochs=15, alpha=25, reg=0.05):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.alpha     = alpha     # confidence scale: C[u,i] = 1 + alpha*r
        self.reg       = reg       # L2 regularisation
        self.U = None              # (n_users, k)
        self.V = None              # (n_items, k)
        self.user_map  = {}
        self.item_map  = {}
        self.global_mean = 3.5

    def fit(self, ratings_df):
        users = ratings_df['userId'].unique()
        items = ratings_df['movieId'].unique()
        self.user_map  = {u: i for i, u in enumerate(users)}
        self.item_map  = {m: i for i, m in enumerate(items)}
        self.global_mean = float(ratings_df['rating'].mean())

        n_u = len(users)
        n_i = len(items)
        k   = self.n_factors

        # Use ONLY liked movies (≥ 4.0) as positive training signal.
        # Movies rated < 4.0 are treated as "unknown" — same as unrated.
        # This prevents low-rated movies from being pushed up in the ranking.
        liked = ratings_df[ratings_df['rating'] >= 4.0]
        row_u = liked['userId'].map(self.user_map).values
        col_i = liked['movieId'].map(self.item_map).values
        # confidence excess: C[u,i]-1  (sparse — zero for unrated/disliked items)
        cvals = (self.alpha * liked['rating'].values).astype(np.float32)

        # C_sparse[u,i] = alpha * r_ui for liked items (add 1 analytically during solve)
        C  = csr_matrix((cvals, (row_u, col_i)), shape=(n_u, n_i), dtype=np.float32)
        Ct = C.T.tocsr()

        np.random.seed(42)
        self.U = (np.random.normal(0, 0.01, (n_u, k))).astype(np.float32)
        self.V = (np.random.normal(0, 0.01, (n_i, k))).astype(np.float32)

        lam_I = (self.reg * np.eye(k, dtype=np.float32))

        print(f"  [iALS] {n_u} users · {n_i} items · k={k} · epochs={self.n_epochs}")

        for epoch in range(self.n_epochs):
            # ── Update Users (fix V, solve each u) ──────────────────────
            VtV = self.V.T @ self.V                    # (k,k) — shared

            for u in range(n_u):
                rated_idx  = C[u].indices              # item indices rated by u
                conf_excess = C[u].data                # alpha*r for those items
                if len(rated_idx) == 0:
                    continue
                V_r = self.V[rated_idx]                # (n_rated, k)
                # A = V^T V + V^T diag(c-1) V + reg*I
                A = VtV + (V_r * conf_excess[:, None]).T @ V_r + lam_I
                # b = V^T c  (c = conf_excess+1 because C_full = C_sparse+1)
                b = V_r.T @ (conf_excess + 1.0)
                self.U[u] = scipy_solve(A, b, assume_a='pos')

            # ── Update Items (fix U, solve each i) ──────────────────────
            UtU = self.U.T @ self.U

            for i in range(n_i):
                rater_idx   = Ct[i].indices
                conf_excess = Ct[i].data
                if len(rater_idx) == 0:
                    continue
                U_r = self.U[rater_idx]
                A = UtU + (U_r * conf_excess[:, None]).T @ U_r + lam_I
                b = U_r.T @ (conf_excess + 1.0)
                self.V[i] = scipy_solve(A, b, assume_a='pos')

            print(f"    epoch {epoch+1}/{self.n_epochs} done", flush=True)

    def predict_for_user(self, user_id):
        """Batch scores for all known items — same interface as SVDModel."""
        if user_id not in self.user_map:
            return {}
        u_idx  = self.user_map[user_id]
        scores = self.V @ self.U[u_idx]          # (n_items,)
        return {mid: float(scores[idx]) for mid, idx in self.item_map.items()}

    def predict(self, user_id, movie_id):
        """Single score — .est attribute matches scikit-surprise interface."""
        class _P: pass
        p = _P()
        if user_id not in self.user_map or movie_id not in self.item_map:
            p.est = self.global_mean; return p
        p.est = float(np.dot(self.U[self.user_map[user_id]],
                             self.V[self.item_map[movie_id]]))
        return p


# ─────────────────────────────────────────────────────────────────────────────
# SVD model — kept for comparison / RMSE computation
# ─────────────────────────────────────────────────────────────────────────────
class SVDModel:
    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.U = self.sigma = self.Vt = None
        self.user_map = {}; self.item_map = {}
        self.user_biases = {}; self.item_biases = {}
        self.global_mean = 3.5

    def fit(self, ratings_df):
        users = ratings_df['userId'].unique()
        items = ratings_df['movieId'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}
        self.global_mean = float(ratings_df['rating'].mean())

        user_means = ratings_df.groupby('userId')['rating'].mean()
        item_means = ratings_df.groupby('movieId')['rating'].mean()
        self.user_biases = (user_means - self.global_mean).to_dict()
        self.item_biases = (item_means - self.global_mean).to_dict()

        u_bias = ratings_df['userId'].map(user_means - self.global_mean)
        i_bias = ratings_df['movieId'].map(item_means - self.global_mean)
        vals   = (ratings_df['rating'] - self.global_mean - u_bias - i_bias).values

        rows = ratings_df['userId'].map(self.user_map).values
        cols = ratings_df['movieId'].map(self.item_map).values
        mat  = csr_matrix((vals.astype(np.float32), (rows, cols)),
                          shape=(len(users), len(items)))

        k = min(self.n_factors, min(len(users), len(items)) - 1)
        self.U, self.sigma, self.Vt = svds(mat, k=k)
        self.U = self.U[:, ::-1]; self.sigma = self.sigma[::-1]; self.Vt = self.Vt[::-1, :]

    def predict(self, user_id, movie_id):
        class _P: pass
        p = _P()
        u_bias = self.user_biases.get(user_id, 0.0)
        i_bias = self.item_biases.get(movie_id, 0.0)
        baseline = self.global_mean + u_bias + i_bias
        if user_id not in self.user_map or movie_id not in self.item_map:
            p.est = float(np.clip(baseline, 1.0, 5.0)); return p
        u_vec = self.U[self.user_map[user_id], :] * self.sigma
        i_vec = self.Vt[:, self.item_map[movie_id]]
        p.est = float(np.clip(baseline + np.dot(u_vec, i_vec), 1.0, 5.0))
        return p

    def predict_for_user(self, user_id):
        if user_id not in self.user_map:
            return {}
        u_idx   = self.user_map[user_id]
        u_bias  = self.user_biases.get(user_id, 0.0)
        u_vec   = self.U[u_idx, :] * self.sigma
        latent  = self.Vt.T @ u_vec
        mids    = list(self.item_map.keys())
        idxs    = [self.item_map[m] for m in mids]
        i_bias  = np.array([self.item_biases.get(m, 0.0) for m in mids], dtype=np.float32)
        preds   = np.clip(self.global_mean + u_bias + i_bias + latent[idxs], 1.0, 5.0)
        return dict(zip(mids, preds.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# CollaborativeRecommender — uses iALS by default
# ─────────────────────────────────────────────────────────────────────────────
class CollaborativeRecommender:
    def __init__(self, ratings_path="data/ratings.csv", use_ials=True):
        ratings = load_data(ratings_path)

        # Top-12K most active users for dense, reliable latent factors
        user_counts = ratings.groupby('userId').size()
        top_users   = user_counts.nlargest(12_000).index
        ratings     = ratings[ratings['userId'].isin(top_users)]

        if use_ials:
            # n_factors=128: more expressive than SVD's 100
            # alpha=40: strong confidence push for liked (≥4.0) movies
            # reg=0.05: moderate regularisation to avoid overfitting small train sets
            self.model = IALSModel(n_factors=128, n_epochs=15, alpha=40, reg=0.05)
        else:
            self.model = SVDModel(n_factors=100)

        self.model.fit(ratings)

    def recommend_for_user(self, user_id, movies_df, top_n=10):
        scores = self.model.predict_for_user(user_id)
        preds  = [(mid, scores.get(mid, 0.0)) for mid in movies_df['movieId'].unique()]
        preds.sort(key=lambda x: x[1], reverse=True)
        return [movies_df[movies_df['movieId'] == mid]['title'].values[0]
                for mid, _ in preds[:top_n]]

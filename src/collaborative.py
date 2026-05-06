import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


# -----------------------------
# Load data
# -----------------------------
def load_data(path="data/ratings.csv"):
    ratings = pd.read_csv(path)
    return ratings[['userId', 'movieId', 'rating']]


# -----------------------------
# Scipy SVD model with user + item bias
# drop-in replacement for scikit-surprise (same .predict() interface)
# -----------------------------
class SVDModel:
    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_map = {}
        self.item_map = {}
        self.user_biases = {}   # user_id  → (user_mean − global_mean)
        self.item_biases = {}   # movie_id → (item_mean − global_mean)
        self.global_mean = 3.5

    def fit(self, ratings_df):
        users = ratings_df['userId'].unique()
        items = ratings_df['movieId'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}
        self.global_mean = float(ratings_df['rating'].mean())

        # user bias = user mean − global mean
        user_means = ratings_df.groupby('userId')['rating'].mean()
        self.user_biases = (user_means - self.global_mean).to_dict()

        # item bias = item mean − global mean
        item_means = ratings_df.groupby('movieId')['rating'].mean()
        self.item_biases = (item_means - self.global_mean).to_dict()

        # center matrix by BOTH user and item bias before SVD
        u_bias_col = ratings_df['userId'].map(user_means - self.global_mean)
        i_bias_col = ratings_df['movieId'].map(item_means - self.global_mean)
        vals = (ratings_df['rating'] - self.global_mean - u_bias_col - i_bias_col).values

        rows = ratings_df['userId'].map(self.user_map).values
        cols = ratings_df['movieId'].map(self.item_map).values

        matrix = csr_matrix((vals.astype(np.float32), (rows, cols)),
                            shape=(len(users), len(items)))

        k = min(self.n_factors, min(len(users), len(items)) - 1)
        self.U, self.sigma, self.Vt = svds(matrix, k=k)

        # svds returns ascending order — flip to descending
        self.U = self.U[:, ::-1]
        self.sigma = self.sigma[::-1]
        self.Vt = self.Vt[::-1, :]

    def predict(self, user_id, movie_id):
        """Single prediction. Returns object with .est — matches scikit-surprise interface."""
        class _Pred:
            pass
        p = _Pred()

        u_bias = self.user_biases.get(user_id, 0.0)
        i_bias = self.item_biases.get(movie_id, 0.0)
        baseline = self.global_mean + u_bias + i_bias

        if user_id not in self.user_map or movie_id not in self.item_map:
            p.est = float(np.clip(baseline, 1.0, 5.0))
            return p

        u_idx = self.user_map[user_id]
        i_idx = self.item_map[movie_id]
        user_vec = self.U[u_idx, :] * self.sigma
        item_vec = self.Vt[:, i_idx]
        pred = baseline + float(np.dot(user_vec, item_vec))
        p.est = float(np.clip(pred, 1.0, 5.0))
        return p

    def predict_for_user(self, user_id):
        """Batch prediction for all items in model — ~200× faster than looping predict().

        Returns dict {movie_id: predicted_rating}.
        Movies not in the model are omitted (caller falls back to baseline for them).
        """
        if user_id not in self.user_map:
            return {}

        u_idx = self.user_map[user_id]
        u_bias = self.user_biases.get(user_id, 0.0)

        # one matrix multiply for all n_items
        user_vec = self.U[u_idx, :] * self.sigma    # shape (k,)
        latent_scores = self.Vt.T @ user_vec        # shape (n_items,)

        # fully vectorised — no Python loop over items
        movie_ids = list(self.item_map.keys())
        i_indices = [self.item_map[m] for m in movie_ids]
        i_biases  = np.array([self.item_biases.get(m, 0.0) for m in movie_ids], dtype=np.float32)
        preds = np.clip(
            self.global_mean + u_bias + i_biases + latent_scores[i_indices],
            1.0, 5.0
        )
        return dict(zip(movie_ids, preds.tolist()))


# -----------------------------
# Build recommender class
# -----------------------------
class CollaborativeRecommender:
    def __init__(self, ratings_path="data/ratings.csv"):
        ratings = load_data(ratings_path)

        # Pick the most active users — they have the densest rating history,
        # which gives SVD cleaner latent factors and better item coverage.
        # 12 000 active users × ~200+ ratings ≈ 2-3M training points.
        user_counts = ratings.groupby('userId').size()
        top_users = user_counts.nlargest(12_000).index
        ratings = ratings[ratings['userId'].isin(top_users)]

        self.model = SVDModel(n_factors=100)
        self.model.fit(ratings)

    def recommend_for_user(self, user_id, movies_df, top_n=10):
        svd_scores = self.model.predict_for_user(user_id)
        movie_ids = movies_df['movieId'].unique()
        predictions = [
            (mid, svd_scores.get(mid, self.model.global_mean))
            for mid in movie_ids
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [
            movies_df[movies_df['movieId'] == mid]['title'].values[0]
            for mid, _ in predictions[:top_n]
        ]

import math
import random
import numpy as np
import pandas as pd
from logger import log_run


# ─── ANSI colour helpers ──────────────────────────────────────────────────────
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _col(text, colour): return f"{colour}{text}{_RESET}"
def _bold(text):        return f"{_BOLD}{text}{_RESET}"

def _grade_precision(v):
    if v >= 0.25: return _col("● GOOD",      _GREEN)
    if v >= 0.10: return _col("● FAIR",      _YELLOW)
    return              _col("● WEAK",       _RED)

def _grade_recall(v):
    if v >= 0.05: return _col("● GOOD",      _GREEN)
    if v >= 0.02: return _col("● FAIR",      _YELLOW)
    return              _col("● WEAK",       _RED)

def _grade_ndcg(v):
    if v >= 0.25: return _col("● GOOD",      _GREEN)
    if v >= 0.10: return _col("● FAIR",      _YELLOW)
    return              _col("● WEAK",       _RED)

def _grade_rmse(v):
    if v < 0.80:  return _col("● EXCELLENT", _GREEN)
    if v < 0.95:  return _col("● GOOD",      _YELLOW)
    return              _col("● POOR",       _RED)

def _colour_score(v, good=0.25, fair=0.10):
    if v >= good: return _col(f"{v:.4f}", _GREEN)
    if v >= fair: return _col(f"{v:.4f}", _YELLOW)
    return              _col(f"{v:.4f}", _RED)


# ─── Metric functions ─────────────────────────────────────────────────────────
def precision_at_k(recommended, relevant, k=10):
    return len(set(recommended[:k]) & set(relevant)) / k if k > 0 else 0

def recall_at_k(recommended, relevant, k=10):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k=10):
    relevant_set = set(relevant)
    dcg  = sum(1.0/math.log2(i+2) for i,m in enumerate(recommended[:k]) if m in relevant_set)
    ideal= sum(1.0/math.log2(i+2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0


# ─── RMSE ─────────────────────────────────────────────────────────────────────
def compute_rmse(collab_model, ratings_df, n_samples=20_000):
    """
    Compute RMSE on a held-out sample of ratings for users and movies
    the SVD model was trained on.
    """
    model      = collab_model.model
    known_u    = set(model.user_map.keys())
    known_m    = set(model.item_map.keys())
    test       = ratings_df[
        ratings_df['userId'].isin(known_u) &
        ratings_df['movieId'].isin(known_m)
    ]
    if len(test) > n_samples:
        test = test.sample(n_samples, random_state=42)

    errors = [
        (model.predict(int(r.userId), int(r.movieId)).est - r.rating) ** 2
        for r in test.itertuples()
    ]
    return float(np.sqrt(np.mean(errors))) if errors else float('nan')


# ─── Ground truth helpers ─────────────────────────────────────────────────────
def get_user_relevant_movies(ratings, threshold=4.0):
    filtered = ratings[ratings['rating'] >= threshold]
    return filtered.groupby('userId')['movieId'].apply(list).to_dict()

def get_movie_id_by_title(movies, title):
    row = movies[movies['title'].str.lower().str.strip() == title.strip().lower()]
    return row.iloc[0]['movieId'] if not row.empty else None


# ─── Main evaluation ──────────────────────────────────────────────────────────
def evaluate_model(hybrid_model, ratings, movies, sample_users=30, k=10):

    user_likes  = get_user_relevant_movies(ratings)
    known_users = set(hybrid_model.collab_model.model.user_map.keys())
    eligible    = [u for u, liked in user_likes.items()
                   if len(liked) >= 20 and u in known_users]

    if not eligible:
        print(_col("⚠  No eligible users found.", _RED))
        return 0.0, 0.0, 0.0

    random.seed(42)
    users = random.sample(eligible, min(sample_users, len(eligible)))

    precisions, recalls, ndcgs = [], [], []
    W = 68   # line width

    print(_col(f"\n{'─'*W}", _DIM))
    print(_bold(f"  Evaluating {len(users)} users  (≥20 liked movies, known to SVD)"))
    print(_col(f"{'─'*W}", _DIM))

    for user_id in users:
        liked_movies = user_likes[user_id]
        split        = len(liked_movies) // 2
        train_movies = liked_movies[:split]
        test_movies  = liked_movies[split:]

        print(f"\n  {_col('User', _CYAN)} {user_id}  "
              f"train={len(train_movies)}  test={len(test_movies)}")

        all_recs = []
        for movie_id in train_movies[:5]:
            row = movies[movies['movieId'] == movie_id]
            if row.empty:
                continue
            try:
                recs = hybrid_model.recommend(user_id, row.iloc[0]['title'], top_n=k)
                all_recs.extend(recs)
            except Exception as e:
                print(f"    {_col('error:', _RED)} {e}")

        rec_ids = []
        seen    = set()
        for title in all_recs:
            mid = get_movie_id_by_title(movies, title)
            if mid and mid not in seen:
                seen.add(mid)
                rec_ids.append(mid)

        if not rec_ids:
            print(f"    {_col('no recommendations generated', _RED)}")
            continue

        p = precision_at_k(rec_ids, test_movies, k)
        r = recall_at_k(rec_ids, test_movies, k)
        n = ndcg_at_k(rec_ids, test_movies, k)

        # colour the per-user precision
        p_str = _colour_score(p, good=0.30, fair=0.10)
        r_str = _colour_score(r, good=0.05, fair=0.02)
        n_str = _colour_score(n, good=0.30, fair=0.10)
        print(f"    Precision@{k}: {p_str}   Recall@{k}: {r_str}   NDCG@{k}: {n_str}")

        precisions.append(p); recalls.append(r); ndcgs.append(n)

    if not precisions:
        print(_col("\n  No valid evaluations completed.", _RED))
        return 0.0, 0.0, 0.0

    return (sum(precisions)/len(precisions),
            sum(recalls)   /len(recalls),
            sum(ndcgs)     /len(ndcgs))


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print(_bold(_col("\n  🎬  MOVIE RECOMMENDATION SYSTEM — EVALUATION", _CYAN)))

    movies  = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    from content_based  import ContentBasedRecommender
    from collaborative  import CollaborativeRecommender
    from hybrid         import HybridRecommender

    content_model = ContentBasedRecommender()
    collab_model  = CollaborativeRecommender()
    hybrid_model  = HybridRecommender(content_model, collab_model, movies)

    print(_bold("\n  SANITY TEST — 'Toy Story' recommendations:"))
    for i, m in enumerate(hybrid_model.recommend(1, "Toy Story", top_n=5), 1):
        print(f"    {i}. {m}")

    # ── Recommendation quality ────────────────────────────────────────────────
    precision, recall, ndcg = evaluate_model(
        hybrid_model, ratings, movies, sample_users=30, k=10
    )

    # ── Rating accuracy (RMSE) ────────────────────────────────────────────────
    # iALS produces ranking scores, not rating values — RMSE needs SVD.
    # We train a lightweight SVD alongside iALS purely for RMSE reporting.
    print(_col(f"\n  {'─'*68}", _DIM))
    print("  Computing RMSE (SVD rating predictor)…")
    from collaborative import CollaborativeRecommender as CR
    svd_model_for_rmse = CR(use_ials=False)
    rmse = compute_rmse(svd_model_for_rmse, ratings)

    # ── Print final dashboard ─────────────────────────────────────────────────
    log_run(precision, recall, ndcg, rmse, n_users=30)

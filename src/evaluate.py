import math
import random
import pandas as pd
from logger import log_run


# -----------------------------
# Precision@K  (standard: hits / k)
# -----------------------------
def precision_at_k(recommended, relevant, k=10):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k if k > 0 else 0


# -----------------------------
# Recall@K
# -----------------------------
def recall_at_k(recommended, relevant, k=10):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant) if relevant else 0


# -----------------------------
# NDCG@K
# -----------------------------
def ndcg_at_k(recommended, relevant, k=10):
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, mid in enumerate(recommended[:k]) if mid in relevant_set
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0


# -----------------------------
# Build Ground Truth (FAST)
# -----------------------------
def get_user_relevant_movies(ratings, threshold=4.0):
    filtered = ratings[ratings['rating'] >= threshold]
    return filtered.groupby('userId')['movieId'].apply(list).to_dict()


# -----------------------------
# Safe title → movieId mapping
# -----------------------------
def get_movie_id_by_title(movies, title):
    title = title.strip().lower()
    row = movies[movies['title'].str.lower().str.strip() == title]

    if row.empty:
        return None

    return row.iloc[0]['movieId']


# -----------------------------
# Evaluate Hybrid Model (FIXED)
# -----------------------------
def evaluate_model(hybrid_model, ratings, movies, sample_users=20, k=10):

    user_likes = get_user_relevant_movies(ratings)

    # Only evaluate users the SVD model was trained on — otherwise predictions degrade
    # to global mean and scores become meaningless.
    known_users = set(hybrid_model.collab_model.model.user_map.keys())
    # Require ≥20 liked movies so the 50% train split gives ≥10 training examples.
    # Sparse users (2-6 liked movies) produce unreliable SVD vectors and inflate 0.00 scores.
    eligible = [u for u, liked in user_likes.items()
                if len(liked) >= 20 and u in known_users]

    if not eligible:
        print("⚠️  No known users found — increase training sample in collaborative.py")
        return 0, 0, 0

    random.seed(42)
    users = random.sample(eligible, min(sample_users, len(eligible)))

    precisions, recalls, ndcgs = [], [], []

    for user_id in users:

        print(f"\n🔍 Evaluating User: {user_id}")

        liked_movies = user_likes[user_id]

        split = len(liked_movies) // 2
        train_movies = liked_movies[:split]
        test_movies = liked_movies[split:]

        print(f"Train size: {len(train_movies)} | Test size: {len(test_movies)}")

        all_recommendations = []

        for movie_id in train_movies[:5]:
            movie_row = movies[movies['movieId'] == movie_id]
            if movie_row.empty:
                continue
            movie_title = movie_row.iloc[0]['title']
            print("Input movie:", movie_title)
            try:
                recs = hybrid_model.recommend(user_id, movie_title, top_n=k)
                all_recommendations.extend(recs)
            except Exception as e:
                print("Error:", e)
                continue

        # Convert recommended titles → movieIds
        rec_ids = []
        for title in all_recommendations:
            movie_id = get_movie_id_by_title(movies, title)
            if movie_id:
                rec_ids.append(movie_id)

        if not rec_ids:
            print("No valid recommendations")
            continue

        # deduplicate preserving rank order so Precision@10 sees distinct movies
        seen = set()
        unique_rec_ids = []
        for mid in rec_ids:
            if mid not in seen:
                seen.add(mid)
                unique_rec_ids.append(mid)
        rec_ids = unique_rec_ids

        p = precision_at_k(rec_ids, test_movies, k)
        r = recall_at_k(rec_ids, test_movies, k)
        n = ndcg_at_k(rec_ids, test_movies, k)
        print(f"  Precision@{k}: {p:.4f}  Recall@{k}: {r:.4f}  NDCG@{k}: {n:.4f}")

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

    if not precisions:
        print("\n❌ No valid evaluations done")
        return 0, 0, 0

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_n = sum(ndcgs) / len(ndcgs)

    print(f"\nEvaluated {len(precisions)} users")
    return avg_p, avg_r, avg_n


# -----------------------------
# Run Evaluation
# -----------------------------
if __name__ == "__main__":

    print("🚀 Starting Evaluation...")

    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    from content_based import ContentBasedRecommender
    from collaborative import CollaborativeRecommender
    from hybrid import HybridRecommender

    content_model = ContentBasedRecommender()
    collab_model = CollaborativeRecommender()
    hybrid_model = HybridRecommender(content_model, collab_model, movies)

    # sanity test
    print("\n=== SANITY TEST ===")
    print(hybrid_model.recommend(1, "Toy Story", top_n=5))

    # evaluation
    precision, recall, ndcg = evaluate_model(hybrid_model, ratings, movies, sample_users=30, k=10)

    print(f"\n🎯 Final Precision@10 : {precision:.4f}")
    print(f"🎯 Final Recall@10    : {recall:.4f}")
    print(f"🎯 Final NDCG@10      : {ndcg:.4f}")

    log_run(precision, recall, ndcg, n_users=30)

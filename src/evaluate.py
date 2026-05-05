import pandas as pd
from collections import defaultdict


# -----------------------------
# Precision@K
# -----------------------------
def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / min(k, len(relevant)) if relevant else 0


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
def evaluate_model(hybrid_model, ratings, movies, sample_users=5, k=10):

    user_likes = get_user_relevant_movies(ratings)
    precisions = []

    users = list(user_likes.keys())[:sample_users]

    for user_id in users:

        print(f"\n🔍 Evaluating User: {user_id}")

        liked_movies = user_likes[user_id]

        if len(liked_movies) < 4:
            print("Skipped: not enough data")
            continue

        # 🔥 TRAIN-TEST SPLIT
        split = len(liked_movies) // 2
        train_movies = liked_movies[:split]
        test_movies = liked_movies[split:]

        print("Train size:", len(train_movies), "| Test size:", len(test_movies))

        all_recommendations = []

        # Use multiple training movies as input
        for movie_id in train_movies[:3]:

            movie_row = movies[movies['movieId'] == movie_id]

            if movie_row.empty:
                continue

            movie_title = movie_row.iloc[0]['title']

            print("Input movie:", movie_title)

            try:
                recs = hybrid_model.recommend(user_id, movie_title, top_n=k)
                print("Recs:", recs)
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

        if len(rec_ids) == 0:
            print("No valid recommendations")
            continue

        precision = precision_at_k(rec_ids, test_movies, k)
        print("Precision:", precision)

        precisions.append(precision)

    if len(precisions) == 0:
        print("\n❌ No valid evaluations done")
        return 0

    print("\nAll precision scores:", precisions)

    return sum(precisions) / len(precisions)


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
    score = evaluate_model(hybrid_model, ratings, movies, sample_users=5, k=10)

    print(f"\n🎯 Final Precision@10: {score:.4f}")
    # print(collab_model.model.predict(1, 1))
    # print(collab_model.model.predict(1, 10))
    # print(collab_model.model.predict(2, 10))

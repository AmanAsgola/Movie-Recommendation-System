import pandas as pd
from surprise import Dataset, Reader, SVD

# -----------------------------
# Load data
# -----------------------------
def load_data(path="data/ratings.csv"):
    ratings = pd.read_csv(path)
    ratings = ratings[['userId', 'movieId', 'rating']]
    return ratings

# -----------------------------
# Train model (call ONCE)
# -----------------------------
def train_model(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings, reader)

    trainset = data.build_full_trainset()

    model = SVD(n_factors=30, n_epochs=5)
    model.fit(trainset)

    return model, trainset

# -----------------------------
# Build recommender class
# -----------------------------
class CollaborativeRecommender:
    def __init__(self, ratings_path="data/ratings.csv"):
        self.ratings = load_data(ratings_path)

        # optional speed optimization
        self.ratings = self.ratings.sample(30000, random_state=42)

        self.model, self.trainset = train_model(self.ratings)

    def recommend_for_user(self, user_id, movies_df, top_n=10):

        movie_ids = movies_df['movieId'].unique()

        predictions = []

        for movie_id in movie_ids:
            pred = self.model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))

        # sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        top_movies = predictions[:top_n]

        recommended_movies = [
            movies_df[movies_df['movieId'] == mid]['title'].values[0]
            for mid, _ in top_movies
        ]

        return recommended_movies

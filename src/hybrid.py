class HybridRecommender:
    def __init__(self, content_model, collab_model, movies):
        self.content_model = content_model
        self.collab_model = collab_model
        self.movies = movies

    # -----------------------------
    # title → movieId
    # -----------------------------
    def get_movie_id(self, title):
        row = self.movies[self.movies['title'].str.lower().str.strip() == title.lower().strip()]
        if row.empty:
            return None
        return row.iloc[0]['movieId']

    # -----------------------------
    # movieId → title
    # -----------------------------
    def get_title(self, movie_id):
        row = self.movies[self.movies['movieId'] == movie_id]
        if row.empty:
            return None
        return row.iloc[0]['title']

    # -----------------------------
    # Hybrid Recommendation (FIXED LOGIC)
    # -----------------------------
    def recommend(self, user_id, movie_name, top_n=10):

        # STEP 1: Get all movies
        all_movie_ids = self.movies['movieId'].values

        # STEP 2: SVD ranking (PRIMARY SIGNAL)
        scores = {}

        for movie_id in all_movie_ids:
            pred = self.collab_model.model.predict(user_id, movie_id)
            scores[movie_id] = pred.est

        # STEP 3: Get content-based similar movies (BOOST SET)
        content_results = self.content_model.recommend(movie_name, top_n=50)

        content_set = set()

        for title in content_results:
            mid = self.get_movie_id(title)
            if mid:
                content_set.add(mid)

        # STEP 4: Apply hybrid boost
        final_scores = {}

        for movie_id, score in scores.items():

            boost = 1.2 if movie_id in content_set else 1.0
            final_scores[movie_id] = score * boost

        # STEP 5: Sort final results
        sorted_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        # STEP 6: Convert to titles
        recommendations = []

        for movie_id, _ in sorted_movies[:top_n]:
            title = self.get_title(movie_id)
            if title:
                recommendations.append(title)

        return recommendations

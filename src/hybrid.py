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

        # Cold-start: unknown user → pure content-based
        if user_id not in self.collab_model.model.user_map:
            return self.content_model.recommend(movie_name, top_n=top_n)

        # STEP 1: Batch SVD predictions — only returns movies the model knows
        svd_scores = self.collab_model.model.predict_for_user(user_id)

        # Normalise ONLY over SVD-known movies so unknown movies don't contaminate
        # the score range. Unknown movies get 0.0 base (below all SVD predictions).
        if svd_scores:
            min_s = min(svd_scores.values())
            max_s = max(svd_scores.values())
            score_range = max_s - min_s or 1.0
            norm_svd = {mid: (s - min_s) / score_range for mid, s in svd_scores.items()}
        else:
            norm_svd = {}

        all_movie_ids = self.movies['movieId'].values
        # Known movies → normalised SVD score  |  Unknown → 0.0
        base_scores = {mid: norm_svd.get(mid, 0.0) for mid in all_movie_ids}

        # STEP 2: Content scores using actual cosine similarity (not linear rank decay).
        # A 0.92-similar movie gets grade 0.92; a 0.55-similar movie gets 0.55.
        # Much more precise than (100-rank)/100 which treats rank-1 and rank-50 the same.
        content_scored = self.content_model.recommend_scored(movie_name, top_n=150)
        content_grades = {}
        for title, similarity in content_scored:
            mid = self.get_movie_id(title)
            if mid:
                content_grades[mid] = similarity

        # STEP 3: Weighted additive fusion  (SVD 70% + content 30%)
        # SVD-known movies score up to 0.7 + 0.3 = 1.0
        # Content-only unknowns score up to 0.0 + 0.3 = 0.3  (below SVD top movies)
        alpha, beta = 0.7, 0.3
        final_scores = {
            mid: alpha * base_scores[mid] + beta * content_grades.get(mid, 0.0)
            for mid in all_movie_ids
        }

        # STEP 4: Sort and return top-N titles
        sorted_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for movie_id, _ in sorted_movies[:top_n]:
            title = self.get_title(movie_id)
            if title:
                recommendations.append(title)
        return recommendations

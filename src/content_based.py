import pandas as pd
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ContentBasedRecommender:
    def __init__(self, movies_path="data/movies.csv"):
        self.movies = pd.read_csv(movies_path)

        # -----------------------------
        # Clean data
        # -----------------------------
        self.movies = self.movies.dropna()
        self.movies = self.movies.drop_duplicates(subset="title")

        # -----------------------------
        # Feature engineering
        # -----------------------------
        self.movies['clean_title'] = (
            self.movies['title']
            .str.replace(r"\(\d{4}\)", "", regex=True)
            .str.strip()
        )

        # Genres repeated 2× so genre similarity dominates over title word overlap.
        # A Sci-Fi/Action movie matches another Sci-Fi/Action far more strongly than
        # two films that merely share a word in their titles.
        genres_clean = self.movies['genres'].fillna('').str.replace('|', ' ', regex=False)
        self.movies['features'] = (
            genres_clean + ' ' + genres_clean          # 2× genre weight
            + ' ' + self.movies['clean_title'].fillna('')
        )

        # -----------------------------
        # TF-IDF Vectorization
        # -----------------------------
        # max_df=0.8  → drops terms in >80 % of docs (e.g. "Drama" alone is near-useless)
        # max_features → keeps vocabulary focused on most discriminative terms
        # sublinear_tf → log-damps term frequency so rare genres aren't buried
        # ngram_range  → bigrams capture "Sci Fi", "Road Movie" combos
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            max_features=8000,
        )
        self.feature_matrix = self.vectorizer.fit_transform(self.movies['features'])

        # -----------------------------
        # Nearest Neighbours
        # algorithm='brute' is correct for sparse TF-IDF + cosine metric
        # (ball_tree / kd_tree do not support cosine on sparse matrices).
        # n_jobs=-1 uses all CPU cores for distance computation.
        # -----------------------------
        self.model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_jobs=-1,
        )
        self.model.fit(self.feature_matrix)

        # -----------------------------
        # Mapping title → row index
        # -----------------------------
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies['title']
        ).drop_duplicates()

    # -----------------------------
    # Helper: fuzzy title match
    # -----------------------------
    def _get_movie_title(self, user_input):
        titles = self.movies['clean_title'].tolist()
        match, score, _ = process.extractOne(user_input, titles)
        if score < 75:
            return None
        return self.movies[self.movies['clean_title'] == match]['title'].iloc[0]

    # -----------------------------
    # Scored recommendation — returns [(title, cosine_similarity), ...]
    # cosine_similarity = 1 − cosine_distance  ∈ [0, 1]
    # Used by HybridRecommender for accurate content-grade weighting.
    # -----------------------------
    def recommend_scored(self, movie_name, top_n=10):
        movie_title = self._get_movie_title(movie_name)
        if movie_title is None:
            return []

        idx = self.indices[movie_title]
        movie_vector = self.feature_matrix[idx]

        distances, indices_nn = self.model.kneighbors(
            movie_vector, n_neighbors=top_n + 1
        )

        # Remove the query movie itself (index 0)
        distances   = distances.flatten()[1:]
        indices_nn  = indices_nn.flatten()[1:]

        similarities = 1.0 - distances          # cosine similarity ∈ [0, 1]
        titles = self.movies['title'].iloc[indices_nn].tolist()

        return list(zip(titles, similarities.tolist()))

    # -----------------------------
    # Plain title list — backward-compatible for cold-start fallback
    # -----------------------------
    def recommend(self, movie_name, top_n=10):
        scored = self.recommend_scored(movie_name, top_n)
        if not scored:
            return ["Movie not found"]
        return [title for title, _ in scored]

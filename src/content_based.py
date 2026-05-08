import os
import numpy as np
import pandas as pd
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse


METADATA_PATH = "data/movies_metadata.csv"


class ContentBasedRecommender:
    """
    Two-stage retrieval when metadata is available:

    Stage 1 (Genre model, all 27K movies):
        Broad genre-based retrieval → top-300 genre-similar candidates.
        Uses genres × 2 + title. All movies covered equally.

    Stage 2 (Rich model, metadata movies only):
        Re-ranks Stage 1 candidates using overview + keywords + cast + director.
        Movies without metadata keep their Stage-1 genre similarity score.

    This prevents the inconsistency where movies WITH metadata cluster away
    from movies WITHOUT metadata inside a single TF-IDF space.
    """

    def __init__(self, movies_path="data/movies.csv"):
        self.movies = pd.read_csv(movies_path).dropna().drop_duplicates(subset="title")

        self.movies['clean_title'] = (
            self.movies['title']
            .str.replace(r"\(\d{4}\)", "", regex=True)
            .str.strip()
        )

        self.has_metadata = os.path.exists(METADATA_PATH)

        # ── Stage 1: genre model (all movies) ────────────────────
        self._build_genre_model()

        # ── Stage 2: rich metadata model (movies with metadata) ──
        if self.has_metadata:
            self._build_rich_model()
            print(f"[ContentBasedRecommender] mode=two-stage  "
                  f"genre_vocab={len(self.genre_vec.vocabulary_)}  "
                  f"rich_movies={len(self.meta_df)}  "
                  f"total={len(self.movies)}")
        else:
            print(f"[ContentBasedRecommender] mode=genre-only  "
                  f"Run python src/fetch_metadata.py to enable two-stage retrieval.")

        # ── title → row index ─────────────────────────────────────
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies['title']
        ).drop_duplicates()

    # ─────────────────────────────────────────────────────────────
    # Stage 1 — Genre model
    # ─────────────────────────────────────────────────────────────
    def _build_genre_model(self):
        genres = self.movies['genres'].fillna('').str.replace('|', ' ', regex=False)
        features = genres + ' ' + genres + ' ' + self.movies['clean_title'].fillna('')

        self.genre_vec = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            max_features=8000,
        )
        self.genre_matrix = self.genre_vec.fit_transform(features)

        self.genre_model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_jobs=-1
        )
        self.genre_model.fit(self.genre_matrix)

    # ─────────────────────────────────────────────────────────────
    # Stage 2 — Rich metadata model
    # ─────────────────────────────────────────────────────────────
    def _build_rich_model(self):
        meta = pd.read_csv(METADATA_PATH).drop_duplicates('movieId')
        meta = meta[['movieId', 'overview', 'keywords', 'cast', 'director']].fillna('')

        self.meta_df = self.movies.merge(meta, on='movieId', how='inner')

        genres = self.meta_df['genres'].fillna('').str.replace('|', ' ', regex=False)

        # Weight: genres × 4 ensures genre similarity still dominates even
        # for movies with long overviews, keeping Stage-1 and Stage-2 comparable.
        self.meta_df['rich_features'] = (
            genres + ' ' + genres + ' ' + genres + ' ' + genres
            + ' ' + self.meta_df['keywords'] + ' ' + self.meta_df['keywords']
            + ' ' + self.meta_df['overview']
            + ' ' + self.meta_df['cast']
            + ' ' + self.meta_df['director']
        )

        self.rich_vec = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            max_features=15000,
        )
        self.rich_matrix = self.rich_vec.fit_transform(self.meta_df['rich_features'])

        self.rich_model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_jobs=-1
        )
        self.rich_model.fit(self.rich_matrix)

        # movieId → row index in rich_matrix
        self.rich_idx = {
            mid: i for i, mid in enumerate(self.meta_df['movieId'].tolist())
        }

    # ─────────────────────────────────────────────────────────────
    # Fuzzy title match
    # ─────────────────────────────────────────────────────────────
    def _get_movie_title(self, user_input):
        titles = self.movies['clean_title'].tolist()
        match, score, _ = process.extractOne(user_input, titles)
        if score < 75:
            return None
        return self.movies[self.movies['clean_title'] == match]['title'].iloc[0]

    # ─────────────────────────────────────────────────────────────
    # Core recommendation — two-stage if metadata available
    # ─────────────────────────────────────────────────────────────
    def recommend_scored(self, movie_name, top_n=10):
        """Returns [(title, similarity_score), …] sorted by descending score."""
        movie_title = self._get_movie_title(movie_name)
        if movie_title is None:
            return []

        idx = self.indices[movie_title]
        movie_id = self.movies.loc[idx, 'movieId']

        # ── Stage 1: broad genre retrieval ───────────────────────
        genre_vector = self.genre_matrix[idx]
        stage1_n = min(top_n * 30, len(self.movies))   # retrieve wide pool

        distances, indices_nn = self.genre_model.kneighbors(
            genre_vector, n_neighbors=stage1_n + 1
        )
        distances  = distances.flatten()[1:]
        indices_nn = indices_nn.flatten()[1:]

        # {row_in_movies_df: genre_similarity}
        genre_sims = {
            int(indices_nn[i]): float(1.0 - distances[i])
            for i in range(len(indices_nn))
        }

        if not self.has_metadata or movie_id not in self.rich_idx:
            # No metadata — return top_n by genre similarity
            sorted_rows = sorted(genre_sims.items(), key=lambda x: x[1], reverse=True)
            results = []
            for row, sim in sorted_rows[:top_n]:
                title = self.movies.iloc[row]['title']
                results.append((title, sim))
            return results

        # ── Stage 2: rich re-ranking within Stage-1 pool ─────────
        rich_idx = self.rich_idx[movie_id]
        rich_vector = self.rich_matrix[rich_idx]

        # Only re-rank candidates that have rich metadata
        candidates_with_meta = []
        for row_idx in genre_sims:
            cand_mid = self.movies.iloc[row_idx]['movieId']
            if cand_mid in self.rich_idx:
                candidates_with_meta.append((row_idx, cand_mid))

        # Compute rich similarity for metadata candidates
        rich_sims = {}
        if candidates_with_meta:
            cand_rich_rows = np.array([self.rich_idx[mid] for _, mid in candidates_with_meta])
            cand_vectors = self.rich_matrix[cand_rich_rows]
            dots = cand_vectors.dot(rich_vector.T)
            if issparse(dots):
                dots = dots.toarray().flatten()
            else:
                dots = np.array(dots).flatten()

            norms_cand  = np.array(cand_vectors.power(2).sum(axis=1)).flatten() ** 0.5
            norm_query  = float(rich_vector.power(2).sum() ** 0.5) or 1.0
            norms_cand  = np.where(norms_cand == 0, 1.0, norms_cand)
            cosine_rich = dots / (norms_cand * norm_query)

            for i, (row_idx, _) in enumerate(candidates_with_meta):
                rich_sims[row_idx] = float(np.clip(cosine_rich[i], 0, 1))

        # ── Combine: genre base + additive rich bonus ────────────────
        # Additive (not averaging) ensures metadata never HURTS a movie:
        #   with metadata    → genre_sim + 0.4 × rich_sim  (max ≈ 1.4)
        #   without metadata → genre_sim                    (max ≈ 1.0)
        # Movies without metadata keep their full genre score, so they
        # are not penalised for missing TMDB data.
        final_scores = {}
        for row_idx, g_sim in genre_sims.items():
            r_sim = rich_sims.get(row_idx, None)
            if r_sim is not None:
                final_scores[row_idx] = g_sim + 0.4 * r_sim
            else:
                final_scores[row_idx] = g_sim

        sorted_rows = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for row, score in sorted_rows[:top_n]:
            title = self.movies.iloc[row]['title']
            results.append((title, score))
        return results

    def recommend(self, movie_name, top_n=10):
        scored = self.recommend_scored(movie_name, top_n)
        if not scored:
            return ["Movie not found"]
        return [title for title, _ in scored]

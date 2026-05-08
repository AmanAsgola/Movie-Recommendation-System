"""
Content-Based Recommender — Sentence-BERT two-stage retrieval.

Stage 1  (all 27 K movies, genre text):
    Encode "Genres: Action Adventure. Title: Toy Story" with all-MiniLM-L6-v2.
    Dense 384-dim semantic vectors → cosine NearestNeighbors.
    Captures genre semantics: "psychological thriller" ≈ "dark crime drama".

Stage 2  (movies with TMDB metadata, rich text):
    Encode "Genres: … Plot: … Cast: … Director: … Keywords: …"
    Additive cosine-similarity boost on top of Stage 1 genre score.
    TF-IDF would miss: "heist comedy" ≈ "caper film" — SBERT gets it right.

Embeddings are cached to data/ so encoding only runs once (~60s first time).
"""

import os
import numpy as np
import pandas as pd
from rapidfuzz import process
from sklearn.neighbors import NearestNeighbors

METADATA_PATH      = "data/movies_metadata.csv"
CACHE_GENRE        = "data/sbert_genre_embeddings.npy"
CACHE_GENRE_IDS    = "data/sbert_genre_ids.npy"
CACHE_RICH         = "data/sbert_rich_embeddings.npy"
CACHE_RICH_IDS     = "data/sbert_rich_ids.npy"
SBERT_MODEL        = "all-MiniLM-L6-v2"   # 80 MB, 384-dim, fast + accurate


def _get_device():
    """Return the best available device string for sentence-transformers."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"     # Apple M-series GPU
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _load_sbert():
    from sentence_transformers import SentenceTransformer
    device = _get_device()
    print(f"  [SBERT] Loading {SBERT_MODEL} on device={device}")
    return SentenceTransformer(SBERT_MODEL, device=device)


def _encode(model, texts, batch_size=256, desc="encoding"):
    print(f"  [SBERT] {desc} {len(texts)} texts …", flush=True)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-norm → cosine = dot product
        convert_to_numpy=True,
    )


class ContentBasedRecommender:
    def __init__(self, movies_path="data/movies.csv"):
        self.movies = pd.read_csv(movies_path).dropna().drop_duplicates(subset="title")
        self.movies['clean_title'] = (
            self.movies['title']
            .str.replace(r"\(\d{4}\)", "", regex=True)
            .str.strip()
        )
        self.has_metadata = os.path.exists(METADATA_PATH)

        # ── Stage 1: genre SBERT embeddings (all movies) ─────────────────────
        self._build_genre_model()

        # ── Stage 2: rich SBERT embeddings (metadata movies) ─────────────────
        if self.has_metadata:
            self._build_rich_model()
            print(f"  [ContentModel] mode=SBERT-two-stage  "
                  f"genre={len(self.movies)}  rich={len(self.meta_df)}")
        else:
            print("  [ContentModel] mode=SBERT-genre-only  "
                  "Run src/fetch_metadata.py for rich mode.")

        # ── title → row index ─────────────────────────────────────────────────
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies['title']
        ).drop_duplicates()

    # ─── Stage 1 ─────────────────────────────────────────────────────────────
    def _build_genre_model(self):
        movie_ids = self.movies['movieId'].values

        if (os.path.exists(CACHE_GENRE) and os.path.exists(CACHE_GENRE_IDS)
                and np.load(CACHE_GENRE_IDS).shape[0] == len(movie_ids)):
            print("  [SBERT] Loading cached genre embeddings …")
            embs = np.load(CACHE_GENRE)
        else:
            genres = (
                self.movies['genres'].fillna('')
                    .str.replace('|', ' ', regex=False)
            )
            texts = [
                f"Genres: {g}. Title: {t}"
                for g, t in zip(genres, self.movies['clean_title'].fillna(''))
            ]
            model = _load_sbert()
            embs  = _encode(model, texts, desc="genre embeddings")
            np.save(CACHE_GENRE,     embs)
            np.save(CACHE_GENRE_IDS, movie_ids)
            print(f"  [SBERT] Cached genre embeddings → {CACHE_GENRE}")

        self.genre_embs  = embs                    # (n_movies, 384)
        self.genre_model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_jobs=-1
        )
        self.genre_model.fit(embs)

    # ─── Stage 2 ─────────────────────────────────────────────────────────────
    def _build_rich_model(self):
        meta = pd.read_csv(METADATA_PATH).drop_duplicates('movieId')
        meta = meta[['movieId', 'overview', 'keywords', 'cast', 'director']].fillna('')
        self.meta_df = self.movies.merge(meta, on='movieId', how='inner')

        meta_ids = self.meta_df['movieId'].values

        if (os.path.exists(CACHE_RICH) and os.path.exists(CACHE_RICH_IDS)
                and np.load(CACHE_RICH_IDS).shape[0] == len(meta_ids)):
            print("  [SBERT] Loading cached rich embeddings …")
            embs = np.load(CACHE_RICH)
        else:
            genres = (
                self.meta_df['genres'].fillna('')
                    .str.replace('|', ' ', regex=False)
            )
            texts = [
                f"Genres: {g}. Plot: {ov}. Cast: {ca}. Director: {di}. Keywords: {kw}."
                for g, ov, ca, di, kw in zip(
                    genres,
                    self.meta_df['overview'],
                    self.meta_df['cast'],
                    self.meta_df['director'],
                    self.meta_df['keywords'],
                )
            ]
            model = _load_sbert()
            embs  = _encode(model, texts, desc="rich embeddings")
            np.save(CACHE_RICH,     embs)
            np.save(CACHE_RICH_IDS, meta_ids)
            print(f"  [SBERT] Cached rich embeddings → {CACHE_RICH}")

        self.rich_embs = embs                      # (n_meta, 384)
        self.rich_model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_jobs=-1
        )
        self.rich_model.fit(embs)
        # movieId → row index in rich_embs
        self.rich_idx = {mid: i for i, mid in enumerate(meta_ids)}

    # ─── Fuzzy title match ────────────────────────────────────────────────────
    def _get_movie_title(self, user_input):
        titles = self.movies['clean_title'].tolist()
        match, score, _ = process.extractOne(user_input, titles)
        if score < 75:
            return None
        return self.movies[self.movies['clean_title'] == match]['title'].iloc[0]

    # ─── Core recommendation ─────────────────────────────────────────────────
    def recommend_scored(self, movie_name, top_n=10):
        """Returns [(title, similarity_score), …] — used by HybridRecommender."""
        movie_title = self._get_movie_title(movie_name)
        if movie_title is None:
            return []

        idx      = self.indices[movie_title]
        movie_id = self.movies.loc[idx, 'movieId']

        # Stage 1 — broad genre retrieval (all movies, SBERT genre embeddings)
        query_vec  = self.genre_embs[idx].reshape(1, -1)
        pool_n     = min(top_n * 30, len(self.movies))
        dists, nns = self.genre_model.kneighbors(query_vec, n_neighbors=pool_n + 1)
        dists      = dists.flatten()[1:]
        nns        = nns.flatten()[1:]
        genre_sims = {int(nns[i]): float(1.0 - dists[i]) for i in range(len(nns))}

        if not self.has_metadata or movie_id not in self.rich_idx:
            # No rich metadata — return top_n by genre similarity
            sorted_rows = sorted(genre_sims.items(), key=lambda x: x[1], reverse=True)
            return [(self.movies.iloc[r]['title'], s) for r, s in sorted_rows[:top_n]]

        # Stage 2 — semantic rich re-ranking within Stage-1 pool
        r_idx    = self.rich_idx[movie_id]
        rich_vec = self.rich_embs[r_idx]

        # Compute rich similarity for candidates that have metadata
        rich_sims = {}
        cand_rich_rows, cand_movie_rows = [], []
        for row_idx in genre_sims:
            cand_mid = self.movies.iloc[row_idx]['movieId']
            if cand_mid in self.rich_idx:
                cand_rich_rows.append(self.rich_idx[cand_mid])
                cand_movie_rows.append(row_idx)

        if cand_rich_rows:
            cand_vecs   = self.rich_embs[cand_rich_rows]   # (N, 384), unit-normed
            cosine_rich = cand_vecs @ rich_vec              # dot product = cosine similarity
            for i, row_idx in enumerate(cand_movie_rows):
                rich_sims[row_idx] = float(np.clip(cosine_rich[i], 0, 1))

        # Additive boost: genre_sim + 0.4 × rich_sim (metadata never hurts)
        final_scores = {
            row_idx: (genre_sims[row_idx] + 0.4 * rich_sims[row_idx])
                     if row_idx in rich_sims
                     else genre_sims[row_idx]
            for row_idx in genre_sims
        }

        sorted_rows = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.movies.iloc[r]['title'], s) for r, s in sorted_rows[:top_n]]

    def recommend(self, movie_name, top_n=10):
        scored = self.recommend_scored(movie_name, top_n)
        return [t for t, _ in scored] if scored else ["Movie not found"]

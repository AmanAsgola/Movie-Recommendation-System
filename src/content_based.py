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

        # Combine features (better than genres alone) but Not better 
        self.movies['features'] = self.movies['genres'].fillna('')


        # -----------------------------
        # TF-IDF Vectorization
        # -----------------------------
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.feature_matrix = self.vectorizer.fit_transform(self.movies['features'])


        # -----------------------------
        # Introducing nearest Neighbor
        # -----------------------------

        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.feature_matrix)

        # -----------------------------
        # Mapping title -> index
        # -----------------------------
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies['title']
        ).drop_duplicates()

    # -----------------------------
    # Helper: find closest match
    # -----------------------------
    def _get_movie_title(self, user_input):
        titles = self.movies['clean_title'].tolist()

        match, score, _ = process.extractOne(user_input, titles)

        #  critical: reject weak matches
        # print("Debug Match:", match, "SCORE:", score)
        if score < 75:
            return None

        return self.movies[self.movies['clean_title'] == match]['title'].iloc[0]    

    # -----------------------------
    # Recommendation function
    # -----------------------------
    def recommend(self, movie_name, top_n=10):

        movie_title = self._get_movie_title(movie_name)

        if movie_title is None:
            return ["Movie not found"]

        idx = self.indices[movie_title]

        # reshape ensures correct vector handling
        movie_vector = self.feature_matrix[idx]

        # get nearest neighbors
        distances, indices_nn = self.model.kneighbors(movie_vector, n_neighbors=top_n + 1)

        indices_nn = indices_nn.flatten()[1:]  # remove itself

        return self.movies['title'].iloc[indices_nn].tolist()

import streamlit as st
import pandas as pd
import requests

from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender


# -----------------------------
# Poster Fetch
# -----------------------------
def fetch_poster(movie_name):
    try:
        api_key = "3efac53b49bf40e722929616edeea69f"
        clean_name = movie_name.split("(")[0].strip()

        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_name}"

        data = requests.get(url, timeout=5).json()

        if data.get('results') and len(data['results']) > 0:

            for movie in data['results']:
                if movie.get('poster_path'):
                    return f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"

        return None

    except:
        return None


# -----------------------------
# CACHE MODELS (ONLY HERE)
# -----------------------------
@st.cache_resource
def load_models():
    movies = pd.read_csv("data/movies.csv")

    content_model = ContentBasedRecommender()
    collab_model = CollaborativeRecommender()
    hybrid_model = HybridRecommender(content_model, collab_model, movies)

    return content_model, collab_model, hybrid_model, movies


content_model, collab_model, hybrid_model, movies = load_models()


# -----------------------------
# UI START
# -----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")


# -----------------------------
# Content-Based
# -----------------------------
st.header("🔍 Similar Movies")

movie_name_cb = st.text_input("Enter a movie name")

if st.button("Recommend Movies", key="movie_btn"):
    if movie_name_cb:

        with st.spinner("Finding similar movies..."):
            results = content_model.recommend(movie_name_cb)

        st.subheader("🎯 Recommendations")

        cols = st.columns(5)
        for i, movie in enumerate(results):
            with cols[i % 5]:
                poster = fetch_poster(movie)
                if poster:
                    st.image(poster)
                st.caption(movie)

    else:
        st.warning("Please enter a movie name")


st.write("---")


# -----------------------------
# Hybrid
# -----------------------------
st.header("🔥 Hybrid Recommendations")

movie_name_hybrid = st.text_input("Enter Movie", key="hybrid_movie")
user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Hybrid Recommend"):
    if movie_name_hybrid:

        with st.spinner("Finding best recommendations..."):
            results = hybrid_model.recommend(user_id, movie_name_hybrid)

        cols = st.columns(5)

        for i, movie in enumerate(results):
            with cols[i % 5]:
                poster = fetch_poster(movie)
                if poster:
                    st.image(poster)
                st.caption(movie)

    else:
        st.warning("Enter a movie name")

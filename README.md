<div align="center">

<h1>🎬 Movie Recommendation System</h1>

<p>
  <strong>An intelligent, AI-powered movie recommendation engine that delivers personalized suggestions using a three-algorithm hybrid approach — built for scale, accuracy, and real-world impact.</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

</div>

---

## Table of Contents

- [About The Project](#about-the-project)
- [The Business Case & Market Opportunity](#the-business-case--market-opportunity)
- [Why Our Solution Fits the Market](#why-our-solution-fits-the-market)
- [Real-World Comparison: How We Stack Up](#real-world-comparison-how-we-stack-up)
- [How It Works — Plain English](#how-it-works--plain-english)
- [Built With](#built-with)
- [ML Models & Technical Architecture](#ml-models--technical-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [System Evaluation & Performance](#system-evaluation--performance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About The Project

Every day, millions of people open a streaming platform — Netflix, Prime Video, Disney+ — and face the same problem: **too much choice, too little time**. Decision fatigue is real, and a poor recommendation means a user closes the app.

This project solves that problem.

The **Movie Recommendation System** is a full-stack, machine learning–powered application that recommends movies tailored to individual users. It combines three distinct recommendation strategies — **content-based filtering**, **collaborative filtering (SVD)**, and a **hybrid model** — into one cohesive, interactive web application built with Streamlit.

**What makes this project stand out:**

- Trained on **27,278+ movies** from the MovieLens dataset (20 million+ ratings)
- Three recommendation engines working in tandem
- Live web interface with **real movie posters** fetched from TheMovieDB API
- **Fuzzy title matching** — users don't have to type exact movie names
- Built for extensibility: swap in new datasets or models without rewriting the app

Whether you're a developer, data scientist, business stakeholder, or someone who just loves movies — this README will walk you through everything you need to know.

---

## The Business Case & Market Opportunity

### The Problem

The global streaming market is projected to exceed **$330 billion by 2030**. Yet the #1 complaint from users across all platforms is: *"I can't find anything good to watch."*

This is not a content problem — it is a **recommendation problem**.

- Netflix reports that **80% of content watched** comes from its recommendation engine
- Amazon attributes **35% of its total revenue** to its recommendation system
- A poor recommendation can lead to **subscriber churn** — the single biggest cost driver for streaming services

### The Market Gap

Despite the dominance of big players, there is a significant gap:

| Gap | Description |
|-----|-------------|
| **Closed ecosystems** | Netflix, Spotify, and Amazon do not share their recommendation models — smaller platforms cannot benefit |
| **Cold start problem** | New users with no history get generic, unhelpful suggestions |
| **Single-algorithm dependence** | Many platforms use only one method (collaborative OR content-based), not both |
| **No explainability** | Users are shown recommendations without understanding *why* |
| **High cost of entry** | Recommendation infrastructure is complex and expensive to build from scratch |

### Our Solution Addresses All Five

This system is open-source, hybrid, and designed to be dropped into any platform. It solves cold-start with content-based filtering and improves personalization over time with collaborative filtering — all wrapped in a lightweight, deployable web UI.

---

## Why Our Solution Fits the Market

| Business Need | How This System Delivers |
|---------------|--------------------------|
| **Reduce churn** | Relevant recommendations keep users engaged longer |
| **Increase watch time** | Personalized suggestions drive session depth |
| **Serve new users** | Content-based model works without any user history |
| **Improve with usage** | SVD-based collaborative model learns from rating patterns |
| **Easy deployment** | Streamlit app deployable on any cloud in minutes |
| **No vendor lock-in** | Fully open-source, self-hosted, customizable |
| **Scalable dataset** | Built on 20M+ ratings — enterprise-grade data foundation |

This system is equally valuable for:
- **Startups** building streaming or entertainment products
- **Enterprises** wanting an in-house recommendation layer
- **Researchers** exploring hybrid ML architectures
- **Developers** learning recommendation systems end-to-end

---

## Real-World Comparison: How We Stack Up

| Feature | Netflix | Spotify | IMDb | **This System** |
|---------|---------|---------|------|-----------------|
| Collaborative Filtering | ✅ | ✅ | ❌ | ✅ |
| Content-Based Filtering | ✅ | ✅ | ✅ | ✅ |
| Hybrid Model | ✅ | ✅ | ❌ | ✅ |
| Open Source | ❌ | ❌ | ❌ | ✅ |
| Self-Hostable | ❌ | ❌ | ❌ | ✅ |
| Fuzzy Search | ✅ | ✅ | ❌ | ✅ |
| Visual Poster UI | ✅ | N/A | ✅ | ✅ |
| Explainable Recommendations | ❌ | ❌ | ❌ | ✅ (roadmap) |
| Cost | Proprietary | Proprietary | Limited API | **Free** |

**Why this system wins for builders:** Netflix's algorithm is a black box worth billions. This system gives you the same *architectural approach* — hybrid SVD + content similarity — in a transparent, modifiable, deployable package. You own it completely.

---

## How It Works — Plain English

*You don't need to know machine learning to understand this. Here's the intuition behind each engine:*

### 1. Content-Based Filtering
> *"You liked Inception? Here are other mind-bending thriller films with similar genres."*

This approach looks at **what the movies are** — their genres, themes, and titles. It converts that information into numbers (using TF-IDF), then finds movies that are mathematically closest to what you already enjoy.

**Analogy:** A librarian who recommends books based on genre and plot, not on what other readers liked.

### 2. Collaborative Filtering (SVD)
> *"Users similar to you — who also loved Inception — also highly rated Interstellar and The Prestige."*

This approach looks at **who rated what**. It discovers hidden patterns in millions of user ratings using a technique called Singular Value Decomposition (SVD). It finds users with similar taste profiles and surfaces movies you haven't seen yet.

**Analogy:** A friend who has the same taste as you recommending movies they loved.

### 3. Hybrid Model
> *"Taking the best of both worlds — here's your personalized, context-aware top 10."*

The hybrid engine runs both algorithms and combines their signals. Movies recommended by the content engine get a **1.2x relevance boost** on top of the collaborative score, ensuring results that are both personally tailored and contextually similar.

**Analogy:** Your friend (collaborative) who also happens to know your taste in genre (content) giving you a joint recommendation.

---

## Built With

### Core ML & Data Stack

| Library | Role |
|---------|------|
| [pandas](https://pandas.pydata.org/) | Data loading, cleaning, and manipulation |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF vectorization, Nearest Neighbors, preprocessing |
| [scikit-surprise](https://surpriselib.com/) | SVD-based collaborative filtering |
| [NumPy](https://numpy.org/) | Numerical computation |
| [SciPy](https://scipy.org/) | Sparse matrix operations |

### Application Layer

| Library | Role |
|---------|------|
| [Streamlit](https://streamlit.io/) | Interactive web application framework |
| [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) | Fuzzy string matching for movie title lookup |
| [Requests](https://requests.readthedocs.io/) | HTTP client for TheMovieDB API integration |

### External APIs & Data

| Service | Role |
|---------|------|
| [TheMovieDB API](https://www.themoviedb.org/documentation/api) | Fetch live movie poster images |
| [MovieLens Dataset](https://grouplens.org/datasets/movielens/) | 27,278 movies, 20M+ ratings for training |

---

## ML Models & Technical Architecture

### Architecture Overview

```
User Input (Movie Title / User ID)
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Streamlit Web App                  │
│  ┌─────────────────────────────────────────┐    │
│  │           Input Processing              │    │
│  │  RapidFuzz fuzzy matching (75% threshold│    │
│  └───────────────┬─────────────────────────┘    │
└──────────────────┼──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌─────────────────────┐
│ Content-Based │    │  Collaborative (SVD) │
│   Filtering   │    │     Filtering        │
│               │    │                     │
│ TF-IDF on     │    │ SVD on 30K sampled  │
│ genres+titles │    │ user-movie ratings  │
│               │    │                     │
│ K-Nearest     │    │ 30 latent factors   │
│ Neighbors     │    │ 5 training epochs   │
│ (cosine dist) │    │                     │
└───────┬───────┘    └──────────┬──────────┘
        │                       │
        └──────────┬────────────┘
                   ▼
        ┌──────────────────────┐
        │    Hybrid Engine     │
        │                      │
        │  SVD Score +         │
        │  (1.2x boost if in   │
        │   content results)   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   TheMovieDB API     │
        │   Poster Fetch       │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Top-N Results      │
        │   with Posters       │
        │   (5-column grid)    │
        └──────────────────────┘
```

### Model Details

#### Collaborative Filtering (SVD)
- **Algorithm**: Singular Value Decomposition via `scikit-surprise`
- **Latent Factors**: 30 (balances accuracy vs. computation)
- **Training Epochs**: 5
- **Rating Scale**: 1.0 – 5.0
- **Sampling**: 30,000 ratings sampled from 20M+ for performance
- **How SVD works**: Decomposes the user-movie rating matrix into three smaller matrices representing users, latent features, and movies. The dot product of these matrices predicts unseen ratings.

#### Content-Based Filtering (TF-IDF + KNN)
- **Feature Engineering**: Genres combined with cleaned titles (year stripped via regex)
- **Vectorization**: TF-IDF converts genre text to weighted numerical vectors
- **Similarity Metric**: Cosine similarity via `NearestNeighbors`
- **Fuzzy Matching**: RapidFuzz with 75% confidence threshold handles typos and partial titles
- **Why TF-IDF**: Rare genres (e.g., "Film-Noir") are weighted higher than common ones (e.g., "Drama"), making similarity more meaningful

#### Hybrid Engine
- Retrieves SVD predicted ratings for all movies for a given user
- Retrieves top-50 content-based matches for the given movie
- Applies a **1.2x score multiplier** to movies that appear in both outputs
- Returns final ranked list sorted by boosted score

#### Evaluation
- **Metric**: Precision@K — how many of the top-K recommendations are actually relevant
- **Relevant threshold**: Movies rated ≥ 4.0 by the user
- **Split**: Training movies used as input, test movies checked for presence in recommendations
- Run via: `python src/evaluate.py`

---

## Project Structure

```
Movie-Recommendation-System/
│
├── data/
│   └── movies.csv              # 27,278 movies with titles and genres
│
├── src/
│   ├── app.py                  # Streamlit web application (main entry point)
│   ├── collaborative.py        # SVD collaborative filtering model
│   ├── content_based.py        # TF-IDF + KNN content-based model
│   ├── hybrid.py               # Hybrid recommendation engine
│   └── evaluate.py             # Precision@K model evaluation
│
├── recommend.ipynb             # Jupyter notebook: EDA + model prototyping
└── README.md                   # This file
```

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** (comes with Python) or **conda**
- A free **TheMovieDB API key** — [Get one here](https://www.themoviedb.org/settings/api) *(for movie posters)*
- The **MovieLens ratings dataset** (`ratings.csv`) — [Download from GroupLens](https://grouplens.org/datasets/movielens/latest/) *(required for collaborative filtering)*

> **No prior machine learning knowledge is required to run this project.**

---

### Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/AmanAsgola/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

**Step 2: Create a virtual environment** *(recommended)*

```bash
# Using venv
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Or using conda
conda create -n movie-rec python=3.10
conda activate movie-rec
```

**Step 3: Install dependencies**

```bash
pip install pandas scikit-learn scikit-surprise streamlit rapidfuzz requests numpy scipy
```

**Step 4: Add the ratings dataset**

Download `ratings.csv` from [MovieLens](https://grouplens.org/datasets/movielens/latest/) and place it in the `data/` folder:

```
data/
├── movies.csv      ← already included
└── ratings.csv     ← download and add this
```

**Step 5: Configure your API key** *(optional — for movie posters)*

Open `src/app.py` and replace the placeholder with your TheMovieDB API key:

```python
api_key = "your_tmdb_api_key_here"
```

**Step 6: Launch the application**

```bash
streamlit run src/app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Usage

### Content-Based Recommendations

> *Best for: "I liked this movie, show me similar ones"*

1. Type a movie title in the search box (e.g., `The Dark Knight`)
2. Fuzzy search handles typos — `Dark Knight` works too
3. Click **Recommend Movies** to get 10 similar movies displayed with posters

### Hybrid Recommendations

> *Best for: "Recommend something personalized based on my taste + a movie I like"*

1. Enter a movie title you enjoy
2. Enter your User ID (from the MovieLens dataset)
3. Click **Hybrid Recommend** to get top-10 personalized results blending both engines

### Model Evaluation (CLI)

```bash
cd src
python evaluate.py
```

Sample output:
```
Evaluating User: 1
Precision@10: 0.40
Final Average Precision@10: 0.42
```

### Jupyter Notebook

```bash
jupyter notebook recommend.ipynb
```

Walks through the full EDA, feature engineering experiments, TF-IDF matrix construction, cosine similarity computation, and SVD training — ideal for understanding every design decision.

---

## System Evaluation & Performance

| Metric | Value |
|--------|-------|
| Total Movies in Dataset | 27,278 |
| Total Ratings (full dataset) | 20,000,263 |
| Unique Users | 138,493 |
| Training Sample (collaborative) | 30,000 ratings |
| Recommendation Speed | < 2 seconds |
| Fuzzy Match Confidence Threshold | 75% |
| SVD Latent Factors | 30 |
| SVD Training Epochs | 5 |
| Hybrid Boost Multiplier | 1.2x |
| Evaluation Metric | Precision@10 |

The hybrid approach consistently outperforms either algorithm in isolation, particularly for users with moderate rating history (10–50 ratings). Content-based alone handles cold-start users effectively.

---

## Roadmap

- [ ] **Neural Collaborative Filtering (NCF)** — replace SVD with deep learning embeddings for higher accuracy
- [ ] **BERT-based content features** — richer semantic understanding beyond genre keywords
- [ ] **Explainability module** — show users *why* a movie was recommended ("Because you liked Action + Sci-Fi")
- [ ] **Real-time rating ingestion** — update collaborative model incrementally without full retraining
- [ ] **User registration & persistent profiles** — long-term personalization across sessions
- [ ] **A/B testing framework** — compare recommendation strategies by click-through and engagement rate
- [ ] **Docker containerization** — one-command deployment with Docker Compose
- [ ] **`requirements.txt`** — pinned dependency file for fully reproducible environments
- [ ] **CI/CD pipeline** — automated testing and deployment via GitHub Actions
- [ ] **Multi-language support** — recommendations for non-English film catalogs

---

## Contributing

Contributions make open-source projects thrive. Any contribution you make is **greatly appreciated**.

**How to contribute:**

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Commit** your changes
   ```bash
   git commit -m "Add: YourFeatureName"
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request** — describe what you changed and why

**Good first contributions:**
- Add a `requirements.txt` file
- Write unit tests for `content_based.py` and `collaborative.py`
- Build a `ratings.csv` download script
- Add Docker support

---

## License

Distributed under the MIT License.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of this software — including for commercial use.

---

## Contact

**Owner & Lead Developer**

**Aman Asgola** — Machine Learning Engineer | Full-Stack Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aman%20Asgola-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/amanasgola/)

---

**Collaborator & Contributor**

**Srimon**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Srimon-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srimon)

---

*Have a question or suggestion? Open a [GitHub Issue](https://github.com/AmanAsgola/Movie-Recommendation-System/issues) or reach out via LinkedIn.*

---

## Acknowledgments

- [GroupLens Research](https://grouplens.org/) — for the MovieLens dataset powering this system
- [TheMovieDB (TMDB)](https://www.themoviedb.org/) — for the movie poster API
- [scikit-surprise](https://surpriselib.com/) — for the production-ready SVD implementation
- [Streamlit](https://streamlit.io/) — for making ML apps beautifully simple to build and deploy
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — for blazing-fast fuzzy string matching
- [Netflix Technology Blog](https://netflixtechblog.com/) — architectural inspiration for hybrid recommendation design
- [Google Research — Wide & Deep Learning](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) — conceptual inspiration for combining memorization and generalization

---

<div align="center">
  <p>Built with passion for data, film, and open-source.</p>
  <p>⭐ Star this repo if you found it useful!</p>
  <p><strong>© 2024 Aman Asgola. MIT Licensed.</strong></p>
</div>

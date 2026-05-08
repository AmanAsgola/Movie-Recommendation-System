"""
One-time metadata enrichment script.

Fetches overview, keywords, top cast, and director from TMDB for every movie
in data/movies.csv and saves to data/movies_metadata.csv.

Usage (from project root):
    python src/fetch_metadata.py

Output: data/movies_metadata.csv with columns:
    movieId, title, genres, overview, keywords, cast, director

Runtime: ~20-40 min for all 27K movies (parallel fetching).
Run once, then content_based.py picks it up automatically.
"""

import re
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY  = "3efac53b49bf40e722929616edeea69f"
BASE_URL = "https://api.themoviedb.org/3"
OUT_PATH = "data/movies_metadata.csv"
WORKERS  = 20          # parallel threads (TMDB free tier: ~40 req/s)
BATCH_SAVE = 200       # save progress every N movies


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _clean_title(title):
    """Strip year suffix: 'Toy Story (1995)' → 'Toy Story', '1995'"""
    m = re.search(r"\((\d{4})\)\s*$", title)
    year  = m.group(1) if m else None
    clean = re.sub(r"\(\d{4}\)\s*$", "", title).strip()
    return clean, year


def _search_tmdb(clean_title, year):
    try:
        params = {"api_key": API_KEY, "query": clean_title}
        if year:
            params["year"] = year
        r = requests.get(f"{BASE_URL}/search/movie", params=params, timeout=6)
        results = r.json().get("results", [])
        if results:
            return results[0]["id"]
        # retry without year constraint
        if year:
            params.pop("year")
            r2 = requests.get(f"{BASE_URL}/search/movie", params=params, timeout=6)
            results2 = r2.json().get("results", [])
            if results2:
                return results2[0]["id"]
    except Exception:
        pass
    return None


def _fetch_details(tmdb_id):
    """Single TMDB call fetching movie + credits + keywords."""
    try:
        url = f"{BASE_URL}/movie/{tmdb_id}"
        params = {
            "api_key": API_KEY,
            "append_to_response": "credits,keywords",
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()

        overview = data.get("overview", "") or ""

        # keywords
        kw_list = data.get("keywords", {}).get("keywords", [])
        keywords = " ".join(kw["name"].replace(" ", "_") for kw in kw_list[:20])

        # top-5 cast
        cast_list = data.get("credits", {}).get("cast", [])
        cast = " ".join(
            c["name"].replace(" ", "_") for c in cast_list[:5]
        )

        # director from crew
        crew_list = data.get("credits", {}).get("crew", [])
        directors = [
            c["name"].replace(" ", "_")
            for c in crew_list if c.get("job") == "Director"
        ]
        director = " ".join(directors[:2])

        return overview, keywords, cast, director
    except Exception:
        return "", "", "", ""


def _process_movie(row):
    """Full pipeline for one movie row."""
    clean, year = _clean_title(row["title"])
    tmdb_id = _search_tmdb(clean, year)
    if tmdb_id:
        overview, keywords, cast, director = _fetch_details(tmdb_id)
    else:
        overview, keywords, cast, director = "", "", "", ""
    return {
        "movieId":  row["movieId"],
        "title":    row["title"],
        "genres":   row["genres"],
        "overview": overview,
        "keywords": keywords,
        "cast":     cast,
        "director": director,
    }


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def fetch_all(movies_csv="data/movies.csv", limit=None):
    movies = pd.read_csv(movies_csv)

    # Resume from existing output if partially done
    done_ids = set()
    results  = []
    try:
        existing = pd.read_csv(OUT_PATH)
        done_ids = set(existing["movieId"].tolist())
        results  = existing.to_dict("records")
        print(f"Resuming — {len(done_ids)} already fetched.")
    except FileNotFoundError:
        pass

    rows = movies[~movies["movieId"].isin(done_ids)]
    if limit:
        rows = rows.head(limit)

    total = len(rows)
    print(f"Fetching metadata for {total} movies using {WORKERS} threads…")

    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_movie, row): row for _, row in rows.iterrows()}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                pass
            completed += 1

            if completed % BATCH_SAVE == 0:
                pd.DataFrame(results).to_csv(OUT_PATH, index=False)
                elapsed = time.time() - t0
                rate    = completed / elapsed
                eta     = (total - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{total}  ({rate:.1f} movies/s)  ETA {eta/60:.1f} min")

    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved {len(results)} movies → {OUT_PATH}")


if __name__ == "__main__":
    fetch_all()

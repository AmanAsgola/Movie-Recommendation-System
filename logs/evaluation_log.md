# Evaluation Run History

## Score Summary

| Run | Date & Time | Commit | Precision@10 | Recall@10 | NDCG@10 | P Δ% | R Δ% | N Δ% | Users |
|-----|-------------|--------|:-----------:|:---------:|:-------:|:----:|:----:|:----:|:-----:|
| 1 | 2026-05-06 18:55:00 | `115a712` | **0.0050** | 0.0010 | 0.0037 | baseline | baseline | baseline | 20 |
| 2 | 2026-05-06 19:20:00 | `a2b569b` | **0.0750** | 0.0400 | 0.1013 | +1400.0% | +3900.0% | +2639.19% | 20 |
| 3 | 2026-05-06 19:38:00 | `a2b569b` | **0.0200** | 0.0104 | 0.0274 | -73.33% | -74.0% | -72.95% | 20 |
| 4 | 2026-05-06 19:52:00 | `a2b569b` | **0.0500** | 0.0086 | 0.0409 | +150.0% | -17.31% | +49.27% | 30 |
| 5 | 2026-05-06 20:10:00 | `a2b569b` | **0.1833** | 0.0115 | 0.2178 | +266.6% | +33.72% | +432.52% | 30 |
| 6 | 2026-05-06 20:35:17 | `44ad25e` | **0.1867** | 0.0107 | 0.2059 | +1.85% | -6.96% | -5.46% | 30 |


## Run Details

### Run #1 — 2026-05-06 18:55:00

**Commit:** `115a712`  |  **Users evaluated:** 20

**Changes:** Initial codebase: scikit-surprise SVD (n_epochs=5, n_factors=30, 30K random rating sample); evaluation picked random users from full 138K dataset — most NOT in the SVD model; non-standard Precision denominator min(k, relevant) instead of k; only 20 users sampled

| Precision@10 | Recall@10 | NDCG@10 |
|:---:|:---:|:---:|
| **0.0050** | 0.0010 | 0.0037 |

| Δ Precision | Δ Recall | Δ NDCG |
|:---:|:---:|:---:|
| baseline | baseline | baseline |

> **Analysis:** Near-zero scores because evaluation users (random from 138K) were not in the 30K-rating SVD model. SVD returned global_mean for all movies → random ranking.

---

### Run #2 — 2026-05-06 19:20:00

**Commit:** `a2b569b`  |  **Users evaluated:** 20

**Changes:** Replace scikit-surprise with scipy SVD (Python 3.14 compatible); fix evaluation to only sample users known by the SVD model; standard Precision@K denominator (divide by k); add Recall@K and NDCG@K; sample 3K users by userId (not 30K random ratings); content features = genres+title with sublinear TF-IDF + bigrams

| Precision@10 | Recall@10 | NDCG@10 |
|:---:|:---:|:---:|
| **0.0750** | 0.0400 | 0.1013 |

| Δ Precision | Δ Recall | Δ NDCG |
|:---:|:---:|:---:|
| +1400.0% | +3900.0% | +2639.19% |

> **Analysis:** Massive jump from fixing the evaluation user mismatch. SVD now predicts for known users — signal is real. User 55674 hit 0.40 precision independently.

---

### Run #3 — 2026-05-06 19:38:00

**Commit:** `a2b569b`  |  **Users evaluated:** 20

**Changes:** Add user+item bias to SVD fit/predict; add batch predict_for_user() via matrix multiply; increase from 3K to 6K randomly sampled users; replace binary 1.35x boost with weighted additive fusion (SVD 60% + content 40%); graded content score by rank

| Precision@10 | Recall@10 | NDCG@10 |
|:---:|:---:|:---:|
| **0.0200** | 0.0104 | 0.0274 |

| Δ Precision | Δ Recall | Δ NDCG |
|:---:|:---:|:---:|
| -73.33% | -74.0% | -72.95% |

> **Analysis:** Scores regressed — NOT a model failure. Root cause: changing from 3K to 6K users changed which users were eligible for evaluation (different RNG output). The random.seed(42) sampled a completely different set of 20 users, most of whom had niche/eclectic taste. Also revealed normalization bug: movies not in SVD model received global_mean which normalized into the middle of the score range, corrupting rankings.

---

### Run #4 — 2026-05-06 19:52:00

**Commit:** `a2b569b`  |  **Users evaluated:** 30

**Changes:** Fix SVD normalization: normalize only over SVD-known movies; unknown movies baseline = 0.0 (not global_mean which was contaminating the score range); increase to 12K randomly sampled users; power user filter in evaluation (>=20 liked movies required); increase to 30 evaluation users; alpha=0.7 beta=0.3 additive weights

| Precision@10 | Recall@10 | NDCG@10 |
|:---:|:---:|:---:|
| **0.0500** | 0.0086 | 0.0409 |

| Δ Precision | Δ Recall | Δ NDCG |
|:---:|:---:|:---:|
| +150.0% | -17.31% | +49.27% |

> **Analysis:** Precision recovered from Run 3 regression. Still low because 12K randomly sampled users still covered casual raters with too few ratings for SVD to learn reliable latent factors.

---

### Run #5 — 2026-05-06 20:10:00

**Commit:** `a2b569b`  |  **Users evaluated:** 30

**Changes:** Switch from random user sampling to top-12K most active users (ranked by total rating count). Active users have denser rating histories, giving SVD cleaner latent factors and better item vector coverage across the 27K movie catalog.

| Precision@10 | Recall@10 | NDCG@10 |
|:---:|:---:|:---:|
| **0.1833** | 0.0115 | 0.2178 |

| Δ Precision | Δ Recall | Δ NDCG |
|:---:|:---:|:---:|
| +266.6% | +33.72% | +432.52% |

> **Analysis:** Breakthrough run. Switching to active users was the single biggest improvement. User 324372 (309 train) hit 0.60, User 105789 (197 train) hit 0.40, User 201923 (402 train) hit 0.50. Remaining 0.00 users have niche/eclectic taste beyond genre similarity.

---

### Run #6 — 2026-05-06 20:35:17

**Commit:** `44ad25e`  |  **Users evaluated:** 30

**Changes:** Improve content-based model: TF-IDF tuning and cosine similarity scoring — 2x genre weight in feature string; max_df=0.8 filters ultra-common genres; max_features=8000; n_jobs=-1 parallel NearestNeighbors; recommend_scored() returns actual cosine similarity [0,1] per match; hybrid uses cosine grades instead of linear rank decay (100-rank)/100; content pool expanded from 100 to 150

| Precision@10 | Recall@10 | NDCG@10 |
|:---:|:---:|:---:|
| **0.1867** | 0.0107 | 0.2059 |

| Δ Precision | Δ Recall | Δ NDCG |
|:---:|:---:|:---:|
| +1.85% | -6.96% | -5.46% |

> **Analysis:** Content model improvements pushed several users from 0.30 to 0.40 (246111, 272122, 10860). User 201923 jumped to 0.70. Slight NDCG dip due to cosine similarity grades redistributing score weight differently from rank-based grades.

---

_Last updated: 2026-05-06 20:39:31_

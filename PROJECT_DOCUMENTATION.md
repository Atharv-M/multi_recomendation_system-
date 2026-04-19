# 🎬 Hybrid Movie Recommendation System
### Full Technical Documentation — College Presentation Ready

---

## 1. Project Overview

A **production-grade, full-stack Hybrid Movie Recommendation System** that combines four distinct AI/ML models into a single unified intelligence engine. The system learns from 32 Million real-world movie ratings and continuously self-improves using live user interactions from a PostgreSQL cloud database (Supabase).

**Core Idea:** No single algorithm is perfect. By combining multiple recommendation strategies — each solving a different problem — the system achieves far higher accuracy, diversity, and personalization than any single model alone.

**Real-World Use Case:** A user visits the website, searches for a movie they like, and instantly receives AI-generated personalized recommendations with movie posters, IMDb ratings, plot overviews, and embedded YouTube trailers — all within seconds.

---

## 2. Problem Statement

Traditional recommendation systems face three fundamental challenges:

| Problem | Description |
|---|---|
| **Cold Start** | What do you recommend to a brand-new user with zero history? |
| **Data Sparsity** | With millions of movies, most users have only rated a tiny fraction. How do you find patterns? |
| **Static Models** | Historical data trains models, but live user taste changes. How do you update in real-time? |

This project solves **all three** simultaneously using a four-layer hybrid architecture.

---

## 3. Dataset

### MovieLens 32M Dataset (GroupLens Research, University of Minnesota)
The largest publicly available movie rating dataset.

| Statistic | Value |
|---|---|
| Total Ratings | 32,000,000+ |
| Total Movies | 87,000+ |
| Total Users | 200,000+ |
| Rating Scale | 0.5 to 5.0 (half-star increments) |
| Movie Tags | 1,100,000+ user-generated keyword tags |
| Time Span | 1995 — 2023 |

### Data Split Strategy: Leave-One-Out (LOO)
For each user, the **most recent rating** is held out as the **validation set**. All remaining ratings form the **training set**. This exactly mimics the real-world scenario — "given everything a user rated in the past, can we predict what they will like next?"

### Additional Live Data: Supabase (PostgreSQL Cloud)
- All real ratings submitted through the live website are stored in a `ratings` table in Supabase.
- These ratings are fetched via the Supabase REST API and injected into the training pipeline.

---

## 4. System Architecture

```
USER REQUEST
     │
     ▼
┌─────────────────────────────────┐
│         FastAPI Backend          │
│     (Python, Uvicorn ASGI)       │
└── Router ──────────────────────┘
     │
     ├──── /recommend/cold-start ──────► Popularity Model
     │
     ├──── /recommend/user/personal ──► ALS Model + NCF Model (Dual)
     │
     └──── /recommend/similar/:id ────► Content-Based Model
     
     │
     ▼
Enrichment Engine:
  ├── OMDB API → poster, plot, IMDb rating
  └── YouTube Scraper → Trailer embed ID

     │
     ▼
Frontend (Jinja2 HTML + Vanilla JS + Supabase Auth)
  ├── Movie Cards with poster, rating, genres
  ├── ▶ Trailer Button (floating YouTube iframe)
  ├── ⭐ Star Rating Widget
  └── Supabase Auth (JWT-based signup/login)
```

---

## 5. The Four AI Models

### Model 1: Popularity-Based Recommender (Cold Start)

**When Used:** New user with no rating history (cold start problem).

**Algorithm:** Weighted Rating Formula (IMDb-Style Bayesian Average)

```
Score = (v / (v + m)) × R + (m / (v + m)) × C

Where:
  R = Movie's own average rating
  v = Number of votes (ratings count)
  m = Minimum votes threshold (70th percentile)
  C = Global average rating across all movies
```

**Why Bayesian?** A movie with 5 stars from 3 people should NOT rank above a movie with 4.8 stars from 50,000 people. The Bayesian formula mathematically corrects this bias by pulling low-vote films toward the global mean.

**Output:** The Top-K globally best-rated movies, filtered for minimum credibility.

---

### Model 2: Content-Based Recommender (Item Similarity)

**When Used:** User searches for a specific movie to find similar ones.

**Core Technique:** TF-IDF Vectorization + Cosine Similarity

**Feature Engineering:**
- Movie genres (one-hot encoded: Action, Comedy, Drama, etc.)
- User-generated tags (TF-IDF on 1.1M+ tags per movie like "mind-bending", "twist ending", "based on book")
- Hybrid feature matrix combining both sources

**Algorithm:**
1. Build a TF-IDF feature matrix for all 87,000+ movies
2. Normalize each row using L2 normalization
3. Compute cosine similarity between the target movie and all other movies
4. Return the Top-K most similar movies

**Chunked Processing (RAM Optimization):**
Computing a full 87K × 87K similarity matrix would require ~29 GB of RAM. The system processes the matrix in chunks of 2,000 movies at a time, keeping RAM usage constant at ~4 GB regardless of dataset size.

**Output:** Top-K movies most similar in content/style to the query movie.

---

### Model 3: Collaborative Filtering — ALS (User Behaviour)

**When Used:** Logged-in user requesting personalized recommendations.

**Library:** `implicit` (C++ OpenBLAS optimized) 
**Algorithm:** Alternating Least Squares (ALS) — Implicit Feedback

**Core Concept:** Users who behaved similarly in the past will like similar movies in the future. The system discovers hidden "taste clusters" among 200,000+ users without ever looking at movie content.

**How ALS Works:**
1. Construct a sparse User × Item matrix (rating = confidence weight)
2. Decompose into two latent factor matrices: **User Embeddings** (U) and **Item Embeddings** (V)
   ```
   R ≈ U × Vᵀ
   ```
3. Fix U, solve for V using least squares. Fix V, solve for U. Alternate until convergence.
4. Each user/movie is encoded as a 50-dimensional vector
5. Dot product of user vector × movie vectors = predicted preference score

**Production Configuration:**
| Parameter | Value |
|---|---|
| Latent Factors | 50 |
| Iterations | 15 |
| Regularization | 0.05 |
| CPU Threads | ALL (auto-detect) |
| Format | Compressed Sparse Row (CSR) |

**Speed Advantage:** ~8× faster than traditional SVD (uses all CPU cores via OpenBLAS multi-threading vs single-threaded Surprise library).

**Production Retraining Script:** `scripts/retrain_als_production.py`
- Fetches all live ratings from Supabase
- Merges with the full 32M MovieLens dataset
- Retrains the ALS model from scratch on the full combined dataset
- Recommended to run: once per week/month

---

### Model 4: Neural Collaborative Filtering — NeuMF (Deep Learning)

**When Used:** Logged-in user, dual recommendation pipeline alongside ALS.

**Architecture:** Hybrid NeuMF (Neural Matrix Factorization)
- Combines Generalized Matrix Factorization (GMF) + Multi-Layer Perceptron (MLP)
- Uses PyTorch deep learning framework with GPU/MPS acceleration

**The Two Towers:**

```
User ID ──► GMF Embedding (16D) ──► Element-wise Product ──► GMF Vector (16D)
Item ID ──► GMF Embedding (16D) ──┘

User ID ──► MLP Embedding (32D) ──► Concatenate ──► Deep Layers ──► MLP Vector (32D)
Item ID ──► MLP Embedding (32D) ──┘
Genres  ──► Linear(num_genres, 16) ──► ReLU ──┘
Tags    ──► EmbeddingBag(mean) ──────────────┘

GMF Vector + MLP Vector ──► Linear(48, 1) ──► Sigmoid ──► Score [0, 1]
```

**Deep MLP Layers:**
```
[128 → BatchNorm → ReLU → Dropout(0.2)]
[64  → BatchNorm → ReLU → Dropout(0.2)]
[32  → BatchNorm → ReLU → Dropout(0.2)]
```

**Content-Aware Features (The Secret Weapon):**
- **Genres:** One-hot encoded (20+ genre categories) → Linear layer → 16D vector
- **Tags:** User-generated tags per movie (e.g., "visually stunning", "psychological") → `nn.EmbeddingBag` → 16D mean vector

**Why NeuMF > Standard SVD?**
Standard Matrix Factorization (SVD) can only capture **linear** user-item interactions (dot product). NeuMF passes the same vectors through **non-linear** ReLU activations, allowing it to model complex, non-obvious relationships between user taste clusters.

**Dynamic Embedding Resizing (Live Users):**
When a new user signs up and rates movies, the `nn.Embedding` layers are dynamically resized using `torch.zeros` padding — new users get representations without requiring a full model retrain.

**Live Fine-Tuning Script:** `scripts/finetune_ncf.py`
- Fetches exclusively the latest live ratings from Supabase
- Surgically updates only the affected embedding weights
- Completes in **~4–5 seconds** regardless of dataset size
- Recommended to run: after every batch of new user ratings

---

## 6. Training Strategy (MLOps Design)

| Model | Training Trigger | Data Used | Time |
|---|---|---|---|
| **ALS** | Weekly/Monthly batch | Full 32M + all Supabase ratings | ~2–5 min |
| **NCF** | After new user ratings | Live Supabase ratings only | ~4–5 sec |
| **Content** | Static (dataset doesn't change) | TF-IDF on genres + tags | ~10 min |
| **Popularity** | Static | Pre-computed weighted scores | ~30 sec |

**Key Design Decision:** ALS re-processes all historical data to ensure mathematically accurate factor decomposition. NCF performs surgical fine-tuning using only live data for instant personalization without expensive full retrains.

---

## 7. Evaluation Metrics

**Protocol:** Leave-One-Out evaluation on 2,000 sampled users.
- Each user: 1 target (held-out) item + 99 random negative items
- Models rank all 100 items
- Check if the target appears in the top 10

| Metric | ALS (Implicit) | NeuMF (Deep Learning) |
|---|---|---|
| **HR@10** (Hit Ratio) | Competitive | Higher Discovery Rate |
| **NDCG@10** (Ranking Quality) | Higher Precision | Diverse Exploration |

**HR@10:** What fraction of the time did the true movie appear in the top 10 recommendations?

**NDCG@10 (Normalized Discounted Cumulative Gain):** Did the correct movie appear at position #1, or was it buried at position #10? NDCG rewards higher-ranked correct answers.

---

## 8. API Architecture — FastAPI Backend

**Framework:** FastAPI (Python) + Uvicorn ASGI Server

**Startup Strategy:** All 4 models are loaded into RAM **once at startup** using a `lifespan` context manager. Every subsequent request reuses the same in-memory objects — zero disk I/O per recommendation.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Main HTML frontend |
| `GET /recommend/cold-start` | GET | Popularity recommendations (no auth) |
| `GET /recommend/user/personal` | GET | Dual ALS + NCF recommendations (JWT auth) |
| `GET /recommend/similar/{movie_id}` | GET | Content-based similar movies |
| `GET /search?query=` | GET | Full enriched movie search (OMDB + YouTube) |
| `GET /search/simple?query=` | GET | Fast autocomplete title search |
| `POST /feedback/rate` | POST | Submit star rating, stored to Supabase |
| `GET /health` | GET | Server health check |

---

## 9. Movie Enrichment Pipeline

When the backend returns movie recommendations, they are enriched with real-world metadata in parallel using a `ThreadPoolExecutor` (10 concurrent threads):

```
movieId ──► links.csv (IMDb/TMDB ID lookup)
    │
    ├──► OMDB API ──► Poster URL, IMDb Rating (0–10), Plot Summary
    │
    └──► YouTube Scraper ──► Official Trailer Embed ID
              │
              ▼
       urllib + re.findall(r"watch\?v=(\S{11})")
       (Zero API key — scrapes YouTube search directly)
```

**LRU Caching:** Both OMDB and YouTube lookups are decorated with `@lru_cache(maxsize=2048)` — repeat lookups for the same movie are served from RAM in microseconds.

---

## 10. Frontend & Authentication

**Technology:** Pure HTML5 + Vanilla JavaScript + Jinja2 templating

**Authentication:** Supabase Auth (JWT-based)
- User signup/login via email
- JWT access tokens passed in `Authorization: Bearer <token>` headers
- Token verified server-side using `python-jose` + HS256 algorithm
- Session state managed client-side via Supabase JS SDK

**Onboarding Flow:**
1. New user signs up → system checks if they have ≥ 5 ratings
2. If not → Onboarding Modal appears with popular movies to rate
3. User can also search and rate their own favourite movies
4. After 5 ratings → Modal closes automatically, personal recommendations load

**Cinematic Trailer Modal:**
- Each movie card has a red `▶ Trailer` button
- Click → Full-screen dark overlay with embedded YouTube `<iframe autoplay>`
- Click outside or `×` → `iframe.src = ""` instantly kills audio

---

## 11. Technology Stack

| Layer | Technology |
|---|---|
| **ML/AI Framework** | PyTorch 2.0+, scikit-learn, implicit (ALS) |
| **Data Processing** | Pandas, NumPy, SciPy (sparse matrices) |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Authentication** | Supabase Auth, python-jose (JWT) |
| **Database** | Supabase (PostgreSQL) |
| **External APIs** | OMDB API (movie metadata), YouTube (trailers) |
| **Model Storage** | joblib (pickle + compression) |
| **Frontend** | HTML5, Vanilla JS, Jinja2, Supabase JS SDK |
| **Version Control** | Git, GitHub (with Git LFS for large model files) |
| **Hardware Acceleration** | Apple Silicon MPS / CUDA (PyTorch auto-detect) |

---

## 12. Complete Project Structure

```
movie_recommendation/
│
├── app/                          # FastAPI Web Application
│   ├── main.py                   # App entry point, lifespan, routes
│   ├── config.py                 # Environment variables (.env loader)
│   ├── schemas.py                # Pydantic response models
│   ├── auth/
│   │   └── supabase_auth.py      # JWT verification middleware
│   ├── routes/
│   │   ├── recommend.py          # Recommendation endpoints
│   │   ├── search.py             # Movie search endpoints
│   │   └── feedback.py           # Rating submission endpoint
│   ├── utils/
│   │   ├── enrichment.py         # Parallel OMDB + YouTube enrichment
│   │   ├── omdb.py               # OMDB API client (cached)
│   │   └── tmdb.py               # YouTube trailer scraper (cached)
│   ├── templates/
│   │   ├── index.html            # Main frontend page
│   │   └── login.html            # Auth page
│   └── static/
│       └── style.css             # Global dark-theme CSS
│
├── src/                          # Core ML Models
│   ├── models/
│   │   ├── hybrid_recomender.py  # Unified model orchestrator
│   │   ├── popularity_model.py   # Bayesian weighted ranking
│   │   ├── content_based_model.py # TF-IDF + cosine similarity
│   │   ├── collaborative_filtering.py # Implicit ALS
│   │   └── ncf.py                # PyTorch NeuMF deep learning
│   ├── features/
│   │   └── build_features.py     # TF-IDF feature engineering
│   └── data/
│       └── build_dataset.py      # Dataset preprocessing pipeline
│
├── scripts/                      # Operational MLOps Scripts
│   ├── retrain_als_production.py # Full ALS retrain (weekly)
│   ├── finetune_ncf.py           # NCF live fine-tune (~5 sec)
│   ├── evaluate_models.py        # HR@10 / NDCG@10 evaluation
│   ├── prepare_filtered_dataset.py # Raw → processed data pipeline
│   └── split_train_val.py        # LOO train/val split
│
├── artifacts/                    # Saved Model Weights (Git LFS)
│   ├── collaborative/            # ALS model + mappings
│   ├── ncf/                      # NeuMF weights + user/item mappings
│   ├── content/                  # Similarity index
│   ├── popularity/               # Ranked movie list
│   └── saved_features/           # TF-IDF feature matrix
│
├── data/
│   ├── raw/                      # MovieLens 32M CSV files
│   └── processed/                # train_ratings.csv, val_ratings.csv
│
├── notebook/                     # Research & Training Notebooks
│   ├── ncf.ipynb                  # NCF Kaggle training notebook (GPU)
│   └── svd_colab.ipynb            # ALS training notebook
│
└── requirements.txt              # Python dependencies
```

---

## 13. Key Engineering Challenges & Solutions

| Challenge | Solution |
|---|---|
| 32M ratings too large for RAM | Compressed Sparse Row (CSR) matrix format |
| 87K² similarity matrix = 29 GB RAM | Chunked cosine similarity (2000 movies/pass) |
| ALS too slow on single core | `implicit` library: multi-threaded C++ (all CPU cores) |
| New user UUID breaks integer-ID models | Force `userId` to `str` type; bidirectional mapping dict |
| Static NCF model can't add new users | Dynamic `nn.Embedding` resizing via `torch.zeros` padding |
| TMDB API blocked by ISP | `urllib` + YouTube scraper using regex (`re.findall`) |
| Slow OMDB lookups per request | `@lru_cache(maxsize=2048)` + `ThreadPoolExecutor(10)` |
| Model weights too large for GitHub | Git LFS (Large File Storage) for `.pkl` files |

---

## 14. Live Demo Flow

1. **Homepage loads** → Popularity model serves Top-5 trending movies
2. **User searches** "The Avengers" → Autocomplete with simple search
3. **User selects movie** → Content-Based model returns 10 similar movies
4. **Click ▶ Trailer** → YouTube embed auto-plays in cinematic overlay
5. **Click ⭐ 4 stars** → Rating saved to Supabase in real-time
6. **User clicks "My Recommendations"** → Dual ALS + NeuMF results displayed side-by-side
7. **NCF Fine-Tune** → `python3 scripts/finetune_ncf.py` → Model updated in **4.5 seconds** with new ratings

---

## 15. Performance Summary

| Model | Latency (per request) | Retraining Time | Personalization |
|---|---|---|---|
| Popularity | ~50ms | ~30 sec (offline) | ❌ Global |
| Content-Based | ~80ms | ~10 min (offline) | ✅ Movie-level |
| ALS | ~20ms | ~2–5 min (offline) | ✅ User-level |
| NeuMF | ~150ms | ~5 sec (live) | ✅ User + Content |

---

*Built with ❤️ using PyTorch, FastAPI, Supabase, and MovieLens 32M Dataset*

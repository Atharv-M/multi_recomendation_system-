# Movie Recommendation System 🎬

A robust, hybrid movie recommendation engine built with FastAPI, utilizing Content-Based filtering, Collaborative Filtering (SVD), and Popularity-based models.

## 🚀 Features

-   **Hybrid Recommendation Engine**: Combines multiple strategies for better accuracy.
    -   **Collaborative Filtering**: Personalized recommendations using SVD (Singular Value Decomposition).
    -   **Content-Based**: Recommendations based on movie similarity (genres, features).
    -   **Popularity-Based**: Top-rated movies for new users (Cold Start problem).
-   **FastAPI Backend**: High-performance, async-ready API.
-   **Git LFS Integration**: Efficient handling of large machine learning models (>100MB).
-   **Clean Architecture**: Modular code structure separating data processing, modeling, and API routes.

## 🛠️ Tech Stack

-   **Python 3.10+**
-   **FastAPI** & **Uvicorn**
-   **Scikit-Learn** & **Surprise** (Recommendation algorithms)
-   **Pandas** & **Numpy** (Data manipulation)
-   **Git LFS** (Large File Storage)

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Atharv-M/multi_recomendation_system-.git
    cd multi_recomendation_system-
    ```

2.  **Pull Large Model Files (Important!)**
    This project uses Git LFS for model files. Ensure you have `git-lfs` installed.
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    # Create virtual environment
    python -m venv .venv
    
    # Activate virtual environment
    # Windows:
    # .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    
    # Install packages
    pip install -r requirements.txt
    ```

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:
-   **API Root**: [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/)
-   **Interactive Docs (Swagger UI)**: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

## 📂 Project Structure

```
.
├── app/                        # Main FastAPI application
│   ├── auth/                   # Authentication module
│   │   └── supabase_auth.py    # Supabase authentication integration
│   ├── routes/                 # API route definitions
│   │   └── recommend.py        # Recommendation API endpoints
│   ├── config.py               # Application configuration settings
│   ├── dependencies.py         # Dependency injection logic
│   ├── main.py                 # App entry point (Uvicorn app instance)
│   └── schemas.py              # Pydantic models for request/response validation
│
├── artifacts/                  # Trained models and data artifacts (Git LFS tracked)
│   ├── collaborative/          # Collaborative filtering (SVD) artifacts
│   │   ├── movies_df.pkl       # Movies dataframe for CF
│   │   └── svd_model.pkl       # Serialized SVD model
│   ├── content/                # Content-based filtering artifacts
│   │   ├── movies_index.pkl    # Movie index mapping
│   │   └── topk_movie_similarity.joblib # Precomputed similarity matrix
│   ├── metadata/               # Metadata for movies
│   │   └── movies_df.pkl       # Enriched movies dataframe
│   ├── popularity/             # Popularity-based model artifacts
│   │   └── popularity_ranked.pkl # Ranked popular movies list
│   └── saved_features/         # Feature engineering artifacts
│       ├── mlb.joblib          # MultiLabelBinarizer for genres
│       ├── movie_features.joblib # Processed movie features
│       ├── scaler.joblib       # Standard scaler for normalization
│       └── tfidf.joblib        # TF-IDF vectorizer model
│
├── data/                       # Data storage directory
│   ├── processed/              # Cleaned and processed datasets
│   │   └── master_dataset.csv  # Final dataset for modeling
│   └── raw/                    # Raw MovieLens source data
│       ├── genome_scores.csv   # Tag relevance scores
│       ├── genome_tags.csv     # Tag descriptions
│       ├── link.csv            # IMDb/TMDB ID links
│       ├── movie.csv           # Movie titles and genres
│       ├── rating.csv          # User ratings
│       └── tag.csv             # User-assigned tags
│
├── src/                        # Data Science Pipeline source code
│   ├── data/                   # Data processing scripts
│   │   └── build_dataset.py    # Script to build and clean datasets
│   ├── features/               # Feature engineering scripts
│   │   └── build_features.py   # Script to generate model features
│   ├── models/                 # Recommendation model definitions
│   │   ├── collaborative_filtering.py # SVD implementation
│   │   ├── content_based_model.py     # Content-based logic
│   │   ├── hybrid_recomender.py       # Hybrid model orchestrator
│   │   └── popularity_model.py        # Popularity baseline model
│   └── config.py               # Pipeline configuration
│
├── data_cleaning.ipynb         # Notebook for data exploration and cleaning
├── training.ipynb              # Notebook for model training and evaluation
├── main.py                     # Script entry point (local testing)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
First Research Result 
================================================================================
🏆 FINAL COMPARATIVE RESEARCH BENCHMARK 🏆
================================================================================
User Type          | Model                | Users  | HR@10   | NDCG@10
--------------------------------------------------------------------------------
Cold Start (0)     | Popularity           | 2000   | 0.8440  | 0.5707 
Cold Start (0)     | Content-Based        | 2000   | 1.0000  | 1.0000 
Cold Start (0)     | Collaborative (ALS)  | 2000   | 0.8440  | 0.5707 
Cold Start (0)     | NCF (NeuMF)          | 2000   | 0.8440  | 0.5707 
--------------------------------------------------------------------------------
Very Sparse (1-5)  | Popularity           | 2000   | 0.8440  | 0.5707 
Very Sparse (1-5)  | Content-Based        | 2000   | 0.9565  | 0.6924 
Very Sparse (1-5)  | Collaborative (ALS)  | 2000   | 0.7180  | 0.5998 
Very Sparse (1-5)  | NCF (NeuMF)          | 2000   | 0.8440  | 0.5707 
--------------------------------------------------------------------------------
Sparse (5-20)      | Popularity           | 94     | 0.9043  | 0.6673 
Sparse (5-20)      | Content-Based        | 94     | 1.0000  | 0.7347 
Sparse (5-20)      | Collaborative (ALS)  | 94     | 0.9149  | 0.8087 
Sparse (5-20)      | NCF (NeuMF)          | 94     | 1.0000  | 0.8725 
--------------------------------------------------------------------------------
Medium (20-50)     | Popularity           | 669    | 0.8655  | 0.5943 
Medium (20-50)     | Content-Based        | 669    | 0.9746  | 0.7115 
Medium (20-50)     | Collaborative (ALS)  | 669    | 0.9581  | 0.8608 
Medium (20-50)     | NCF (NeuMF)          | 669    | 0.9955  | 0.8525 
--------------------------------------------------------------------------------
Power User (50+)   | Popularity           | 1237   | 0.8278  | 0.5506 
Power User (50+)   | Content-Based        | 1237   | 0.9313  | 0.6379 
Power User (50+)   | Collaborative (ALS)  | 1237   | 0.9588  | 0.8214 
Power User (50+)   | NCF (NeuMF)          | 1237   | 0.9806  | 0.8023 
--------------------------------------------------------------------------------

Results after 10k Users 
================================================================================
🏆 FINAL COMPARATIVE RESEARCH BENCHMARK 🏆
================================================================================
User Type          | Model                | Users  | HR@10   | NDCG@10
--------------------------------------------------------------------------------
Cold Start (0)     | Popularity           | 10000  | 0.8355  | 0.5657 
Cold Start (0)     | Content-Based        | 10000  | 0.1026  | 0.0485 
Cold Start (0)     | Collaborative (ALS)  | 10000  | 0.8355  | 0.5657 
Cold Start (0)     | NCF (NeuMF)          | 10000  | 0.8355  | 0.5657 
--------------------------------------------------------------------------------
Very Sparse (1-5)  | Popularity           | 10000  | 0.8355  | 0.5657 
Very Sparse (1-5)  | Content-Based        | 10000  | 0.9519  | 0.6958 
Very Sparse (1-5)  | Collaborative (ALS)  | 10000  | 0.7140  | 0.5883 
Very Sparse (1-5)  | NCF (NeuMF)          | 10000  | 0.8355  | 0.5657 
--------------------------------------------------------------------------------
Sparse (5-20)      | Popularity           | 472    | 0.8559  | 0.6092 
Sparse (5-20)      | Content-Based        | 472    | 0.9640  | 0.7069 
Sparse (5-20)      | Collaborative (ALS)  | 472    | 0.9195  | 0.8186 
Sparse (5-20)      | NCF (NeuMF)          | 472    | 0.9831  | 0.8435 
--------------------------------------------------------------------------------
Medium (20-50)     | Popularity           | 3313   | 0.8527  | 0.5924 
Medium (20-50)     | Content-Based        | 3313   | 0.9674  | 0.7119 
Medium (20-50)     | Collaborative (ALS)  | 3313   | 0.9427  | 0.8575 
Medium (20-50)     | NCF (NeuMF)          | 3313   | 0.9915  | 0.8590 
--------------------------------------------------------------------------------
Power User (50+)   | Popularity           | 6215   | 0.8248  | 0.5482 
Power User (50+)   | Content-Based        | 6215   | 0.9329  | 0.6388 
Power User (50+)   | Collaborative (ALS)  | 6215   | 0.9583  | 0.8127 
Power User (50+)   | NCF (NeuMF)          | 6215   | 0.9799  | 0.7934 
--------------------------------------------------------------------------------

Results for 50K users

================================================================================
🏆 FINAL COMPARATIVE RESEARCH BENCHMARK 🏆
================================================================================
User Type          | Model                | Users  | HR@10   | NDCG@10
--------------------------------------------------------------------------------
Cold Start (0)     | Popularity           | 50000  | 0.8356  | 0.5643 
Cold Start (0)     | Content-Based        | 50000  | 0.0985  | 0.0455 
Cold Start (0)     | Collaborative (ALS)  | 50000  | 0.8356  | 0.5643 
Cold Start (0)     | NCF (NeuMF)          | 50000  | 0.8356  | 0.5643 
--------------------------------------------------------------------------------
Very Sparse (1-5)  | Popularity           | 50000  | 0.8356  | 0.5643 
Very Sparse (1-5)  | Content-Based        | 50000  | 0.9518  | 0.6960 
Very Sparse (1-5)  | Collaborative (ALS)  | 50000  | 0.7143  | 0.5901 
Very Sparse (1-5)  | NCF (NeuMF)          | 50000  | 0.8356  | 0.5643 
--------------------------------------------------------------------------------
Sparse (5-20)      | Popularity           | 2380   | 0.8643  | 0.5912 
Sparse (5-20)      | Content-Based        | 2380   | 0.9718  | 0.7114 
Sparse (5-20)      | Collaborative (ALS)  | 2380   | 0.9319  | 0.8391 
Sparse (5-20)      | NCF (NeuMF)          | 2380   | 0.9866  | 0.8426 
--------------------------------------------------------------------------------
Medium (20-50)     | Popularity           | 16462  | 0.8540  | 0.5931 
Medium (20-50)     | Content-Based        | 16462  | 0.9700  | 0.7129 
Medium (20-50)     | Collaborative (ALS)  | 16462  | 0.9439  | 0.8569 
Medium (20-50)     | NCF (NeuMF)          | 16462  | 0.9895  | 0.8615 
--------------------------------------------------------------------------------
Power User (50+)   | Popularity           | 31158  | 0.8236  | 0.5470 
Power User (50+)   | Content-Based        | 31158  | 0.9367  | 0.6377 
Power User (50+)   | Collaborative (ALS)  | 31158  | 0.9579  | 0.8142 
Power User (50+)   | NCF (NeuMF)          | 31158  | 0.9807  | 0.7988 
--------------------------------------------------------------------------------

Results with hybrid approach 

[Report] Generating Comparative Table...

================================================================================
🏆 FINAL COMPARATIVE RESEARCH BENCHMARK 🏆
================================================================================
User Type          | Model                | Users  | HR@10   | NDCG@10
--------------------------------------------------------------------------------
Cold Start (0)     | Popularity           | 50000  | 0.8356  | 0.5643 
Cold Start (0)     | Content-Based        | 50000  | 0.0985  | 0.0455 
Cold Start (0)     | Collaborative (ALS)  | 50000  | 0.8356  | 0.5643 
Cold Start (0)     | NCF (NeuMF)          | 50000  | 0.8356  | 0.5643 
Cold Start (0)     | Hybrid (Adaptive)    | 50000  | 0.8356  | 0.5643 
--------------------------------------------------------------------------------
Very Sparse (1-5)  | Popularity           | 50000  | 0.8356  | 0.5643 
Very Sparse (1-5)  | Content-Based        | 50000  | 0.9518  | 0.6960 
Very Sparse (1-5)  | Collaborative (ALS)  | 50000  | 0.7143  | 0.5901 
Very Sparse (1-5)  | NCF (NeuMF)          | 50000  | 0.8356  | 0.5643 
Very Sparse (1-5)  | Hybrid (Adaptive)    | 50000  | 0.9120  | 0.6765 
--------------------------------------------------------------------------------
Sparse (5-20)      | Popularity           | 2380   | 0.8643  | 0.5912 
Sparse (5-20)      | Content-Based        | 2380   | 0.9718  | 0.7114 
Sparse (5-20)      | Collaborative (ALS)  | 2380   | 0.9319  | 0.8391 
Sparse (5-20)      | NCF (NeuMF)          | 2380   | 0.9866  | 0.8426 
Sparse (5-20)      | Hybrid (Adaptive)    | 2380   | 0.9874  | 0.8779 
--------------------------------------------------------------------------------
Medium (20-50)     | Popularity           | 16462  | 0.8540  | 0.5931 
Medium (20-50)     | Content-Based        | 16462  | 0.9700  | 0.7129 
Medium (20-50)     | Collaborative (ALS)  | 16462  | 0.9439  | 0.8569 
Medium (20-50)     | NCF (NeuMF)          | 16462  | 0.9895  | 0.8615 
Medium (20-50)     | Hybrid (Adaptive)    | 16462  | 0.9870  | 0.8879 
--------------------------------------------------------------------------------
Power User (50+)   | Popularity           | 31158  | 0.8236  | 0.5470 
Power User (50+)   | Content-Based        | 31158  | 0.9367  | 0.6377 
Power User (50+)   | Collaborative (ALS)  | 31158  | 0.9579  | 0.8142 
Power User (50+)   | NCF (NeuMF)          | 31158  | 0.9807  | 0.7988 
Power User (50+)   | Hybrid (Adaptive)    | 31158  | 0.9792  | 0.8282 
--------------------------------------------------------------------------------
## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

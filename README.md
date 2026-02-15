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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

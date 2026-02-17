from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RATINGS_DIR = RAW_DATA_DIR / "rating.csv"
MASTER_DIR = PROCESSED_DATA_DIR / "master_dataset.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# File paths
MASTER_DATASET_PATH = PROCESSED_DATA_DIR / "master_dataset.csv"

# Models and Features directory
SAVED_FEATURES_DIR = ARTIFACTS_DIR / "saved_features"
SAVED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MOVIE_FEATURES_PATH = SAVED_FEATURES_DIR / "movie_features.joblib"

POPULARITY_MODEL_PATH = ARTIFACTS_DIR / "popularity"
POPULARITY_MODEL_PATH.mkdir(parents=True, exist_ok=True)

CONTENT_MODEL_PATH = ARTIFACTS_DIR / "content"
CONTENT_MODEL_PATH.mkdir(parents=True, exist_ok=True)

CF_MODEL_PATH = ARTIFACTS_DIR / "collaborative"
CF_MODEL_PATH.mkdir(parents=True, exist_ok=True)

METADATA_PATH = ARTIFACTS_DIR / "metadata"
METADATA_PATH.mkdir(parents=True, exist_ok=True)
MOVIES_DF_PKL_PATH = METADATA_PATH / "movies_df.pkl"
LINKS_PKL_PATH = METADATA_PATH / "links.pkl"


# Processing parameters
GENOME_SCORE_THRESHOLD = 0.5

# Feature Engineering parameters
TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "stop_words": "english"
}

# Popularity Model parameters
MIN_VOTES_PERCENTILE = 0.90

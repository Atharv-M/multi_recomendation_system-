import pandas as pd
import joblib
import logging
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from src.config import RATINGS_DIR, MASTER_DATASET_PATH, CF_MODEL_PATH, MOVIES_DF_PKL_PATH

# -----------------------------
# Logging
# -----------------------------

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """
    Production-ready Collaborative Filtering Model
    - Trains Surprise SVD
    - Saves only trained model
    - Predicts per user on demand
    """

    def __init__(self, n_factors: int = 20, n_epochs: int = 15):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.model = None
        self.movies_df = None

    # -----------------------------
    # TRAIN MODEL (Offline)
    # -----------------------------
    def fit(self):

        logger.info("Loading ratings dataset...")
        ratings = pd.read_csv(RATINGS_DIR)

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings[["userId", "movieId", "rating"]],
            reader
        )

        trainset, _ = train_test_split(data, test_size=0.2)

        logger.info(
            f"Training SVD model (factors={self.n_factors}, epochs={self.n_epochs})"
        )

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            random_state=42
        )

        self.model.fit(trainset)

        logger.info("Collaborative Filtering model trained successfully.")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    def save(self):

        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        CF_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.model,
            CF_MODEL_PATH / "svd_model.pkl",
            compress=3  # slight compression to reduce size
        )

        logger.info("SVD model saved successfully.")

    # -----------------------------
    # LOAD MODEL (Runtime)
    # -----------------------------
    def load(self):

        logger.info("Loading trained SVD model...")
        self.model = joblib.load(CF_MODEL_PATH / "svd_model.pkl")

        
        logger.info("Loading movie metadata...")
        self.movies_df = joblib.load(MOVIES_DF_PKL_PATH)

        logger.info("Collaborative model loaded successfully.")

    # -----------------------------
    # RECOMMEND
    # -----------------------------
    def recommend(self, user_id: str, top_k: int = 10):

        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        all_movie_ids = self.movies_df["movieId"].values

        predictions = []

        # Predict ratings for this user
        for movie_id in all_movie_ids:
            pred = self.model.predict(user_id, movie_id).est
            predictions.append((movie_id, pred))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        top_movies = predictions[:top_k]
        top_movie_ids = [m_id for m_id, _ in top_movies]

        return self.movies_df[
            self.movies_df["movieId"].isin(top_movie_ids)
        ][["movieId", "title", "genres"]]

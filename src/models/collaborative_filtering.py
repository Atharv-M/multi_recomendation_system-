import pandas as pd
import logging 
import joblib
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split #Inspried by scikit Learn used for recommendation system 
from surprise import accuracy

from src.config import RATINGS_DIR,MASTER_DATASET_PATH,CF_MODEL_PATH

## Configuring Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Collaborative Filtering Model 
class CollaborativeFilter():
    def __init__(self, n_factors=100,n_epochs=50):
        self.n_factors=n_factors
        self.n_epochs=n_epochs
        self.model=None
        self.movies_df = None
    
    def fit(self):
        logger.info("Loading Data")
        ratings = pd.read_csv(RATINGS_DIR)

        reader = Reader(rating_scale=(0.5,5.0)) # Rating scale is from 0.5 to 5.0  it will be used to normalize the ratings 
        data = Dataset.load_from_df(
            ratings[["userId", "movieId", "rating"]],
            reader
        )

        trainset, _ = train_test_split(data, test_size=0.2)
        logger.info("Training SVD Collaborative filtering Model..")
        self.model = SVD( 
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            random_state=42,
          )
        self.model.fit(trainset)
        logger.info("Loading the Master Dataset")
        self.movies_df = pd.read_csv(MASTER_DATASET_PATH)

        logger.info("Collaborative Filtering Model Trained Successfully")

    
    def save(self):
        joblib.dump(self.model, CF_MODEL_PATH/"svd_model.pkl")
        joblib.dump(self.movies_df,CF_MODEL_PATH/"movies_df.pkl")
       
    def load(self):
        self.model = joblib.load(CF_MODEL_PATH/"svd_model.pkl")
        self.movies_df = joblib.load(CF_MODEL_PATH/"movies_df.pkl")
    
    def recommend(self, user_id, top_k=10):
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first")
        
        all_movie_ids = self.movies_df["movieId"].unique().tolist()
        predictions=[
            (movie_id, self.model.predict(user_id, movie_id).est)
            for movie_id in all_movie_ids
        ]
        predictions=sorted(
            predictions,
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        movie_ids = [p[0] for p in predictions]

        return self.movies_df[
            self.movies_df["movieId"].isin(movie_ids)
            ][["movieId", "title", "genres"]]
        
    
    

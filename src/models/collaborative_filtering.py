import pandas as pd
import logging 
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split #Inspried by scikit Learn used for recommendation system 
from surprise import accuracy

from src.config import RATINGS_DIR,MASTER_DATASET_PATH

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

        reader = Reader(rating_scale=(0.5,5.0))
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
        

    

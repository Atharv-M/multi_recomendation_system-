import numpy as np 
import pandas as pd
import logging
import joblib   
from src.config import MASTER_DATASET_PATH,CONTENT_MODEL_PATH

## Configuring Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self):
        self.movies_df=None
        self.topk_similarity=None
        
    def load_models(self):
        logger.info("Loading Master Dataset")
        self.movies_df = pd.read_csv(MASTER_DATASET_PATH)
       
        logger.info("Loading Top-K Similarity Model")
        self.topk_similarity = joblib.load(CONTENT_MODEL_PATH/"topk_movie_similarity.joblib")

    def get_recommendations(self,movie_id:int,top_k:int=10):
        if self.topk_similarity is None:
            raise ValueError("Model is not Loaded call load() first")
        
        if movie_id not in self.topk_similarity:
            raise ValueError(f"Movie ID {movie_id} not found in the dataset")
        
        similar_movies = self.topk_similarity[movie_id][:top_k]
        
        similar_movies_ids =[m_id for m_id, _ in similar_movies]
        
        recommendations = self.movies_df[self.movies_df["movieId"].isin(similar_movies_ids)][["movieId","title","genres"]   ]
        
        return recommendations
        
    
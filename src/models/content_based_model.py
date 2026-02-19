import numpy as np 
import pandas as pd
import logging
import joblib   
from src.config import MASTER_DATASET_PATH,CONTENT_MODEL_PATH,RATINGS_DIR,CF_MODEL_PATH,MOVIE_FEATURES_PATH,MOVIES_DF_PKL_PATH
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

## Configuring Logging 

logger = logging.getLogger(__name__)

def build_topk_similarity(top_k: int=20):
        
        logger.info("Loading movies Dataset")
        movies_df = joblib.load(MOVIES_DF_PKL_PATH)
        movie_features = joblib.load(MOVIE_FEATURES_PATH)

        logger.info("Calculating Cosine Similarity")
        logger.info("Computing Cosine Similarity Matrix (temporary full Matrix)")
        cosine_sim = cosine_similarity(movie_features,movie_features)


        logger.info("Building Top K Similarity Matrix")
        topk_similarity ={}

        movie_ids = movies_df["movieId"].tolist()

        for idx in tqdm(range(len(movie_ids))):
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores =[(i,s) for i,s in sim_scores if i!=idx]
            #Sort in decesending order based on similarity scores
            sim_scores.sort(key=lambda x:x[1], reverse=True)
            # Picking only topk similar movies 
            top_k_similarity =sim_scores[:top_k]

            topk_similarity[movie_ids[idx]] = [(movie_ids[i], float(score)) for i , score in top_k_similarity] 

        logger.info("Saving the Top K Similarity Matrix")
        joblib.dump(topk_similarity,CONTENT_MODEL_PATH/"topk_movie_similarity.joblib")
        CONTENT_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        logger.info(f"Top K Similarity Matrix saved to {CONTENT_MODEL_PATH/"topk_movie_similarity.joblib"}")  
        
        # Free Memory explicitly 
        del cosine_sim
        del movie_features

        logger.info("Full Cosine Matrix Deleted from memory") 

class ContentBasedRecommender:
    def __init__(self):
        self.movies_df=None
        self.topk_similarity=None

    def load(self):
        
        logger.info("Loading Master Dataset")
        self.movies_df = joblib.load(MOVIES_DF_PKL_PATH)
       
        logger.info("Loading Top-K Similarity Model")
        self.topk_similarity = joblib.load(CONTENT_MODEL_PATH/"topk_movie_similarity.joblib")

    def recommend(self,movie_id:int,top_k:int=10):
        if self.topk_similarity is None:
            raise ValueError("Model is not Loaded call load() first")
        
        if movie_id not in self.topk_similarity:
            raise ValueError(f"Movie ID {movie_id} not found in the dataset")
        
        similar_movies = self.topk_similarity[movie_id][:top_k]
        
        similar_movies_ids =[m_id for m_id, _ in similar_movies]
        
        recommendations = (
            self.movies_df[self.movies_df["movieId"].isin(similar_movies_ids)]
            .set_index("movieId")
            .loc[similar_movies_ids]
            .reset_index()
        )

        
        return recommendations
        
    
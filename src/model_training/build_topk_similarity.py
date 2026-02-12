import numpy as np 
import pandas as pd 
import joblib 
import logging 
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 
from pathlib import Path 
from src.config import MASTER_DATASET_PATH, SAVED_FEATURES_DIR, MOVIE_FEATURES_PATH,CONTENT_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_topk_similarity(top_k: int=20):
    logger.info("Loading movies Dataset")
    movies_df = pd.read_csv(MASTER_DATASET_PATH)
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
    logger.info(f"Top K Similarity Matrix saved to {CONTENT_MODEL_PATH/"topk_movie_similarity.joblib"}")  
    
    # Free Memory explicitly 
    del cosine_sim
    del movie_features

    logger.info("Full Cosine Matrix Deleted from memory")

    


if __name__ == "__main__":
    build_topk_similarity(top_k=20)
import numpy as np 
import pandas as pd
import logging
import joblib   
from src.config import MASTER_DATASET_PATH,CONTENT_MODEL_PATH,RATINGS_DIR,CF_MODEL_PATH,MOVIE_FEATURES_PATH,MOVIES_DF_PKL_PATH
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

## Configuring Logging 

logger = logging.getLogger(__name__)

def build_topk_similarity(top_k: int = 20, chunk_size: int = 2000):
    """
    Computes top-K similar movies using chunked cosine similarity.
    Processes `chunk_size` movies at a time instead of the full N×N matrix,
    so RAM usage stays at O(chunk_size × N) regardless of dataset size.
    Safe for 87K+ movies on a 16GB machine.
    """
    logger.info("Loading movies Dataset")
    movies_df     = joblib.load(MOVIES_DF_PKL_PATH)
    movie_features = joblib.load(MOVIE_FEATURES_PATH)
    n_movies       = movie_features.shape[0]
    movie_ids      = movies_df["movieId"].tolist()

    logger.info(f"Computing chunked Top-{top_k} similarity for {n_movies:,} movies "
                f"(chunk_size={chunk_size}, RAM-safe)")

    # Normalize feature rows once so dot-product == cosine similarity
    from sklearn.preprocessing import normalize
    from scipy.sparse import issparse
    if issparse(movie_features):
        features_norm = normalize(movie_features, norm="l2")
    else:
        features_norm = normalize(np.array(movie_features), norm="l2")

    topk_similarity = {}

    for start in tqdm(range(0, n_movies, chunk_size), desc="Chunked similarity"):
        end        = min(start + chunk_size, n_movies)
        chunk      = features_norm[start:end]          # shape: (chunk_size, F)

        # Dot product of chunk against ALL movies → cosine sim matrix (chunk_size × N)
        if issparse(chunk):
            sim_chunk = (chunk @ features_norm.T).toarray()
        else:
            sim_chunk = chunk @ features_norm.T        # numpy matmul

        for local_idx in range(end - start):
            global_idx = start + local_idx
            sim_row    = sim_chunk[local_idx]          # similarity to all N movies

            # Exclude self, get top-K indices
            sim_row[global_idx] = -1                   # mask self
            top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sim_row[top_indices])[::-1]]

            topk_similarity[movie_ids[global_idx]] = [
                (movie_ids[i], float(sim_row[i])) for i in top_indices
            ]

        del sim_chunk   # free chunk RAM immediately

    sim_save_path = CONTENT_MODEL_PATH / "topk_movie_similarity.joblib"
    CONTENT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving Top-{top_k} Similarity Matrix...")
    joblib.dump(topk_similarity, sim_save_path)
    logger.info(f"Saved to {sim_save_path}")

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
        
    
import numpy as np 
import pandas as pd
import logging
from pathlib import Path
import joblib   
from sklearn.metrics.pairwise import cosine_similarity
from src.config import MASTER_DATASET_PATH, SAVED_FEATURES_DIR, MOVIE_FEATURES_PATH

## Configuring Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

## Contnet  Based Recomendar 

class ContentBasedRecommender():
    def __init__(self):
        self.movies_df=None
        self.movie_features=None
        self.cosine_ma=None
        self.similarity_matrix=None
    
    def fit(self):
        logger.info("Loading Data")
        self.movies_df = pd.read_csv(MASTER_DATASET_PATH)
        logger.info("Loading the Precomputed Features")
        self.movie_features = joblib.load(MOVIE_FEATURES_PATH)

        logger.info("Calculating Cosine Similarity")
        self.cosine_ma = cosine_similarity(self.movie_features)

        logger.info("Content Based Model Trained Successfully ")
        
    def recommend(self, movie_id, top_k=10):
        if self.cosine_ma is None:
            raise ValueError("Model is Not trained. call fit() first")
        
        # Now Get the Index of the Movie 
        movie_indx = self.movies_df.index[self.movies_df["movieId"] == movie_id].tolist()
        if not movie_indx:
            raise ValueError(f"Movie with ID {movie_id} not Found in the dataset")
        
        movie_indx = movie_indx[0]

        ## Get the Similarity Scores 
        similarity_scores = list(
            enumerate(self.cosine_ma[movie_indx])
        )

        ## sort the movies based on similarity scores 
        similarity_scores = sorted(
            similarity_scores,
            key = lambda x: x[1],
            reverse= True
        )

        ## Exclude the movie itself from recommendations 
        similarity_scores = similarity_scores[1:top_k+1]

        # Get the Recommended Movie indices
        movie_indices = [i[0] for i in similarity_scores]
        return self.movies_df.iloc[movie_indices][["movieId", "title",  "genres"]]

import pandas as pd
import logging
import joblib
from src.config import MASTER_DATASET_PATH,MIN_VOTES_PERCENTILE,POPULARITY_MODEL_PATH,ARTIFACTS_DIR,MOVIES_DF_PKL_PATH

## Configuring Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PopularityRecommender():  
    def __init__(self, min_votes_percentile=MIN_VOTES_PERCENTILE):
        self.min_votes_percentile = min_votes_percentile
        self.master_df = None
    
    def fit(self):

        logger.info("Loading Master Dataset")
        df = joblib.load(MOVIES_DF_PKL_PATH)


        """ Using a weighted rating formula (IMDb-style):

        score= v/(v+m)*R + m/(v+m)*C
        Where:
        R = movie’s average rating
        v = rating count
        C = global mean rating
        m = minimum votes threshold
        This prevents movies with few ratings from ranking too high.
        """
        
        # Global Average Rating
        C = df["avg_rating"].mean()

        # Minimum Number of Votes threshold 
        m=df["rating_count"].quantile(self.min_votes_percentile)


        logger.info(f"Global mean rating (C): {C:.2f}")
        logger.info(f"Minimum Votes Threshold (m): {m:.2f}")

        # Weighted Rating formula 
        df["popularity_score"]=(
            (df["rating_count"]/df["rating_count"]+m)*df["avg_rating"]
            +(m/(df["rating_count"]+m))*C
        )

        self.master_df = df.sort_values(by="popularity_score",ascending=False)
        logger.info("Popularity model trained  successfully")

    def save(self):
        joblib.dump(self.master_df, POPULARITY_MODEL_PATH/"popularity_ranked.pkl")
        logger.info(f"Popularity model saved to {POPULARITY_MODEL_PATH/"popularity_ranked.pkl"}")
    
    def load(self):
        self.master_df = joblib.load(POPULARITY_MODEL_PATH/"popularity_ranked.pkl")
        logger.info(f"Popularity model loaded from {POPULARITY_MODEL_PATH/"popularity_ranked.pkl"}")

    def recommend(self, top_k=10):
        if self.master_df is None:
            raise ValueError("Model is not trained. Call fit() first")
        
        return self.master_df[["movieId", "title", "popularity_score", "genres"]].head(top_k)     
       
    
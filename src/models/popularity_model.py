import pandas as pd
import logging
from src.config import MASTER_DATASET_PATH,MIN_VOTES_PERCENTILE

## Configuring Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def build_popularity_model():
    def __init__(self, min_votes_percentile=MIN_VOTES_PERCENTILE):
        self.min_votes_percentile = min_votes_percentile
        self.master_df = None
    
    def fit(self):
        logger.info("Loading Master Dataset")
        df = pd.read_csv(MASTER_DATASET_PATH)
        
        # Global Average Rating
        C = df["avg_rating"].mean()

        # Minimum Number of Votes threshold 
        m=df["rating_count"].quantile(self.min_votes_percentile)
        



   
if __name__ == "__main__":
    build_popularity_model()
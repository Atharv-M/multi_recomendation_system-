import pandas as pd
import logging
from pathlib import Path

#### Configure logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#### Define paths

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the raw data from the raw data directory"""
    logger.info("Loading raw datasets")
    movies = pd.read_csv(RAW_DATA_DIR / "movie.csv")
    ratings =pd.read_csv(RAW_DATA_DIR / "rating.csv")
    tags = pd.read_csv(RAW_DATA_DIR/"tag.csv")
    links = pd.read_csv(RAW_DATA_DIR /"links.csv")
    genome_tags = pd.read_csv(RAW_DATA_DIR /"genome_tags.csv")
    genome_scores = pd.read_csv(RAW_DATA_DIR /"genome_scores.csv")
    return movies,ratings,tags,links,genome_tags,genome_scores

## Rating Aggregation 
""" We need to Aggregate the Ratings as it have one to many Relation in Datasets"""

def aggregate_ratings(ratings):
    logger.info("Aggregating Ratings")
    return ratings.groupby("movieId").agg(
        avg_rating = ("rating", "mean"),
        rating_count = ("rating", "count")
    ).reset_index()

## Tag Processing

def aggregate_user_tags(tags):
    logger.info("Aggregating User Tags")
    return tags.groupby("movieId")['tag'].apply(lambda x: " ".join(x.astype(str))).reset_index()
   

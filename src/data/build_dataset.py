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
    links = pd.read_csv(RAW_DATA_DIR /"link.csv")
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
    return tags.groupby("movieId")['tag'].apply(lambda x: " ".join(x.fillna('').astype(str))).reset_index()
   
## Genome Tags Preprocessing 
def prcess_genome_data(genome_scores, genome_tags, threshold=0.5):
    logger.info("processing genome tag relevance data ")
    genome = genome_scores.merge(
        genome_tags,
        on = "tagId",
        how = "left"
    )
    genome = genome[genome["relevance"]>=threshold]
    return genome.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str))).reset_index().rename(columns={"tag":"genome_tags"})

## Building Master Dataset 

def build_master_dataset():
    logger.info("Building MAster Dataset")
    movies, ratings, tags,links, genome_tags, genome_scores = load_data()
    rating_agg = aggregate_ratings(ratings)
    tag_add = aggregate_user_tags(tags)
    genome_agg = prcess_genome_data(genome_scores, genome_tags)
    
    logger.info("merging all Datasets")
    master_df = (
        movies
        .merge(rating_agg, on ="movieId", how = "left")
        .merge(tag_add, on ='movieId', how ='left')
        .merge(genome_agg, on ='movieId',how = 'left')
        .merge(links, on="movieId", how="left")
    )
    logger.info("Handling missing Values")

    master_df["avg_rating"].fillna(0, inplace=True)
    master_df["rating_count"].fillna(0,inplace=True)
    master_df["tag"].fillna("", inplace=True)
    master_df["genome_tags"].fillna("",inplace=True)

    master_df["genres"] = master_df["genres"].str.replace("|", " ", regex=False)

    master_df.to_csv(PROCESSED_DATA_DIR / "master_dataset.csv", index=False)
    logger.info("Master dataset saved to data/processed/master_dataset.csv")
    logger.info(f"Master dataset shape: {master_df.shape}" )

    return master_df


if __name__== "__main__":
    build_master_dataset()
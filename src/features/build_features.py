import pandas as pd
import numpy as np 
import re
import logging 
from pathlib import Path
import joblib 

from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from scipy.sparse import hstack

## Configureing Logging 
logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger =logging.getLogger(__name__)

import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import MASTER_DATASET_PATH, SAVED_FEATURES_DIR, TFIDF_PARAMS, MOVIES_DF_PKL_PATH, LINKS_PKL_PATH, RAW_DATA_DIR


## Text Cleaning 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

## Build Features 
def build_features():
    
    logger.info("Loading Processed Dataset")
    df = pd.read_csv(MASTER_DATASET_PATH)

    # Save the dataframe as a pickle file for faster loading
    logger.info(f"Saving Master Dataset to {MOVIES_DF_PKL_PATH}")
    joblib.dump(df, MOVIES_DF_PKL_PATH)

    logger.info("Loading Links Dataset")
    links_df = pd.read_csv(RAW_DATA_DIR / "link.csv")
    
    logger.info(f"Saving Links Dataset to {LINKS_PKL_PATH}")
    joblib.dump(links_df, LINKS_PKL_PATH)

    ## Text Features
    """
    combining all the text features from the dataset to create a single text feature for each movie.
    """
    
    logger.info("Building Text Features")
    df["text_features"] = (
        df["genres"].fillna("") + " " 
        + df["tag"].fillna("") + " " 
        + df["genome_tags"].fillna(""))
    df["text_features"]=df["text_features"].apply(clean_text)

    ## Using TfidfVectorizer to convert text features into numeric features
    tfidf=TfidfVectorizer(**TFIDF_PARAMS)

    text_vectors = tfidf.fit_transform(df["text_features"])

    ## Genre Features 

    """
    using MultiLabelBinarizer to convert the genre column into a binary matrix.
    for each genre, create a column and fill it with 1 if the movie belongs to that genre and 0 otherwise.
    """

    logger.info("Building Genre Features")
    mlb = MultiLabelBinarizer()
    df["genre_list"] = df["genres"].str.split("|")
    genre_vectors = mlb.fit_transform(df["genre_list"])

    ## Numeric Features 

    """
    Scaling the numeric features to make them comparable to each other.
    first apply log transformation to the rating_count column to reduce skewness.
    then apply standard scaling to the avg_rating and rating_count_log columns.
    """
    df["avg_rating"].fillna(0,inplace=True)
    df["rating_count"].fillna(0,inplace=True) ## Helps in Cosine similarity Calculation 

    logger.info("Scaling Numeric Features")
    df["rating_count_log"] = np.log1p(df["rating_count"])
    numeric_data = df[["avg_rating", "rating_count_log"]].values
    # Replace any remaining NaN values with 0
    numeric_data = np.nan_to_num(numeric_data)
    scaler = StandardScaler()
    numeric_vectors = scaler.fit_transform(numeric_data)

    ## Merging all Features 

    """
    combining all the features into a single feature matrix.
    """

    logger.info("Merging all Features ")
    movie_features = hstack([
        text_vectors,
        genre_vectors,
        numeric_vectors
    ])
    ## Check for NaNs
    if issparse(movie_features):
        assert not np.isnan(movie_features.data).any(), "NaNs in sparse features"
    else:
        assert not np.isnan(movie_features).any(), "NaNs in features"
    
    ## Saving Features 

    """
    Saving all the features into joblib files.
    """

    joblib.dump(movie_features,SAVED_FEATURES_DIR / "movie_features.joblib")
    joblib.dump(tfidf,SAVED_FEATURES_DIR / "tfidf.joblib")
    joblib.dump(mlb,SAVED_FEATURES_DIR / "mlb.joblib")
    joblib.dump(scaler,SAVED_FEATURES_DIR / "scaler.joblib")
    

    logger.info(f"Final feature matrix shape: {movie_features.shape}")

if __name__ == "__main__":
    build_features()
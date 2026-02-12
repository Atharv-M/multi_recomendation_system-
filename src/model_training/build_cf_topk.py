import pandas as pd
import numpy as np
import joblib
import logging
from tqdm import tqdm
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from src.config import RATINGS_DIR, MASTER_DATASET_PATH, CF_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def build_cf_topk(top_k:int=20):
    logger.info("Loading ratings data ")
    ratings_df = pd.read_csv(RATINGS_DIR)
    reader =Reader(rating_scale=(0.5,5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    trainset,_ = train_test_split(data,test_size =0.2)
    logger.info("Training SVD Model")
    model =SVD(n_factors=100, n_epochs=20,random_state=42)
    model.fit(trainset)

    logger.info("loading movie Dataset")
    movies_df = pd.read_csv(MASTER_DATASET_PATH)
    all_movie_ids = movies_df["movieId"].unique()

    logger.info("Building Top-K Colaborative Filtering Recomendations")
    user_topk={}

    user_ids=ratings_df["userId"].unique()

    for user_id in tqdm(user_ids):
        predictions=[]
        for movie_id in all_movie_ids:
            pred = model.predict(user_id,movie_id).est
            predictions.append((movie_id,pred))
        
        predictions.sort(key=lambda x:x[1],reverse=True)
        user_topk[user_id] = predictions[:top_k]

    logger.info("Saving the Top-K CF Recomendations")
    joblib.dump(user_topk,CF_MODEL_PATH/"user_topk_cf.joblib")
    logger.info("Top-K CF Recomendations Saved")


if __name__ == "__main__":
    build_cf_topk(top_k=20)
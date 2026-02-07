import pandas as pd
import joblib
from pathlib import Path
from src.config import MASTER_DATASET_PATH

df = pd.read_csv("data/processed/master_dataset.csv")

movies_df = df[["movieId", "title", "genres"]]

Path("artifacts/metadata").mkdir(parents=True, exist_ok=True)
joblib.dump(movies_df, "artifacts/metadata/movies_df.pkl")



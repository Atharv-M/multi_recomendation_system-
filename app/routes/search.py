from fastapi import APIRouter, Query
import pandas as pd
from app.config import MASTER_DATASET_PATH

router = APIRouter()

movies_df = pd.read_csv(MASTER_DATASET_PATH)


@router.get("/search")
def search_movies(query: str = Query(..., min_length=2)):

    results = movies_df[
        movies_df["title"].str.contains(query, case=False, na=False)
    ][["movieId", "title"]].head(10)

    return results.to_dict(orient="records")

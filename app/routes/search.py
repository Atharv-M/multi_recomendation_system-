from fastapi import APIRouter, Query
import pandas as pd
import joblib
from src.config import MOVIES_DF_PKL_PATH

router = APIRouter()

movies_df = joblib.load(MOVIES_DF_PKL_PATH)


@router.get("/search")
def search_movies(query: str = Query(..., min_length=2)):

    results = movies_df[
        movies_df["title"].str.contains(query, case=False, na=False)
    ][["movieId", "title"]].head(10)

    return results.to_dict(orient="records")

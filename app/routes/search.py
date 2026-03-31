from fastapi import APIRouter, Query, Depends
import pandas as pd
import joblib
from src.config import MOVIES_DF_PKL_PATH
from app.dependencies import get_links_df
from app.utils.enrichment import enrich_movies

router = APIRouter()

movies_df = joblib.load(MOVIES_DF_PKL_PATH)

# --- LIGHTWEIGHT search: used by HOME SCREEN typeahead ---
# Returns only movieId + title. No OMDB calls. Extremely fast.
@router.get("/search/simple")
def search_movies_simple(query: str = Query(..., min_length=2)):
    results = movies_df[
        movies_df["title"].str.contains(query, case=False, na=False, regex=False)
    ][["movieId", "title"]].head(10)
    return results.to_dict(orient="records")

# --- ENRICHED search: used by FEEDBACK / ONBOARDING form ---
# Returns full movie details including OMDB poster, rating, overview.
@router.get("/search")
def search_movies(
    query: str = Query(..., min_length=2),
    links_df: pd.DataFrame = Depends(get_links_df)
):
    results = movies_df[
        movies_df["title"].str.contains(query, case=False, na=False, regex=False)
    ][["movieId", "title", "genres"]].head(10)

    enriched_results = enrich_movies(results, links_df)
    return enriched_results

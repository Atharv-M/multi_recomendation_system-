from fastapi import APIRouter, Depends, HTTPException
from app.auth.supabase_auth import get_current_user_object, get_current_user
from app.database import supabase
import joblib
from src.config import MOVIES_DF_PKL_PATH

router = APIRouter(prefix="/feedback", tags=["Feedback"])

# Preload movies lookup once
_movies_df = None
def get_movies_df():
    global _movies_df
    if _movies_df is None:
        _movies_df = joblib.load(MOVIES_DF_PKL_PATH)
    return _movies_df

@router.post("/rate")
def rate_movie(
    movie_id: int,
    rating: float,
    user_obj = Depends(get_current_user_object)
):
    if rating < 0.5 or rating > 5:
        raise HTTPException(status_code=400, detail="Invalid rating")

    user_id = user_obj.id
    user_name = user_obj.user_metadata.get("full_name", "Unknown User")

    try:
        # Check if this user has already rated this movie
        existing = supabase.table("ratings") \
            .select("id") \
            .eq("user_id", user_id) \
            .eq("movie_id", movie_id) \
            .limit(1) \
            .execute()

        if existing.data:
            # UPDATE the existing rating instead of inserting a duplicate
            supabase.table("ratings") \
                .update({"rating": rating}) \
                .eq("user_id", user_id) \
                .eq("movie_id", movie_id) \
                .execute()
            return {"message": "Rating updated successfully", "action": "updated"}
        else:
            # INSERT brand new rating
            supabase.table("ratings").insert({
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
                "user_name": user_name
            }).execute()
            return {"message": "Rating saved successfully", "action": "created"}

    except Exception as e:
        print(f"Supabase Backend Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Database Rejected Rating: {str(e)}")


@router.get("/my-ratings")
def get_my_ratings(user_id: str = Depends(get_current_user)):
    """Returns the logged-in user's 20 most recent ratings — private."""
    try:
        response = supabase.table("ratings") \
            .select("movie_id, rating, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(20) \
            .execute()

        rows = response.data or []
        movies_df = get_movies_df()
        movie_lookup = movies_df.set_index("movieId")[["title", "genres"]].to_dict("index")

        enriched = []
        for row in rows:
            mid = row.get("movie_id")
            meta = movie_lookup.get(mid, {})
            enriched.append({
                "movie_id": mid,
                "title": meta.get("title", "Unknown Movie"),
                "genres": meta.get("genres", ""),
                "rating": row.get("rating"),
                "rated_at": row.get("created_at")
            })
        return enriched
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/community/feed")
def get_community_feed():
    """Returns the public recent community activity — no auth required."""
    try:
        response = supabase.table("ratings") \
            .select("movie_id, rating, user_name, created_at") \
            .order("created_at", desc=True) \
            .limit(60) \
            .execute()

        rows = response.data or []
        movies_df = get_movies_df()
        movie_lookup = movies_df.set_index("movieId")[["title", "genres"]].to_dict("index")

        enriched = []
        for row in rows:
            mid = row.get("movie_id")
            meta = movie_lookup.get(mid, {})
            enriched.append({
                "movie_id": mid,
                "title": meta.get("title", "Unknown Movie"),
                "genres": meta.get("genres", ""),
                "rating": row.get("rating"),
                "user_name": row.get("user_name", "Anonymous"),
                "rated_at": row.get("created_at")
            })
        return enriched
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
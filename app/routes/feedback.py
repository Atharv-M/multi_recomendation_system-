from fastapi import APIRouter, Depends, HTTPException
from app.auth.supabase_auth import get_current_user
from app.database import supabase  # your supabase client

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/rate")
def rate_movie(
    movie_id: int,
    rating: float,
    current_user: str = Depends(get_current_user)
):
    if rating < 0.5 or rating > 5:
        raise HTTPException(status_code=400, detail="Invalid rating")

    response = supabase.table("ratings").insert({
        "user_id": current_user,
        "movie_id": movie_id,
        "rating": rating
    }).execute()

    return {"message": "Rating saved successfully"}
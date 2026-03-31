from fastapi import APIRouter, Depends, HTTPException
from app.auth.supabase_auth import get_current_user_object
from app.database import supabase  

router = APIRouter(prefix="/feedback", tags=["Feedback"])

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
        response = supabase.table("ratings").insert({
            "user_id": user_id,
            "movie_id": movie_id,
            "rating": rating,
            "user_name": user_name
        }).execute()
        return {"message": "Rating saved successfully"}
    except Exception as e:
        print(f"Supabase Backend Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Database Rejected Rating: {str(e)}")
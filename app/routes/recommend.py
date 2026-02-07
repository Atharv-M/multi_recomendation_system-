from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from app.dependencies import get_hybrid_model
from app.schemas import RecommendationResponse, RecommendationList
from app.auth.supabase_auth import get_current_user

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


@router.get("/cold-start", response_model=RecommendationList)
def cold_start_recommendations(
    top_k: int = Query(10, ge=1, le=50),
    model=Depends(get_hybrid_model)
):
    df = model.recommend(top_k=top_k)

    return {
        "recommendations": df.to_dict(orient="records")
    }


@router.get("/user/{user_id}", response_model=RecommendationList)
def user_recommendations(
    user_id: int,
    top_k: int = Query(10, ge=1, le=50),
    model=Depends(get_hybrid_model)
):
    df = model.recommend(user_id=user_id, top_k=top_k)

    return {
        "recommendations": df.to_dict(orient="records")
    }


@router.get("/similar/{movie_id}", response_model=RecommendationList)
def similar_movies(
    movie_id: int,
    top_k: int = Query(10, ge=1, le=50),
    model=Depends(get_hybrid_model)
):
    df = model.recommend(movie_id=movie_id, top_k=top_k)

    return {
        "recommendations": df.to_dict(orient="records")
    }
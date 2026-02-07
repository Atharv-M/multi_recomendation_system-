from pydantic import BaseModel 
from typing import Optional, List

class RecommendationResponse(BaseModel):
    movieId: int
    title: str
    genres: str

class RecommendationList(BaseModel):
    recommendations: List[RecommendationResponse]
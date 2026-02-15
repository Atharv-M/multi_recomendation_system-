from pydantic import BaseModel 
from typing import Optional, List

class RecommendationResponse(BaseModel):
    movieId: int
    title: str
    genres: str
    poster: Optional[str] = None
    rating: Optional[float] = None
    overview: Optional[str] = None

class RecommendationList(BaseModel):
    recommendations: List[RecommendationResponse]
    
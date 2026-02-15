import logging
from fastapi import FastAPI
from app.routes import recommend, search

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Movie Recommendation System",
    description="Hybrid recommendation system using popularity, content-based and collaborative filtering",
    version="1.0.0"
)

app.include_router(recommend.router)
app.include_router(search.router)


@app.get("/")
def health_check():
    return {"status": "OK", "message": "Movie Recommender API is running"}

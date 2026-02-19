import requests
import logging
from app.config import OMDB_API_KEY

logger = logging.getLogger(__name__)

BASE_URL = "http://www.omdbapi.com/"

def get_movie_details(imdb_id: str):

    if not imdb_id:
        return None

    # IMDb IDs are usually 7 digits padded with zeros, e.g., tt0114709
    imdb_id = f"tt{int(imdb_id):07d}"

    url = BASE_URL
    params = {
        "i": imdb_id,
        "apikey": OMDB_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        logger.info(f"OMDB API Status: {response.status_code} for IMDb ID: {imdb_id}")
        if response.status_code != 200:
            return None
            
        data = response.json()
        if data.get("Response") == "False":
            return None
            
        return {
            "poster": data.get("Poster"),
            "rating": data.get("imdbRating"),
            "overview": data.get("Plot")
        }
    except Exception as e:
        logger.error(f"Error fetching OMDB data: {e}")
        return None

    return {
        "poster": data.get("Poster"),
        "rating": data.get("imdbRating"),
        "overview": data.get("Plot")
    }

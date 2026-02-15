import pandas as pd
from app.utils.omdb import get_movie_details

from concurrent.futures import ThreadPoolExecutor, as_completed

def enrich_movies(recommendations_df: pd.DataFrame, links_df: pd.DataFrame):
    
    # Create a dictionary for O(1) lookup: movieId -> imdbId
    imdb_map = links_df.set_index("movieId")["imdbId"].to_dict()
    enriched = []
    
    # Helper function for parallel execution
    def fetch_details(row):
        movie_id = row["movieId"]
        imdb_id = imdb_map.get(movie_id)
        omdb_data = None
        
        if imdb_id is not None and not pd.isna(imdb_id):
            try:
                # OMDB ID is usually an integer in the CSV, get_movie_details handles the "tt" prefix
                omdb_data = get_movie_details(imdb_id)
            except Exception as e:
                print(f"Error fetching OMDB data for movie {movie_id}: {e}")
        
        return {
            "movieId": movie_id,
            "title": row["title"],
            "genres": row["genres"],
            "poster": omdb_data["poster"] if omdb_data else None,
            "rating": omdb_data["rating"] if omdb_data else None,
            "overview": omdb_data["overview"] if omdb_data else None
        }

    # Use ThreadPoolExecutor to fetch details in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Convert DataFrame rows to a list of row objects (Series)
        rows = [row for _, row in recommendations_df.iterrows()]
        enriched = list(executor.map(fetch_details, rows))

    return enriched

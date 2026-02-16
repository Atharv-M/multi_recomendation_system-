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
        
        imdb_url = None
        if imdb_id is not None and not pd.isna(imdb_id):
            # Format IMDb ID with 'tt' prefix and 7 digits padding
            formatted_imdb_id = f"tt{int(imdb_id):07d}"
            imdb_url = f"https://www.imdb.com/title/{formatted_imdb_id}/"
            
            try:
                # OMDB ID is usually an integer in the CSV, get_movie_details handles the "tt" prefix
                omdb_data = get_movie_details(imdb_id)
            except Exception:
                print(f"Error fetching OMDB data for movie {movie_id}")
        
        return {
            "movieId": movie_id,
            "title": row["title"],
            "genres": row["genres"],
            "poster": omdb_data["poster"] if omdb_data else None,
            "rating": omdb_data["rating"] if omdb_data else None,
            "overview": omdb_data["overview"] if omdb_data else None,
            "imdb_link": imdb_url
        }

    # Use ThreadPoolExecutor to fetch details in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Convert DataFrame rows to a list of row objects (Series)
        rows = [row for _, row in recommendations_df.iterrows()]
        enriched = list(executor.map(fetch_details, rows))

    return enriched

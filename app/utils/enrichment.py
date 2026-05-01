import pandas as pd
from app.utils.omdb import get_movie_details
from concurrent.futures import ThreadPoolExecutor


def enrich_movies(
    recommendations_df: pd.DataFrame,
    links_df: pd.DataFrame,
    include_trailer: bool = False,   # OFF by default — saves YouTube API quota
):
    """
    Enriches a movie recommendations DataFrame with OMDB metadata and,
    optionally, a YouTube trailer ID.

    Args:
        recommendations_df: DataFrame with at least [movieId, title, genres].
        links_df:           DataFrame mapping movieId → imdbId.
        include_trailer:    If True, fetches a YouTube trailer ID for each movie.
                            Keep False for search/onboarding flows to preserve
                            YouTube API daily quota (10,000 units / 100 searches).
    """
    imdb_map = links_df.set_index("movieId")["imdbId"].to_dict()

    def fetch_details(row):
        movie_id = row["movieId"]
        imdb_id = imdb_map.get(movie_id)

        # --- YouTube trailer (only when explicitly requested) ---
        trailer_id = None
        if include_trailer:
            from app.utils.tmdb import get_youtube_trailer_id
            trailer_id = get_youtube_trailer_id(row["title"])

        # --- OMDB poster / rating / overview ---
        omdb_data = None
        imdb_url = None
        if imdb_id is not None and not pd.isna(imdb_id):
            formatted_imdb_id = f"tt{int(imdb_id):07d}"
            imdb_url = f"https://www.imdb.com/title/{formatted_imdb_id}/"
            try:
                omdb_data = get_movie_details(imdb_id)
            except Exception:
                pass

        return {
            "movieId": movie_id,
            "title": row["title"],
            "genres": row["genres"],
            "poster": omdb_data["poster"] if omdb_data else None,
            "rating": omdb_data["rating"] if omdb_data else None,
            "overview": omdb_data["overview"] if omdb_data else None,
            "imdb_link": imdb_url,
            "youtube_trailer_id": trailer_id,
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        rows = [row for _, row in recommendations_df.iterrows()]
        enriched = list(executor.map(fetch_details, rows))

    return enriched

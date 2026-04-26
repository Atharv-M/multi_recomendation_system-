import urllib.request
import urllib.parse
import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def _get_api_key() -> str | None:
    """Lazily load the YouTube API key so it's always read after .env is loaded."""
    from app.config import YOUTUBE_API_KEY
    return YOUTUBE_API_KEY


@lru_cache(maxsize=1024)
def get_youtube_trailer_id(movie_title: str) -> str | None:
    """
    Fetches the official trailer video ID for a movie using the YouTube Data API v3.

    Uses the search endpoint with:
      - query  = "{movie_title} official trailer"
      - type   = video
      - order  = relevance
      - maxResults = 1

    Returns the 11-character video ID (e.g. 'dQw4w9WgXcQ') or None if:
      - The API key is not configured
      - No results are found
      - The API quota is exceeded
      - Any network / parse error occurs
    """
    if not movie_title:
        return None

    api_key = _get_api_key()
    if not api_key:
        logger.warning("YOUTUBE_API_KEY not set — trailer lookup skipped.")
        return None

    # Strip surrounding quotes that may have been added in .env
    api_key = api_key.strip('"').strip("'")

    try:
        query = urllib.parse.quote_plus(f"{movie_title} official trailer")
        url = (
            f"https://www.googleapis.com/youtube/v3/search"
            f"?part=id"
            f"&q={query}"
            f"&type=video"
            f"&order=relevance"
            f"&maxResults=1"
            f"&key={api_key}"
        )

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        items = data.get("items", [])
        if items:
            video_id = items[0]["id"].get("videoId")
            logger.info(f"YouTube API: found trailer '{video_id}' for '{movie_title}'")
            return video_id

        logger.info(f"YouTube API: no results for '{movie_title}'")
        return None

    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 403:
            logger.error(f"YouTube API quota exceeded or forbidden for '{movie_title}': {body[:200]}")
        else:
            logger.error(f"YouTube API HTTP {e.code} for '{movie_title}': {body[:200]}")
        return None
    except Exception as e:
        logger.error(f"YouTube API error for '{movie_title}': {e}")
        return None

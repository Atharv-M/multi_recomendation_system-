import urllib.request
import urllib.parse
import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def _get_api_keys() -> list[str]:
    """
    Returns the list of configured YouTube API keys in priority order.
    Strips any accidental surrounding quotes added in .env files.
    """
    from app.config import YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2
    keys = []
    for k in [YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2]:
        if k:
            keys.append(k.strip('"').strip("'"))
    return keys


def _call_youtube_api(api_key: str, query: str) -> str | None:
    """
    Single API call to the YouTube Data API v3 search endpoint.
    Returns the video ID string, or raises urllib.error.HTTPError on failure.
    """
    encoded_query = urllib.parse.quote_plus(query)
    url = (
        f"https://www.googleapis.com/youtube/v3/search"
        f"?part=id"
        f"&q={encoded_query}"
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
        return items[0]["id"].get("videoId")
    return None


@lru_cache(maxsize=1024)
def get_youtube_trailer_id(movie_title: str) -> str | None:
    """
    Fetches the official trailer video ID for a movie using the YouTube Data API v3.

    Key rotation strategy:
      - Tries YOUTUBE_API_KEY_1 first.
      - If that key returns HTTP 403 (quota exceeded / forbidden), automatically
        falls back to YOUTUBE_API_KEY_2.
      - If both keys fail, returns None silently.

    Results are cached in-process with @lru_cache so the same title is
    never fetched more than once per server restart.

    Returns the 11-character video ID (e.g. 'dQw4w9WgXcQ') or None.
    """
    if not movie_title:
        return None

    api_keys = _get_api_keys()
    if not api_keys:
        logger.warning("No YOUTUBE_API_KEY configured — trailer lookup skipped.")
        return None

    query = f"{movie_title} official trailer"

    for idx, api_key in enumerate(api_keys, start=1):
        try:
            video_id = _call_youtube_api(api_key, query)
            if video_id:
                logger.info(f"YouTube API key {idx}: found trailer '{video_id}' for '{movie_title}'")
            else:
                logger.info(f"YouTube API key {idx}: no results for '{movie_title}'")
            return video_id  # even None is a valid final answer if no results found

        except urllib.error.HTTPError as e:
            if e.code == 403:
                body = e.read().decode(errors="replace")
                if idx < len(api_keys):
                    logger.warning(
                        f"YouTube API key {idx} quota exhausted / forbidden for '{movie_title}' "
                        f"— switching to key {idx + 1}. Details: {body[:120]}"
                    )
                    continue  # try next key
                else:
                    logger.error(
                        f"All YouTube API keys exhausted for '{movie_title}'. "
                        f"Last error: HTTP 403 — {body[:120]}"
                    )
                    return None
            else:
                logger.error(f"YouTube API HTTP {e.code} for '{movie_title}'")
                return None

        except Exception as e:
            logger.error(f"YouTube API error for '{movie_title}': {e}")
            return None

    return None

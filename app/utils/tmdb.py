import urllib.request
import urllib.parse
import re
import logging
import ssl
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

# Detect HuggingFace Spaces environment — outbound SSL connections to YouTube
# are blocked there, so we skip scraping entirely to avoid hanging workers.
_IS_HUGGINGFACE = bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_AUTHOR_NAME"))

# Shared single-thread executor used as an escape hatch for the hard timeout
_yt_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yt_scraper")


def _scrape_youtube(movie_title: str):
    """Inner blocking call — run inside a future so we can impose a hard timeout."""
    search_keyword = f"{movie_title} official trailer"
    query = urllib.parse.quote_plus(search_keyword)
    url = f"https://www.youtube.com/results?search_query={query}"

    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # urlopen timeout = 3 s; the Future.result timeout below is the real guard
    with urllib.request.urlopen(req, timeout=3, context=ctx) as resp:
        html = resp.read().decode(errors="replace")

    video_ids = re.findall(r"watch\?v=(\S{11})", html)
    return video_ids[0] if video_ids else None


@lru_cache(maxsize=1024)
def get_youtube_trailer_id(movie_title: str):
    """
    Scrapes YouTube search results to find the official trailer video ID.
    Returns the 11-character hash (e.g. 'dQw4w9WgXcQ') or None.

    On HuggingFace Spaces the function returns None immediately because
    outbound connections to YouTube are blocked by the platform.
    """
    if not movie_title:
        return None

    # HuggingFace blocks YouTube — skip immediately to avoid SSL hangs
    if _IS_HUGGINGFACE:
        return None

    try:
        future = _yt_executor.submit(_scrape_youtube, movie_title)
        return future.result(timeout=5)   # hard wall-clock timeout: 5 s
    except FuturesTimeoutError:
        logger.warning(f"YouTube scrape timed out for: {movie_title}")
        return None
    except Exception as e:
        logger.error(f"Error natively scraping YouTube for {movie_title}: {e}")
        return None

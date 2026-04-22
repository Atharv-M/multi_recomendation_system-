import urllib.request
import urllib.parse
import re
import logging
import ssl
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1024)
def get_youtube_trailer_id(movie_title: str):
    """
    Intelligently scrapes YouTube search results to locate the official trailer 
    ID for a movie
    Returns the 11-character hash (e.g., 'dQw4w9WgXcQ') or None.
    """
    if not movie_title:
        return None

    try:
        search_keyword = f"{movie_title} official trailer"
        query = urllib.parse.quote_plus(search_keyword)
        url = f"https://www.youtube.com/results?search_query={query}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        html = urllib.request.urlopen(req, timeout=2, context=ctx)
        
        # Regex to locate the 11-char sequence behind the watch param
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        
        if video_ids:
            return video_ids[0]
            
        return None
    except Exception as e:
        logger.error(f"Error natively scraping YouTube for {movie_title}: {e}")
        return None

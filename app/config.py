import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from app directory
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

SUPABASE_PROJECT_URL = os.getenv("SUPABASE_PROJECT_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_JWT_ALGORITHM = os.getenv("SUPABASE_JWT_ALGORITHM", "HS256")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")


# Safety checks (fail fast)
if not SUPABASE_JWT_SECRET:
    raise RuntimeError("SUPABASE_JWT_SECRET not found in .env")
MASTER_DATASET_PATH = Path(__file__).parent.parent / "data" / "processed" / "master_dataset.csv"
LINKS_DATASET_PATH = Path(__file__).parent.parent / "data" / "raw" / "link.csv"
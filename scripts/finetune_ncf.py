"""
Fine-tune NCF on NEW Supabase ratings only.

Watermark system:
  - Reads  scripts/.finetune_watermark.json  → last_finetuned_at (ISO timestamp)
  - Only fetches ratings with created_at > last_finetuned_at
  - After a successful run, writes the current UTC timestamp back as the new watermark

This ensures we NEVER retrain on ratings the model has already seen.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` and `app` are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import logging
import pandas as pd
from datetime import datetime, timezone

from app.database import supabase
from src.models.ncf import NeuralCollaborativeRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

WATERMARK_FILE = Path(__file__).parent / ".finetune_watermark.json"


def load_watermark() -> str | None:
    """Returns the ISO timestamp of the last fine-tune, or None if first run."""
    if WATERMARK_FILE.exists():
        data = json.loads(WATERMARK_FILE.read_text())
        ts = data.get("ncf_last_finetuned_at")
        if ts:
            logger.info(f"Watermark found — only fetching ratings after: {ts}")
            return ts
    logger.info("No NCF watermark found — this is the first fine-tune run.")
    return None


def save_watermark(timestamp: str):
    """Persists the new watermark. Merges with existing file to keep ALS watermark intact."""
    data = {}
    if WATERMARK_FILE.exists():
        data = json.loads(WATERMARK_FILE.read_text())
    data["ncf_last_finetuned_at"] = timestamp
    WATERMARK_FILE.write_text(json.dumps(data, indent=2))
    logger.info(f"NCF watermark updated → {timestamp}")


def synchronize_database_to_neural_network():
    print("\n===========================")
    print("[SUPABASE LIVE NCF FINE-TUNING]")
    print("===========================")
    start_time = time.time()

    # ── 1. Determine which ratings are NEW ───────────────────────────────────
    watermark_ts = load_watermark()

    query = supabase.table("ratings").select("user_id, movie_id, rating, created_at")
    if watermark_ts:
        # Supabase postgrest: gt = greater than
        query = query.gt("created_at", watermark_ts)

    response = query.order("created_at").execute()

    if not response.data:
        if watermark_ts:
            print(f"✅ No new ratings since {watermark_ts}. Model is already up-to-date.")
        else:
            print("No live ratings found in Supabase. Aborting.")
        return

    new_ratings_df = pd.DataFrame(response.data)

    # Record the timestamp of the most-recent rating we're about to train on
    run_timestamp = new_ratings_df["created_at"].max()

    new_ratings_df = new_ratings_df.rename(columns={"user_id": "userId", "movie_id": "movieId"})
    new_ratings_df["movieId"] = new_ratings_df["movieId"].astype(int)
    new_ratings_df["rating"]  = new_ratings_df["rating"].astype(float)

    print(f"🆕 {len(new_ratings_df)} NEW interactions to train on (since last run)")
    print(f"   Users: {new_ratings_df['userId'].nunique()} | Movies: {new_ratings_df['movieId'].nunique()}")

    # ── 2. Load → Fine-tune → Save ───────────────────────────────────────────
    print(f"\nInitiating surgical AI training for {len(new_ratings_df)} new targets...")

    production_ncf = NeuralCollaborativeRecommender()
    production_ncf.load()

    production_ncf.finetune(new_ratings_df, epochs=20, lr=0.003)
    production_ncf.save()

    # ── 3. Advance watermark ──────────────────────────────────────────────────
    save_watermark(run_timestamp)

    print("\n===========================")
    print(f"✅ NCF updated in {time.time() - start_time:.2f}s")
    print(f"   Next run will only train on ratings after: {run_timestamp}")
    print("===========================\n")


if __name__ == "__main__":
    synchronize_database_to_neural_network()

"""
ALS Production Retrain — with watermark-based deduplication.

Watermark system (scripts/.finetune_watermark.json):
  - Reads als_last_finetuned_at → only fetches Supabase ratings NEWER than that timestamp
  - After a successful run, advances the watermark to the newest rating's timestamp
  - Ratings already trained on are NEVER included again

Usage:
    python scripts/retrain_als_production.py
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

from app.database import supabase
from src.config import RATINGS_DIR
from src.models.collaborative_filtering import CollaborativeRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

WATERMARK_FILE = Path(__file__).parent / ".finetune_watermark.json"


def load_als_watermark() -> str | None:
    if WATERMARK_FILE.exists():
        data = json.loads(WATERMARK_FILE.read_text())
        ts = data.get("als_last_finetuned_at")
        if ts:
            logger.info(f"ALS watermark: only fetching ratings after {ts}")
            return ts
    logger.info("No ALS watermark — first-time run.")
    return None


def save_als_watermark(timestamp: str):
    data = {}
    if WATERMARK_FILE.exists():
        data = json.loads(WATERMARK_FILE.read_text())
    data["als_last_finetuned_at"] = timestamp
    WATERMARK_FILE.write_text(json.dumps(data, indent=2))
    logger.info(f"ALS watermark updated → {timestamp}")


def main():
    print("===========================================")
    print("🔄 ALS PRODUCTION RETRAINING SEQUENCE 🔄")
    print("===========================================")
    start_time = time.time()

    # ── 1. Fetch ONLY new ratings from Supabase ──────────────────────────────
    print("\n[1/3] Fetching NEW Live Ratings from Supabase...")
    watermark_ts = load_als_watermark()

    query = supabase.table("ratings").select("user_id, movie_id, rating, created_at")
    if watermark_ts:
        query = query.gt("created_at", watermark_ts)

    response = query.order("created_at").execute()

    if not response.data:
        if watermark_ts:
            print(f"✅ No new ratings since {watermark_ts}. ALS is already up-to-date.")
        else:
            print("No live ratings found in Supabase. Training on base dataset only.")
        new_ratings_df = pd.DataFrame()
        newest_ts = None
    else:
        new_ratings_df = pd.DataFrame(response.data)
        newest_ts = new_ratings_df["created_at"].max()
        new_ratings_df = new_ratings_df.rename(
            columns={"user_id": "userId", "movie_id": "movieId"}
        )
        new_ratings_df["movieId"] = new_ratings_df["movieId"].astype(int)
        print(f"🆕 {len(new_ratings_df)} new ratings | {new_ratings_df['userId'].nunique()} users | {new_ratings_df['movieId'].nunique()} movies")

    # ── 2. Load full MovieLens base dataset ───────────────────────────────────
    print(f"\n[2/3] Loading Base MovieLens Dataset from {RATINGS_DIR}...")
    existing_df = pd.read_csv(RATINGS_DIR)

    if not new_ratings_df.empty:
        # Clean out any previously injected UUID rows to avoid stacking duplicates
        uuid_mask = existing_df["userId"].astype(str).str.contains("-", na=False)
        if uuid_mask.any():
            print(f"  Removing {uuid_mask.sum()} previously injected live ratings (will re-add fresh)...")
            existing_df = existing_df[~uuid_mask]

        combined_df = pd.concat([existing_df, new_ratings_df[["userId", "movieId", "rating"]]], ignore_index=True)
        combined_df.to_csv(RATINGS_DIR, index=False)
        print(f"  Combined: {len(combined_df):,} total rows saved to {RATINGS_DIR}")
    else:
        combined_df = existing_df
        print(f"  Proceeding with {len(combined_df):,} base rows (no new live ratings).")

    # ── 3. Retrain ALS ────────────────────────────────────────────────────────
    print("\n[3/3] Training ALS on full combined dataset...")
    cf_model = CollaborativeRecommender(n_factors=64, n_iterations=20, regularization=0.05)
    cf_model.fit(ratings_path=RATINGS_DIR)

    print("\nSaving ALS weights...")
    cf_model.save()

    # ── 4. Advance watermark ──────────────────────────────────────────────────
    if newest_ts:
        save_als_watermark(newest_ts)
        print(f"  Next run will only train on ratings after: {newest_ts}")

    print(f"\n✅ ALS Production Retrain complete in {time.time()-start_time:.1f}s")
    print("   Restart the FastAPI server to load updated weights.")


if __name__ == "__main__":
    main()

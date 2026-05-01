"""
Fine-tunes both models on NEW Supabase ratings only.

Watermark system (scripts/.finetune_watermark.json):
  - Stores the created_at timestamp of the last rating each model was trained on
  - Each run ONLY fetches ratings newer than that timestamp
  - After a successful run, the watermark advances to the newest rating's timestamp

This guarantees ratings are NEVER trained on twice, even if you run
this script multiple times in a row.

Usage:
    python scripts/finetune_all.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` and `app` are importable
# regardless of which directory the script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import logging
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

WATERMARK_FILE = Path(__file__).parent / ".finetune_watermark.json"


# ──────────────────────────────────────────────────────────────
# WATERMARK HELPERS
# ──────────────────────────────────────────────────────────────

def _load_watermark(key: str) -> str | None:
    if WATERMARK_FILE.exists():
        data = json.loads(WATERMARK_FILE.read_text())
        ts = data.get(key)
        if ts:
            return ts
    return None


def _save_watermark(key: str, timestamp: str):
    data = {}
    if WATERMARK_FILE.exists():
        data = json.loads(WATERMARK_FILE.read_text())
    data[key] = timestamp
    WATERMARK_FILE.write_text(json.dumps(data, indent=2))
    logger.info(f"Watermark [{key}] → {timestamp}")


# ──────────────────────────────────────────────────────────────
# 0.  FETCH NEW RATINGS FROM SUPABASE
# ──────────────────────────────────────────────────────────────

def fetch_new_ratings(watermark_key: str) -> tuple[pd.DataFrame, str | None]:
    """
    Returns (new_ratings_df, newest_timestamp).
    new_ratings_df is empty if nothing new since the watermark.
    """
    from app.database import supabase

    watermark_ts = _load_watermark(watermark_key)
    if watermark_ts:
        logger.info(f"[{watermark_key}] Only fetching ratings after: {watermark_ts}")
    else:
        logger.info(f"[{watermark_key}] No watermark — first-time run.")

    query = supabase.table("ratings").select("user_id, movie_id, rating, created_at")
    if watermark_ts:
        query = query.gt("created_at", watermark_ts)

    resp = query.order("created_at").execute()

    if not resp.data:
        return pd.DataFrame(), None

    df = pd.DataFrame(resp.data)
    newest_ts = df["created_at"].max()

    df = df.rename(columns={"user_id": "userId", "movie_id": "movieId"})
    df["movieId"] = df["movieId"].astype(int)
    df["rating"]  = df["rating"].astype(float)

    logger.info(
        f"[{watermark_key}] {len(df)} new ratings | "
        f"{df['userId'].nunique()} users | {df['movieId'].nunique()} movies"
    )
    return df, newest_ts


# ──────────────────────────────────────────────────────────────
# 1.  NCF FINE-TUNE
# ──────────────────────────────────────────────────────────────

def finetune_ncf():
    from src.models.ncf import NeuralCollaborativeRecommender

    print()
    logger.info("=" * 60)
    logger.info("  PHASE 1 — NCF (NeuMF) FINE-TUNE")
    logger.info("=" * 60)
    t0 = time.time()

    live_df, newest_ts = fetch_new_ratings("ncf_last_finetuned_at")

    if live_df.empty:
        logger.info("  ✅ NCF is already up-to-date. No new ratings to train on.")
        return

    ncf = NeuralCollaborativeRecommender()
    ncf.load()
    ncf.finetune(live_df, epochs=20, lr=0.003)
    ncf.save()

    _save_watermark("ncf_last_finetuned_at", newest_ts)
    logger.info(f"  ✅ NCF fine-tuned on {len(live_df)} new ratings in {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────────────────────
# 2.  ALS RETRAIN (base + new live data merged)
# ──────────────────────────────────────────────────────────────

def finetune_als():
    from implicit.als import AlternatingLeastSquares
    from src.config import PROCESSED_DATA_DIR, MOVIES_DF_PKL_PATH
    from src.models.collaborative_filtering import CollaborativeRecommender
    from app.database import supabase as _supabase

    print()
    logger.info("=" * 60)
    logger.info("  PHASE 2 — ALS COLLABORATIVE FILTERING RETRAIN")
    logger.info("=" * 60)
    t0 = time.time()

    # ── STEP 1: Fetch ALL Supabase ratings (always — no watermark for ALS) ────
    # ALS is a full retrain every time, so we always use the complete live dataset.
    # No CSV mutation — base MovieLens data stays untouched.
    logger.info("  Fetching ALL live ratings from Supabase...")
    resp = _supabase.table("ratings").select("user_id, movie_id, rating").execute()

    if not resp.data:
        logger.info("  No live ratings in Supabase. Training on base MovieLens data only.")
        live_df = pd.DataFrame(columns=["userId", "movieId", "rating"])
    else:
        live_df = pd.DataFrame(resp.data)
        live_df = live_df.rename(columns={"user_id": "userId", "movie_id": "movieId"})
        live_df["userId"]  = live_df["userId"].astype(str)
        live_df["movieId"] = live_df["movieId"].astype("int32")
        live_df["rating"]  = live_df["rating"].astype("float32")
        logger.info(f"  Live ratings: {len(live_df):,} | {live_df['userId'].nunique()} users | {live_df['movieId'].nunique()} movies")

    # ── STEP 2: Load base CSV and merge in-memory (CSV is never modified) ─────
    base_path = PROCESSED_DATA_DIR / "train_ratings.csv"
    logger.info(f"  Loading base MovieLens split from {base_path}...")
    base_df = pd.read_csv(
        base_path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": str, "movieId": "int32", "rating": "float32"},
    )
    logger.info(f"  Base: {len(base_df):,} rows")

    if not live_df.empty:
        combined = (
            pd.concat([base_df, live_df], ignore_index=True)
            .drop_duplicates(subset=["userId", "movieId"], keep="last")
            .reset_index(drop=True)
        )
    else:
        combined = base_df

    logger.info(f"  Combined (in-memory only): {len(combined):,} rows")

    # ── STEP 3: Build sparse matrix ───────────────────────────────────────────
    unique_users = combined["userId"].unique()
    unique_items = combined["movieId"].unique()
    user2idx = {str(u): i for i, u in enumerate(unique_users)}
    idx2user = {i: str(u) for i, u in enumerate(unique_users)}
    item2idx = {int(i): idx for idx, i in enumerate(unique_items)}
    idx2item = {idx: int(i) for idx, i in enumerate(unique_items)}

    n_users = len(unique_users)
    n_items = len(unique_items)
    logger.info(f"  Matrix: {n_users:,} users × {n_items:,} items")

    u_idx = combined["userId"].astype(str).map(user2idx).values
    i_idx = combined["movieId"].astype(int).map(item2idx).values
    conf  = combined["rating"].values.astype(np.float32)

    user_item_csr = sparse.csr_matrix(
        (conf, (u_idx, i_idx)), shape=(n_users, n_items), dtype=np.float32
    )
    user_seen_items = {
        u: user_item_csr.getrow(u).indices for u in range(n_users)
    }

    # ── STEP 4: Train ALS ─────────────────────────────────────────────────────
    als = AlternatingLeastSquares(
        factors=64,
        iterations=20,
        regularization=0.05,
        num_threads=0,
        calculate_training_loss=True,
        random_state=42,
    )
    logger.info("  Training ALS on base + all live ratings (in-memory merge)...")
    als.fit(user_item_csr.T.tocsr())

    # ── STEP 5: Save model ────────────────────────────────────────────────────
    cf = CollaborativeRecommender()
    cf.model           = als
    cf.user2idx        = user2idx
    cf.idx2user        = idx2user
    cf.item2idx        = item2idx
    cf.idx2item        = idx2item
    cf.user_seen_items = user_seen_items
    cf.movies_df       = joblib.load(MOVIES_DF_PKL_PATH)
    cf.save()

    logger.info(f"  ✅ ALS retrained in {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    total_start = time.time()
    print()
    print("=" * 60)
    print("  NEURALFLICK — MODEL FINE-TUNE PIPELINE")

    # Show NCF watermark (ALS has no watermark — always retrains on full data)
    if WATERMARK_FILE.exists():
        wm = json.loads(WATERMARK_FILE.read_text())
        print(f"  NCF watermark : {wm.get('ncf_last_finetuned_at', 'none (first run)')}")
    else:
        print("  NCF watermark : none (first run)")
    print("  ALS           : always retrains on full base + all Supabase data")
    print("=" * 60)

    finetune_ncf()
    finetune_als()

    elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print(f"  🚀 Fine-tune complete in {elapsed:.1f}s")

    # Auto-trigger hot-reload on the running FastAPI server so new weights
    # go live immediately — no manual server restart needed.
    try:
        import urllib.request as _req
        _req.urlopen(
            _req.Request("http://localhost:8000/admin/reload-model", method="POST"),
            timeout=10
        )
        print("  ✅ Model hot-reloaded on running server — new weights are live!")
    except Exception as e:
        print(f"  ⚠️  Could not auto-reload server ({e}).")
        print("     Restart uvicorn manually to load updated weights.")

    print("=" * 60)
    print()



if __name__ == "__main__":
    main()

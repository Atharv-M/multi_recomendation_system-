"""
Train / Validation Split Script
================================
Splits the filtered ratings into:
  - Train set (all but last rating per user) → used to train SVD + NCF
  - Val set   (last 1 rating per user)       → used to measure HR@10, NDCG@10

Strategy: Leave-One-Out (industry standard for recommender evaluation)
  - Sort each user's ratings by timestamp
  - Hold out the LAST (most recent) rated movie per user as the test item
  - Everything else goes to train

Outputs:
  data/processed/train_ratings.csv
  data/processed/val_ratings.csv

Run from project root:
    python scripts/split_train_val.py
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run():
    logger.info("=" * 60)
    logger.info("Train / Validation Split (Leave-One-Out)")
    logger.info("=" * 60)

    # ── Load filtered ratings ──────────────────────────────────────
    logger.info("\nLoading rating.csv...")
    ratings = pd.read_csv(RAW_DIR / "rating.csv")
    logger.info(f"  Total ratings loaded: {len(ratings):,}")
    logger.info(f"  Unique users:         {ratings['userId'].nunique():,}")
    logger.info(f"  Unique movies:        {ratings['movieId'].nunique():,}")

    # ── Sort by timestamp so "last" rating is truly the most recent ─
    logger.info("\nSorting by userId + timestamp...")
    ratings = ratings.sort_values(["userId", "timestamp"])

    # ── Leave-One-Out split ────────────────────────────────────────
    logger.info("Splitting: last 1 rating per user → val set...")
    val_df   = ratings.groupby("userId").tail(1)
    train_df = ratings.drop(val_df.index)

    logger.info(f"\n  Train ratings: {len(train_df):,}  ({len(train_df)/len(ratings)*100:.1f}%)")
    logger.info(f"  Val ratings:   {len(val_df):,}  ({len(val_df)/len(ratings)*100:.1f}%)")
    logger.info(f"  Users in train: {train_df['userId'].nunique():,}")
    logger.info(f"  Users in val:   {val_df['userId'].nunique():,}")

    # ── Sanity check: every val user should also exist in train ────
    val_only_users = set(val_df["userId"]) - set(train_df["userId"])
    if val_only_users:
        logger.warning(f"  ⚠️  {len(val_only_users):,} users exist ONLY in val (single-rating users). "
                       f"These will be excluded from evaluation.")

    # ── Save ───────────────────────────────────────────────────────
    logger.info("\nSaving splits...")
    train_df.to_csv(PROCESSED_DIR / "train_ratings.csv", index=False)
    logger.info(f"  ✅ train_ratings.csv saved  ({len(train_df):,} rows)")

    val_df.to_csv(PROCESSED_DIR / "val_ratings.csv", index=False)
    logger.info(f"  ✅ val_ratings.csv   saved  ({len(val_df):,} rows)")

    # ── Summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SPLIT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train: {len(train_df):,} ratings  → used for SVD + NCF training")
    logger.info(f"Val:   {len(val_df):,} ratings    → used for HR@10 / NDCG@10 evaluation")
    logger.info("\nNext step:")
    logger.info("  python src/data/build_dataset.py")


if __name__ == "__main__":
    run()

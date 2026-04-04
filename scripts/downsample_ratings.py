"""
downsample_ratings.py
─────────────────────
Downsamples train_ratings.csv to ~20M ratings by capping the maximum
number of ratings any single user can have.

Logic:
  - Users with fewer ratings than the cap keep ALL their ratings.
  - Users with more ratings than the cap keep only their NEWEST ratings.
  - This preserves data for casual users while truncating outdated, 
    decades-old history from power users.
  - Target: ~20M ratings -> requires a cap of roughly 225 ratings/user.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

def main():
    MAX_RATINGS_PER_USER = 225
    train_path = PROCESSED_DIR / "train_ratings.csv"
    out_path = PROCESSED_DIR / "train_ratings_20m.csv"

    logger.info(f"Loading {train_path}...")
    
    # We load timestamp if available to sort by it, otherwise we assume the file 
    # is already chronologically sorted from when we did the LOO split.
    try:
        df = pd.read_csv(train_path)
    except FileNotFoundError:
        logger.error(f"Could not find {train_path}! Make sure the path is correct.")
        return

    original_size = len(df)
    logger.info(f"Loaded {original_size:,} ratings.")
    
    # Check if we have a timestamp to ensure chronological sorting
    if 'timestamp' in df.columns:
        logger.info("Sorting by userId and timestamp...")
        df = df.sort_values(['userId', 'timestamp'])
    else:
        logger.info("No timestamp column found. Assuming data is already chronologically ordered.")
    
    logger.info(f"Capping user history to newest {MAX_RATINGS_PER_USER} ratings per user...")
    
    # groupby userId, take the tail (newest) MAX_RATINGS_PER_USER
    df_downsampled = df.groupby('userId').tail(MAX_RATINGS_PER_USER).reset_index(drop=True)
    
    new_size = len(df_downsampled)
    
    logger.info("\n" + "="*50)
    logger.info("DOWNSAMPLING SUMMARY")
    logger.info("="*50)
    logger.info(f"Original ratings   : {original_size:>12,}")
    logger.info(f"New ratings        : {new_size:>12,}")
    logger.info(f"Removed ratings    : {original_size - new_size:>12,} (from power users only)")
    logger.info(f"Unique users       : {df_downsampled['userId'].nunique():>12,} (No users lost!)")
    logger.info(f"Unique movies      : {df_downsampled['movieId'].nunique():>12,} (Catalog highly preserved!)")
    target_ncf_rows = new_size * 3  # Based on 1 pos + 2 neg
    logger.info(f"\nExpected NCF dataset size with 1 pos + 2 neg = {target_ncf_rows:,} rows")
    logger.info("="*50)
    
    logger.info(f"Saving to {out_path}...")
    df_downsampled.to_csv(out_path, index=False)
    logger.info(f"✅ Saved successfully. Size: {out_path.stat().st_size/1e6:.1f} MB")
    
if __name__ == "__main__":
    main()

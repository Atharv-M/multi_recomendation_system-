"""
Dataset Filter Script — ml-32m → Post-2000 Only
================================================
Filters all 4 CSV files so that only movies from 2000 onwards remain.
The cascade order matters:
  1. movies.csv  → extract valid movieIds (year >= 2000)
  2. ratings.csv → keep only rows where movieId is in valid set
  3. tags.csv    → keep only rows where movieId is in valid set
  4. links.csv   → keep only rows where movieId is in valid set

Then saves filtered files using the naming convention expected by config.py:
  movies.csv  → movie.csv
  ratings.csv → rating.csv
  tags.csv    → tag.csv
  links.csv   → link.csv

Run from project root:
    python scripts/prepare_filtered_dataset.py
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
CUTOFF_YEAR = 2000


def run():
    logger.info("=" * 60)
    logger.info(f"MovieLens 32M → Post-{CUTOFF_YEAR} Filter Script")
    logger.info("=" * 60)

    # ── STEP 1: Filter movies.csv ──────────────────────────────────
    logger.info("\n[1/4] Loading movies.csv...")
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    total_movies = len(movies)

    # Extract year from title like "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

    # Keep only 2000+
    movies_filtered = movies[movies["year"] >= CUTOFF_YEAR].copy()
    movies_filtered = movies_filtered.drop(columns=["year"])  # remove helper column

    valid_ids = set(movies_filtered["movieId"].tolist())

    logger.info(f"  Total movies:         {total_movies:>8,}")
    logger.info(f"  Post-{CUTOFF_YEAR} movies:    {len(movies_filtered):>8,}  ({len(movies_filtered)/total_movies*100:.1f}%)")
    logger.info(f"  Removed movies:       {total_movies - len(movies_filtered):>8,}")

    # ── STEP 2: Filter ratings.csv ─────────────────────────────────
    logger.info("\n[2/4] Loading ratings.csv (836MB — takes ~60 seconds)...")
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    total_ratings = len(ratings)

    ratings_filtered = ratings[ratings["movieId"].isin(valid_ids)].copy()

    logger.info(f"  Total ratings:        {total_ratings:>12,}")
    logger.info(f"  Post-{CUTOFF_YEAR} ratings: {len(ratings_filtered):>12,}  ({len(ratings_filtered)/total_ratings*100:.1f}%)")
    logger.info(f"  Removed ratings:      {total_ratings - len(ratings_filtered):>12,}")

    del ratings  # free RAM immediately

    # ── STEP 3: Filter tags.csv ────────────────────────────────────
    logger.info("\n[3/4] Loading tags.csv (69MB)...")
    tags = pd.read_csv(RAW_DIR / "tags.csv")
    total_tags = len(tags)

    tags_filtered = tags[tags["movieId"].isin(valid_ids)].copy()

    logger.info(f"  Total tags:           {total_tags:>8,}")
    logger.info(f"  Post-{CUTOFF_YEAR} tags:     {len(tags_filtered):>8,}  ({len(tags_filtered)/total_tags*100:.1f}%)")

    del tags

    # ── STEP 4: Filter links.csv ───────────────────────────────────
    logger.info("\n[4/4] Loading links.csv (1.9MB)...")
    links = pd.read_csv(RAW_DIR / "links.csv")
    total_links = len(links)

    links_filtered = links[links["movieId"].isin(valid_ids)].copy()

    logger.info(f"  Total links:          {total_links:>8,}")
    logger.info(f"  Post-{CUTOFF_YEAR} links:    {len(links_filtered):>8,}  ({len(links_filtered)/total_links*100:.1f}%)")

    del links

    # ── SAVE: Write filtered files with config.py naming ──────────
    logger.info("\n[Saving] Writing filtered files...")

    movies_filtered.to_csv(RAW_DIR / "movie.csv", index=False)
    logger.info(f"  ✅ movie.csv  saved   ({len(movies_filtered):,} rows)")

    ratings_filtered.to_csv(RAW_DIR / "rating.csv", index=False)
    logger.info(f"  ✅ rating.csv saved   ({len(ratings_filtered):,} rows)")

    tags_filtered.to_csv(RAW_DIR / "tag.csv", index=False)
    logger.info(f"  ✅ tag.csv    saved   ({len(tags_filtered):,} rows)")

    links_filtered.to_csv(RAW_DIR / "link.csv", index=False)
    logger.info(f"  ✅ link.csv   saved   ({len(links_filtered):,} rows)")

    # ── SUMMARY ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Movies:  {total_movies:,}  →  {len(movies_filtered):,}  (kept {CUTOFF_YEAR}+)")
    logger.info(f"Ratings: {total_ratings:,}  →  {len(ratings_filtered):,}")
    logger.info(f"Tags:    {total_tags:,}  →  {len(tags_filtered):,}")
    logger.info(f"Links:   {total_links:,}  →  {len(links_filtered):,}")
    logger.info("\nNext step:")
    logger.info("  python scripts/split_train_val.py")


if __name__ == "__main__":
    run()

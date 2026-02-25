import os
import pandas as pd
from app.database import supabase
from src.config import RATINGS_DIR
from src.models.collaborative_filtering import CollaborativeRecommender

print("Fetching ratings from Supabase...")
response = supabase.table("ratings").select("user_id, movie_id, rating").execute()

if not response.data:
    print("No ratings found in Supabase.")
    exit(0)

# Convert Supabase data to DataFrame
new_ratings_df = pd.DataFrame(response.data)
# Rename columns to match existing rating.csv format (userId, movieId, rating)
new_ratings_df = new_ratings_df.rename(columns={
    "user_id": "userId",
    "movie_id": "movieId",
    "rating": "rating"
})

print(f"Fetched {len(new_ratings_df)} new ratings from Supabase.")

# Load existing ratings
print(f"Loading existing ratings from {RATINGS_DIR}...")
existing_ratings_df = pd.read_csv(RATINGS_DIR)

# Ensure the new data isn't already appended (basic deduplication by user_id string length check since old ones are ints)
# Old userIds are ints, new userIds are UUIDs (strings with hyphens)
existing_uuids = existing_ratings_df[existing_ratings_df['userId'].astype(str).str.contains('-', na=False)]
if not existing_uuids.empty:
    print(f"Found {len(existing_uuids)} UUID-based ratings already in the dataset. Removing them to prevent duplicates during merge.")
    existing_ratings_df = existing_ratings_df[~existing_ratings_df['userId'].astype(str).str.contains('-', na=False)]

# Append new ratings
combined_ratings_df = pd.concat([existing_ratings_df, new_ratings_df], ignore_index=True)

# Save back to rating.csv
print(f"Saving combined dataset with {len(combined_ratings_df)} total ratings...")
combined_ratings_df.to_csv(RATINGS_DIR, index=False)
print("Saved successfully.")

# Retrain SVD Model
print("Initializing SVD Collaborative Filtering Retraining...")
cf_model = CollaborativeRecommender()
cf_model.fit()
cf_model.save()
print("SVD Model Retraining Complete! The model now officially supports UUIDs.")

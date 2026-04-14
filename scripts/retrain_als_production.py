import os
import pandas as pd
from app.database import supabase
from src.config import RATINGS_DIR
from src.models.collaborative_filtering import CollaborativeRecommender

def main():
    print("===========================================")
    print("🔄 ALS PRODUCTION RETRAINING SEQUENCE 🔄")
    print("===========================================")
    
    print("\n[1/3] Fetching Live Ratings from Supabase...")
    response = supabase.table("ratings").select("user_id, movie_id, rating").execute()

    new_ratings_df = pd.DataFrame()
    if response.data:
        new_ratings_df = pd.DataFrame(response.data)
        new_ratings_df = new_ratings_df.rename(columns={
            "user_id": "userId",
            "movie_id": "movieId",
            "rating": "rating"
        })
        print(f"Fetched {len(new_ratings_df)} live ratings from Supabase.")
    else:
        print("No live ratings found in Supabase.")

    print(f"\n[2/3] Loading Full MovieLens 32M Raw Dataset from {RATINGS_DIR}...")
    existing_ratings_df = pd.read_csv(RATINGS_DIR)

    if not new_ratings_df.empty:
        # Supabase IDs are UUID strings, old MovieLens IDs are integers.
        # Remove any previously appended UUIDs so we don't accidentally duplicate
        existing_uuids = existing_ratings_df[existing_ratings_df['userId'].astype(str).str.contains('-', na=False)]
        if not existing_uuids.empty:
            print(f"Found {len(existing_uuids)} previously injected UUID ratings. Cleaning them out for fresh merge...")
            existing_ratings_df = existing_ratings_df[~existing_ratings_df['userId'].astype(str).str.contains('-', na=False)]

        # Combine them!
        combined_ratings_df = pd.concat([existing_ratings_df, new_ratings_df], ignore_index=True)
        
        # Save back to rating.csv so Kaggle also gets them if uploaded
        print(f"Injecting into Master Data and saving {len(combined_ratings_df)} total ratings...")
        combined_ratings_df.to_csv(RATINGS_DIR, index=False)
        print("Injection saved successfully.")
    else:
        combined_ratings_df = existing_ratings_df
        print(f"Proceeding with {len(combined_ratings_df)} total ratings.")

    print("\n[3/3] Initializing Massive Implicit ALS Hardware Training...")
    # Initialize the implicit ALS recommender
    cf_model = CollaborativeRecommender(n_factors=50, n_iterations=15, regularization=0.05)
    
    # Train using the FULL data (which now includes Supabase) via the path
    cf_model.fit(ratings_path=RATINGS_DIR)
    
    print("\nSaving Production ALS weights...")
    cf_model.save()
    print("✅ ALS Model Retraining Complete! The model is fully production-ready with all live users injected.")

if __name__ == "__main__":
    main()

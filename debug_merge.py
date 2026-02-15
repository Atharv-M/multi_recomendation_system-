
import pandas as pd
from app.config import LINKS_DATASET_PATH, MASTER_DATASET_PATH

def debug_merge():
    print(f"Loading Links from: {LINKS_DATASET_PATH}")
    links_df = pd.read_csv(LINKS_DATASET_PATH)
    print("Links Columns:", links_df.columns.tolist())
    print("Links Head:\n", links_df.head())
    
    print(f"\nLoading Master from: {MASTER_DATASET_PATH}")
    master_df = pd.read_csv(MASTER_DATASET_PATH)
    
    # Simulate a recommendation df (top 5 movies)
    recs_df = master_df.head(5)[["movieId", "title", "genres"]].copy()
    print("\nRecs DF Columns before merge:", recs_df.columns.tolist())
    
    print("\nAttempting Merge...")
    try:
        merged_df = recs_df.merge(
            links_df[["movieId", "tmdbId"]],
            on="movieId",
            how="left"
        )
        print("Merged DF Columns:", merged_df.columns.tolist())
        print("Merged Head:\n", merged_df.head())
        
        # Verify access
        print("\nVerifying row access...")
        for _, row in merged_df.iterrows():
            _ = row["tmdbId"]
        print("Access successful!")
        
    except Exception as e:
        print(f"\nMERGE FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_merge()

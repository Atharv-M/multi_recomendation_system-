import pandas as pd
from app.database import supabase
from src.models.ncf import NeuralCollaborativeRecommender
import time

def synchronize_database_to_neural_network():
    print("\n===========================")
    print("[SUPABASE REAL-TIME DATABASE EXTRACTION]")
    print("===========================")
    
    start_time = time.time()
    response = supabase.table("ratings").select("user_id, movie_id, rating").execute()
    
    if not response.data:
        print("No onboarding ratings found inside Supabase pgSQL Database! (Table empty)")
        return
        
    # Translate SQL Rows natively into PyTorch Deep Learning requirements
    new_ratings_df = pd.DataFrame(response.data)
    new_ratings_df = new_ratings_df.rename(columns={"user_id": "userId", "movie_id": "movieId"})
    
    # Supabase gives UUID strings. Our model expects strings.
    # Convert movie_ids to int to match logic!
    new_ratings_df['movieId'] = new_ratings_df['movieId'].astype(int)
    
    print(f"Extracted {len(new_ratings_df)} Live Interactions from the Web UI...")
    
    production_ncf = NeuralCollaborativeRecommender()
    production_ncf.load()
    
    # Turn on the AI Math!
    production_ncf.finetune(new_ratings_df, epochs=25, lr=0.005)
    
    # Lock the modifications down!
    production_ncf.save()
    
    print("\n===========================")
    print(f"✅ Deep Learning Architecture successfully updated and resaved in: {time.time() - start_time:.2f} seconds!")
    print("===========================\n")

if __name__ == "__main__":
    synchronize_database_to_neural_network()

import pandas as pd
from app.database import supabase
from src.models.ncf import NeuralCollaborativeRecommender
import time

def synchronize_database_to_neural_network():
    print("\n===========================")
    print("[SUPABASE LIVE NCF FINE-TUNING]")
    print("===========================")
    
    start_time = time.time()
    
    # 1. Fetch Supabase Live User Data
    response = supabase.table("ratings").select("user_id, movie_id, rating").execute()
    
    if not response.data:
        print("No live ratings found in Supabase. Aborting.")
        return
        
    new_ratings_df = pd.DataFrame(response.data)
    new_ratings_df = new_ratings_df.rename(columns={"user_id": "userId", "movie_id": "movieId"})
    new_ratings_df['movieId'] = new_ratings_df['movieId'].astype(int)
    print(f"Extracted {len(new_ratings_df)} Live Interactions from the Web UI...")
        
    print(f"Initiating surgical AI training for {len(new_ratings_df)} Live Targets...")
    
    production_ncf = NeuralCollaborativeRecommender()
    production_ncf.load()
    
    # Run the backpropagation exclusively on live new interactions 
    # to maintain lightning-fast API responses
    production_ncf.finetune(new_ratings_df, epochs=25, lr=0.005)
    
    # Lock the modifications down
    production_ncf.save()
    
    print("\n===========================")
    print(f"✅ Deep Learning Architecture successfully updated and resaved in: {time.time() - start_time:.2f} seconds!")
    print("===========================\n")

if __name__ == "__main__":
    synchronize_database_to_neural_network()

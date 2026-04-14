import os
import time
import math
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch

from src.config import PROCESSED_DATA_DIR
from src.models.collaborative_filtering import CollaborativeRecommender
from src.models.ncf import NeuralCollaborativeRecommender

def get_hit_ratio(rank_list, target_item):
    return 1 if target_item in rank_list else 0

def get_ndcg(rank_list, target_item):
    for i in range(len(rank_list)):
        if rank_list[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def main():
    print("===========================================")
    print("🏆 MASSIVE 32M DATASET MODEL EVALUATION 🏆")
    print("===========================================\n")

    print("[1/3] Loading Data...")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val_ratings.csv")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_ratings_20m.csv")
    
    # We only evaluate a sample of users for speed (e.g. 2000 users)
    NUM_EVAL_USERS = 2000
    eval_users = val_df['userId'].unique()
    np.random.seed(42)
    sampled_users = np.random.choice(eval_users, size=min(NUM_EVAL_USERS, len(eval_users)), replace=False)
    
    val_sample = val_df[val_df['userId'].isin(sampled_users)]
    
    all_items = pd.concat([train_df['movieId'], val_df['movieId']]).unique()
    
    # Build dictionary of all items interacted by each user in the training set
    print("Building interactions lookup...")
    user_interacted = train_df[train_df['userId'].isin(sampled_users)].groupby('userId')['movieId'].apply(set).to_dict()

    print("\n[2/3] Loading Models...")
    # Load ALS model
    als = CollaborativeRecommender()
    als.load()
    
    # Load NCF model
    ncf = NeuralCollaborativeRecommender()
    ncf.load()
    ncf.model.eval()
    device = ncf.device

    print(f"\n[3/3] Evaluating {len(sampled_users)} users (1 target + 99 negatives each)...")
    als_hits, als_ndcgs = [], []
    ncf_hits, ncf_ndcgs = [], []

    for _, row in tqdm(val_sample.iterrows(), total=len(val_sample)):
        u_str = str(int(row['userId']))
        target_item = int(row['movieId'])
        
        interacted = user_interacted.get(int(row['userId']), set())
        interacted.add(target_item)
        
        # 1 Target + 99 Negatives
        negative_items = []
        while len(negative_items) < 99:
            neg_candidate = random.choice(all_items)
            if neg_candidate not in interacted:
                negative_items.append(neg_candidate)
                
        eval_items = [target_item] + negative_items
        
        # ──────────────────────────────────────────────────────────
        # EVALUATE ALS
        # ──────────────────────────────────────────────────────────
        als_preds = {}
        if u_str in als.user2idx:
            u_idx_als = als.user2idx[u_str]
            user_vector_als = als.model.item_factors[u_idx_als]
            for item in eval_items:
                if int(item) in als.item2idx:
                    i_idx_als = als.item2idx[int(item)]
                    # dot product
                    als_preds[item] = np.dot(user_vector_als, als.model.user_factors[i_idx_als])
                else:
                    als_preds[item] = -np.inf
                    
            als_ranked = sorted(als_preds.items(), key=lambda x: x[1], reverse=True)[:10]
            als_rank_list = [i for i, _ in als_ranked]
            als_hits.append(get_hit_ratio(als_rank_list, target_item))
            als_ndcgs.append(get_ndcg(als_rank_list, target_item))
            
        # ──────────────────────────────────────────────────────────
        # EVALUATE NCF
        # ──────────────────────────────────────────────────────────
        if u_str in ncf.user2idx:
            u_idx_ncf = ncf.user2idx[u_str]
            
            # Map items
            valid_eval_items = [m for m in eval_items if m in ncf.item2idx]
            if target_item not in valid_eval_items:
                continue # Skip if target item was culled or never trained
                
            items_df = pd.DataFrame({
                'movieId': valid_eval_items, 
                'item_idx': [ncf.item2idx[m] for m in valid_eval_items]
            })
            
            items_df = items_df.merge(ncf.movies_df[['movieId', 'genres']], on='movieId', how='left')
            items_df['genres'] = items_df['genres'].fillna("Unknown").str.split('|')
            genres_encoded = ncf.mlb.transform(items_df['genres'])
            
            default_pad = [0] * ncf.max_tags_per_movie
            tags_array = items_df['movieId'].map(lambda m: ncf.movie_tags.get(m, default_pad)).tolist()
            tags_np = np.array(tags_array)
            
            user_tensor = torch.tensor([u_idx_ncf] * len(valid_eval_items), dtype=torch.long).to(device)
            item_tensor = torch.tensor(items_df['item_idx'].values, dtype=torch.long).to(device)
            genre_tensor = torch.tensor(genres_encoded, dtype=torch.float32).to(device)
            tag_tensor = torch.tensor(tags_np, dtype=torch.long).to(device)
            
            with torch.no_grad():
                ncf_probs = ncf.model(user_tensor, item_tensor, genre_tensor, tag_tensor).cpu().numpy()
                
            ncf_preds = {valid_eval_items[i]: ncf_probs[i] for i in range(len(valid_eval_items))}
            ncf_ranked = sorted(ncf_preds.items(), key=lambda x: x[1], reverse=True)[:10]
            ncf_rank_list = [i for i, _ in ncf_ranked]
            ncf_hits.append(get_hit_ratio(ncf_rank_list, target_item))
            ncf_ndcgs.append(get_ndcg(ncf_rank_list, target_item))

    print("\n===========================================")
    print("🏆 FINAL PERFORMANCE COMPARISON 🏆")
    print("===========================================")
    print(f"Metric  | ALS (Implicit) | Ultimate NeuMF (w/ Tags & Genres) ")
    print(f"HR@10   | {np.mean(als_hits):.4f}         | {np.mean(ncf_hits):.4f}")
    print(f"NDCG@10 | {np.mean(als_ndcgs):.4f}         | {np.mean(ncf_ndcgs):.4f}")
    print("===========================================")
    
    if np.mean(ncf_hits) > np.mean(als_hits):
        print("✅ The Ultimate Deep Learning architecture devastated ALS!")
    else:
        print("✅ ALS is highly competitive and extremely robust for its size!")

if __name__ == "__main__":
    main()

import os
import time
import math
import numpy as np
import pandas as pd
import random
from surprise import Dataset, Reader, SVD
from tqdm import tqdm
import joblib

import torch

from src.config import RATINGS_DIR, CF_MODEL_PATH
from src.models.ncf import NeuralCollaborativeRecommender

def get_hit_ratio(rank_list, target_item):
    if target_item in rank_list:
        return 1
    return 0

def get_ndcg(rank_list, target_item):
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def evaluate_leave_one_out_hybrid(svd_model, ncf_recommender, test_data, top_k=10):
    svd_hits, svd_ndcgs = [], []
    neumf_hits, neumf_ndcgs = [], []
    
    device = ncf_recommender.device
    ncf_recommender.model.eval()
    
    user_count = 0
    max_eval_users = 10000 
    
    print("Evaluating Ranking Performance (HR@10, NDCG@10)...")
    for user_str, (target_item, negative_items) in tqdm(test_data.items(), total=min(len(test_data), max_eval_users)):
        if user_str not in ncf_recommender.user2idx:
            continue
            
        eval_items = [target_item] + negative_items
        
        # --- Evaluate SVD ---
        svd_preds = {}
        for item in eval_items:
            svd_preds[item] = svd_model.predict(user_str, item).est
            
        svd_ranked = sorted(svd_preds.items(), key=lambda x: x[1], reverse=True)[:top_k]
        svd_rank_list = [i for i, _ in svd_ranked]
        
        svd_hits.append(get_hit_ratio(svd_rank_list, target_item))
        svd_ndcgs.append(get_ndcg(svd_rank_list, target_item))
        
        # --- Evaluate NeuMF ---
        u_idx = ncf_recommender.user2idx[user_str]
        
        items_df = pd.DataFrame({'movieId': eval_items, 'item_idx': [ncf_recommender.item2idx[m] for m in eval_items]})
        items_df = items_df.merge(ncf_recommender.movies_df[['movieId', 'genres']], on='movieId', how='left')
        items_df['genres'] = items_df['genres'].fillna("Unknown").str.split('|')
        
        genres_encoded = ncf_recommender.mlb.transform(items_df['genres'])
        timestamps_scaled = np.ones(len(items_df))
        
        default_pad = [0] * ncf_recommender.max_tags_per_movie
        tags_array = items_df['movieId'].map(lambda m: ncf_recommender.movie_tags.get(m, default_pad)).tolist()
        tags_np = np.array(tags_array)
        
        user_tensor = torch.tensor([u_idx] * len(eval_items), dtype=torch.long).to(device)
        item_tensor = torch.tensor(items_df['item_idx'].values, dtype=torch.long).to(device)
        genre_tensor = torch.tensor(genres_encoded, dtype=torch.float32).to(device)
        time_tensor = torch.tensor(timestamps_scaled, dtype=torch.float32).to(device)
        tag_tensor = torch.tensor(tags_np, dtype=torch.long).to(device)
        
        with torch.no_grad():
            neumf_probs = ncf_recommender.model(user_tensor, item_tensor, genre_tensor, time_tensor, tag_tensor).cpu().numpy()
            
        neumf_preds = {eval_items[i]: neumf_probs[i] for i in range(len(eval_items))}
        neumf_ranked = sorted(neumf_preds.items(), key=lambda x: x[1], reverse=True)[:top_k]
        neumf_rank_list = [i for i, _ in neumf_ranked]
        
        neumf_hits.append(get_hit_ratio(neumf_rank_list, target_item))
        neumf_ndcgs.append(get_ndcg(neumf_rank_list, target_item))
        
        user_count += 1
        if user_count >= max_eval_users:
            break
            
    return np.mean(svd_hits), np.mean(svd_ndcgs), np.mean(neumf_hits), np.mean(neumf_ndcgs)

def main():
    print("Loading Dataset for Massive Hybrid Test...")
    ratings = pd.read_csv(RATINGS_DIR)
    
    # We must sample the EXACT same 15,000 users we used for the NCF Training!
    SAMPLE_USERS = 15000 
    unique_users = ratings['userId'].unique()
    if len(unique_users) > SAMPLE_USERS:
        np.random.seed(42) # ENSURE identical test split
        sampled_users = np.random.choice(unique_users, size=SAMPLE_USERS, replace=False)
        ratings = ratings[ratings['userId'].isin(sampled_users)]
        
    print(f"Total ratings loaded for experiment: {len(ratings)}")
    
    if 'timestamp' in ratings.columns:
        ratings = ratings.sort_values(by=['userId', 'timestamp'])
    else:
        ratings = ratings.sort_values(by=['userId']) 
        
    test_df = ratings.groupby('userId').tail(1)
    train_df = ratings.drop(test_df.index)
    
    # -------------------------------------------------------------
    # 1. LOAD SVD (Baseline) FROM DISK
    # -------------------------------------------------------------
    print("\n--- Loading SVD Baseline (scikit-surprise) from Disk ---")
    start_time = time.time()
    svd = joblib.load(CF_MODEL_PATH / "svd_model.pkl")
    print(f"SVD Massive 150MB Payload Loaded in: {time.time() - start_time:.2f} seconds")

    # -------------------------------------------------------------
    # 2. LOAD THE MODELS FROM DISK & EVALUATE
    # -------------------------------------------------------------
    print("\n===========================")
    print("[PHASE 2 STARTING] Loading Neural Network exclusively from persistent Disk...")
    print("===========================\n")
    
    production_ncf = NeuralCollaborativeRecommender()
    production_ncf.load()
    
    print("\nPreparing Leave-One-Out Test Set (1 Target + 99 Negatives per User)...")
    all_items = pd.concat([train_df['movieId'], test_df['movieId']]).astype(int).unique()
    all_items_set = set(all_items)
    
    test_data_dict = {}
    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="Test Set"):
        u_str, target_item = str(row.userId), int(row.movieId)
        interacted_m = set(train_df[train_df['userId'] == row.userId]['movieId'].values)
        interacted_m.add(target_item)
        
        negative_items = []
        while len(negative_items) < 99:
            neg_candidate = random.sample(tuple(all_items_set), 1)[0]
            if neg_candidate not in interacted_m:
                negative_items.append(neg_candidate)
                
        test_data_dict[u_str] = (target_item, negative_items)
        
    svd_hr, svd_ndcg, neumf_hr, neumf_ndcg = evaluate_leave_one_out_hybrid(svd, production_ncf, test_data_dict)
    
    print("\n===========================================")
    print("🏆 FINAL PERFORMANCE COMPARISON 🏆")
    print("===========================================")
    print(f"Metric  | SVD         | ULTIMATE NeuMF (w/ Tags & Genres API loading)")
    print(f"HR@10   | {svd_hr:.4f}      | {neumf_hr:.4f}")
    print(f"NDCG@10 | {svd_ndcg:.4f}      | {neumf_ndcg:.4f}")
    print("===========================================")
    if neumf_hr > svd_hr:
        print("✅ The Ultimate Deep Learning architecture devastated SVD!")

if __name__ == "__main__":
    main()

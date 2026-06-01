import os
import sys
import time
import math
import random
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm
# pyrefly: ignore [missing-import]
import torch
import joblib

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    PROCESSED_DATA_DIR,
    MOVIE_FEATURES_PATH,
    MOVIES_DF_PKL_PATH,
    POPULARITY_MODEL_PATH
)
from src.models.popularity_model import PopularityRecommender
from src.models.content_based_model import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeRecommender
from src.models.ncf import NeuralCollaborativeRecommender

# Metrics Helpers
def get_hit_ratio(rank_list, target_item):
    return 1 if target_item in rank_list else 0

def get_ndcg(rank_list, target_item):
    for i in range(len(rank_list)):
        if rank_list[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def normalize_scores(score_dict, eval_items):
    if not score_dict:
        return {k: 0.0 for k in eval_items}
    vals = list(score_dict.values())
    min_val, max_val = min(vals), max(vals)
    denom = max_val - min_val
    if denom == 0:
        return {k: 0.5 for k in score_dict.keys()}
    return {k: float((v - min_val) / denom) for k, v in score_dict.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-users", type=int, default=2000, help="Number of validation users to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("==============================================================")
    print("🔬 RESEARCH-GRADE RECOMMENDATION SYSTEM EVALUATION ENGINE 🔬")
    print("==============================================================")
    
    # ── 1. LOADING DATA ──────────────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    val_path = PROCESSED_DATA_DIR / "val_ratings.csv"
    train_path = PROCESSED_DATA_DIR / "train_ratings.csv"
    
    if not val_path.exists() or not train_path.exists():
        print(f"❌ Error: processed datasets not found in {PROCESSED_DATA_DIR}")
        sys.exit(1)
        
    val_df = pd.read_csv(val_path)
    train_df = pd.read_csv(train_path)
    print(f"   Loaded {len(train_df):,} training ratings, {len(val_df):,} validation ratings.")

    # ── 2. LOADING MODELS ────────────────────────────────────────────────────
    print("\n[2/5] Initializing and loading all models...")
    
    # Model 1: Popularity (Top Rated)
    pop = PopularityRecommender()
    pop.load()
    
    # Model 2: Content-Based
    content = ContentBasedRecommender()
    content.load()
    
    # Model 3: ALS Collaborative Filtering
    als = CollaborativeRecommender()
    als.load()
    
    # Model 4: NCF (NeuMF Deep Learning)
    ncf = NeuralCollaborativeRecommender()
    ncf.load()
    ncf.model.eval()
    device = ncf.device

    # ── 3. FEATURE ENGINEERING FOR CONTENT-BASED SIMILARITY ──────────────────
    print("\n[3/5] Loading and preparing content features...")
    if not MOVIE_FEATURES_PATH.exists():
        print("❌ Error: movie_features.joblib features file not found!")
        sys.exit(1)
        
    movie_features = joblib.load(MOVIE_FEATURES_PATH)
    movies_df = joblib.load(MOVIES_DF_PKL_PATH)
    movie_ids = movies_df["movieId"].tolist()
    movie2idx = {m_id: idx for idx, m_id in enumerate(movie_ids)}
    
    # Normalize features once for O(1) cosine similarity calculation
    from sklearn.preprocessing import normalize
    movie_features_norm = normalize(movie_features, norm="l2")
    
    # All unique movies in the catalog
    all_movies = pd.concat([train_df['movieId'], val_df['movieId']]).astype(int).unique()
    all_movies_set = set(all_movies)

    # ── 4. EVALUATION SAMPLE & CATEGORIES ────────────────────────────────────
    print(f"\n[4/5] Preparing leave-one-out evaluation for {args.num_users} sampled users...")
    
    # Get natural training interaction count for each user in the validation set
    user_train_counts = train_df.groupby('userId').size().to_dict()
    user_train_movies = train_df.groupby('userId')['movieId'].apply(list).to_dict()
    
    val_users = val_df['userId'].unique()
    sampled_users = np.random.choice(val_users, size=min(args.num_users, len(val_users)), replace=False)
    
    val_sample = val_df[val_df['userId'].isin(sampled_users)]
    
    # Set up accumulator tables for metrics
    # Keys: Category -> Model -> Lists of metrics
    categories = ["Cold Start (0)", "Very Sparse (1-5)", "Sparse (5-20)", "Medium (20-50)", "Power User (50+)"]
    models = ["Popularity", "Content-Based", "Collaborative (ALS)", "NCF (NeuMF)", "Hybrid (Adaptive)"]
    
    results = {cat: {model: {"hits": [], "ndcgs": []} for model in models} for cat in categories}
    
    # For Cold Start and Very Sparse, we simulate them for ALL sampled users to get statistical significance
    # For natural types (Sparse, Medium, Power), we map users to their actual train counts.
    
    # ── 5. EVALUATION LOOP ───────────────────────────────────────────────────
    print(f"\n[5/5] Running evaluations (1 target + 99 negatives per user)...")
    
    # Pre-calculated global popularity scores
    global_pop_scores = pop.master_df.set_index("movieId")["popularity_score"].to_dict()
    default_popularity_score = pop.master_df["popularity_score"].mean()

    # Pre-compute NCF movie features index list to avoid rebuilding inside loop
    ncf_item_indices = list(ncf.idx2item.keys())
    ncf_item_ids = [ncf.idx2item[idx] for idx in ncf_item_indices]
    ncf_movie_id_to_idx = ncf.item2idx

    for _, row in tqdm(val_sample.iterrows(), total=len(val_sample), desc="Evaluating Users"):
        userId = row['userId']
        u_str = str(int(userId))
        target_item = int(row['movieId'])
        
        # Get natural training history
        history = user_train_movies.get(userId, [])
        natural_count = len(history)
        
        # Determine natural user category
        if natural_count == 0:
            natural_cat = "Cold Start (0)"
        elif natural_count <= 5:
            natural_cat = "Very Sparse (1-5)"
        elif natural_count <= 20:
            natural_cat = "Sparse (5-20)"
        elif natural_count <= 50:
            natural_cat = "Medium (20-50)"
        else:
            natural_cat = "Power User (50+)"
            
        # Sample 99 Negatives (movies not rated by the user in train or val)
        interacted_set = set(history)
        interacted_set.add(target_item)
        
        negatives = []
        while len(negatives) < 99:
            candidate = random.choice(all_movies)
            if candidate not in interacted_set:
                negatives.append(candidate)
                
        eval_items = [target_item] + negatives
        random.shuffle(eval_items)
        
        # Define evaluation configurations to run:
        # Each config: (target_cat_name, simulated_history)
        eval_configs = []
        
        # 1. Simulated Cold Start
        eval_configs.append(("Cold Start (0)", []))
        
        # 2. Simulated Very Sparse (exactly 3 ratings, or min of 3 and natural_count)
        simulated_history_sparse = random.sample(history, min(3, len(history))) if history else []
        eval_configs.append(("Very Sparse (1-5)", simulated_history_sparse))
        
        # 3. Natural State
        eval_configs.append((natural_cat, history))
        
        # Precompute features/indices of candidate items once per user to optimize
        cand_indices = [movie2idx[m] for m in eval_items if m in movie2idx]
        
        # Run each evaluation configuration
        for cat_name, u_history in eval_configs:
            # ──────────────────────────────────────────────────────
            # MODEL A: Popularity-Based (Global Baseline)
            # ──────────────────────────────────────────────────────
            pop_preds = {}
            for item in eval_items:
                pop_preds[item] = global_pop_scores.get(item, default_popularity_score)
                
            pop_ranked = sorted(pop_preds.items(), key=lambda x: x[1], reverse=True)[:10]
            pop_rank_list = [i for i, _ in pop_ranked]
            results[cat_name]["Popularity"]["hits"].append(get_hit_ratio(pop_rank_list, target_item))
            results[cat_name]["Popularity"]["ndcgs"].append(get_ndcg(pop_rank_list, target_item))
            
            # ──────────────────────────────────────────────────────
            # MODEL B: Content-Based Recommender
            # ──────────────────────────────────────────────────────
            content_preds = {}
            if not u_history:
                # Cold start fallback: rank randomly or set all to 0
                for item in eval_items:
                    content_preds[item] = 0.0
            else:
                # Filter history items that are in features index
                valid_hist = [m for m in u_history if m in movie2idx]
                if not valid_hist:
                    for item in eval_items:
                        content_preds[item] = 0.0
                else:
                    hist_indices = [movie2idx[m] for m in valid_hist]
                    
                    # Compute dot product only on items present in movie2idx
                    valid_cand_items = [m for m in eval_items if m in movie2idx]
                    cand_indices_cb = [movie2idx[m] for m in valid_cand_items]
                    
                    cand_feats = movie_features_norm[cand_indices_cb]
                    hist_feats = movie_features_norm[hist_indices]
                    
                    # Cosine similarity matrix (len(valid_cand_items), N)
                    sim_matrix = cand_feats @ hist_feats.T
                    if sparse.issparse(sim_matrix):
                        sim_matrix = sim_matrix.toarray()
                        
                    # Max similarity to history
                    max_sims = sim_matrix.max(axis=1)
                    
                    idx_mapping = {valid_cand_items[i]: float(max_sims[i]) for i in range(len(valid_cand_items))}
                    for item in eval_items:
                        content_preds[item] = idx_mapping.get(item, 0.0)
                        
            content_ranked = sorted(content_preds.items(), key=lambda x: x[1], reverse=True)[:10]
            content_rank_list = [i for i, _ in content_ranked]
            results[cat_name]["Content-Based"]["hits"].append(get_hit_ratio(content_rank_list, target_item))
            results[cat_name]["Content-Based"]["ndcgs"].append(get_ndcg(content_rank_list, target_item))
            
            # ──────────────────────────────────────────────────────
            # MODEL C: Collaborative Filtering (ALS)
            # ──────────────────────────────────────────────────────
            als_preds = {}
            if not u_history:
                # Cold start: fallback to popularity scores
                for item in eval_items:
                    als_preds[item] = global_pop_scores.get(item, 0.0)
            elif len(u_history) < natural_count:
                # Simulated Sparsity: recalculate user factors on the fly using recalculate_item!
                n_items = len(als.item2idx)
                cols = [als.item2idx[int(m)] for m in u_history if int(m) in als.item2idx]
                # Default ratings of 4.0 for confidence if not lookup-able, or look it up
                # For simplicity we use 4.0
                data = [4.0] * len(cols)
                rows = np.zeros(len(cols), dtype=np.int32)
                
                if cols:
                    user_items_csr = sparse.csr_matrix((data, (rows, cols)), shape=(1, n_items))
                    # Call recalculate_item (transposed)
                    user_vector_als = als.model.recalculate_item(0, user_items_csr)
                    
                    for item in eval_items:
                        if int(item) in als.item2idx:
                            i_idx = als.item2idx[int(item)]
                            als_preds[item] = np.dot(user_vector_als, als.model.user_factors[i_idx])
                        else:
                            als_preds[item] = -np.inf
                else:
                    for item in eval_items:
                        als_preds[item] = -np.inf
            else:
                # Natural state (already in training set)
                if u_str in als.user2idx:
                    u_idx_als = als.user2idx[u_str]
                    user_vector_als = als.model.item_factors[u_idx_als]
                    for item in eval_items:
                        if int(item) in als.item2idx:
                            i_idx_als = als.item2idx[int(item)]
                            als_preds[item] = np.dot(user_vector_als, als.model.user_factors[i_idx_als])
                        else:
                            als_preds[item] = -np.inf
                else:
                    # Fallback if not found in mappings (live database users)
                    for item in eval_items:
                        als_preds[item] = global_pop_scores.get(item, 0.0)
                        
            als_ranked = sorted(als_preds.items(), key=lambda x: x[1], reverse=True)[:10]
            als_rank_list = [i for i, _ in als_ranked]
            results[cat_name]["Collaborative (ALS)"]["hits"].append(get_hit_ratio(als_rank_list, target_item))
            results[cat_name]["Collaborative (ALS)"]["ndcgs"].append(get_ndcg(als_rank_list, target_item))
            
            # ──────────────────────────────────────────────────────
            # MODEL D: Neural Collaborative Filtering (NeuMF)
            # ──────────────────────────────────────────────────────
            ncf_preds = {}
            if not u_history or len(u_history) < 5:
                # Cold start or simulated very sparse (before fine-tuning updates embeddings): fallback to popularity
                for item in eval_items:
                    ncf_preds[item] = global_pop_scores.get(item, 0.0)
            else:
                # Natural state (and has mappings)
                if u_str in ncf.user2idx:
                    u_idx_ncf = ncf.user2idx[u_str]
                    
                    # Filter items in NCF mappings
                    valid_eval_items = [m for m in eval_items if m in ncf.item2idx]
                    
                    if target_item not in valid_eval_items:
                        # If target item is missing, fallback to popularity
                        for item in eval_items:
                            ncf_preds[item] = global_pop_scores.get(item, 0.0)
                    else:
                        items_df_ncf = pd.DataFrame({
                            'movieId': valid_eval_items, 
                            'item_idx': [ncf.item2idx[m] for m in valid_eval_items]
                        })
                        items_df_ncf = items_df_ncf.merge(ncf.movies_df[['movieId', 'genres']], on='movieId', how='left')
                        items_df_ncf['genres'] = items_df_ncf['genres'].fillna("Unknown").str.split('|')
                        genres_encoded = ncf.mlb.transform(items_df_ncf['genres'])
                        
                        default_pad = [0] * ncf.max_tags_per_movie
                        tags_array = items_df_ncf['movieId'].map(lambda m: ncf.movie_tags.get(m, default_pad)).tolist()
                        tags_np = np.array(tags_array)
                        
                        user_tensor = torch.tensor([u_idx_ncf] * len(valid_eval_items), dtype=torch.long).to(device)
                        item_tensor = torch.tensor(items_df_ncf['item_idx'].values, dtype=torch.long).to(device)
                        genre_tensor = torch.tensor(genres_encoded, dtype=torch.float32).to(device)
                        tag_tensor = torch.tensor(tags_np, dtype=torch.long).to(device)
                        
                        with torch.no_grad():
                            ncf_probs = ncf.model(user_tensor, item_tensor, genre_tensor, tag_tensor).cpu().numpy()
                            
                        idx_mapping = {valid_eval_items[i]: float(ncf_probs[i]) for i in range(len(valid_eval_items))}
                        for item in eval_items:
                            ncf_preds[item] = idx_mapping.get(item, -np.inf)
                else:
                    # Fallback to popularity
                    for item in eval_items:
                        ncf_preds[item] = global_pop_scores.get(item, 0.0)
                        
            ncf_ranked = sorted(ncf_preds.items(), key=lambda x: x[1], reverse=True)[:10]
            ncf_rank_list = [i for i, _ in ncf_ranked]
            results[cat_name]["NCF (NeuMF)"]["hits"].append(get_hit_ratio(ncf_rank_list, target_item))
            results[cat_name]["NCF (NeuMF)"]["ndcgs"].append(get_ndcg(ncf_rank_list, target_item))

            # ──────────────────────────────────────────────────────
            # MODEL E: Hybrid (Adaptive) Recommender
            # ──────────────────────────────────────────────────────
            norm_pop = normalize_scores(pop_preds, eval_items)
            norm_content = normalize_scores(content_preds, eval_items)
            
            # ALS clean -inf
            als_valid_scores = {k: v for k, v in als_preds.items() if v != -np.inf}
            als_min = min(als_valid_scores.values()) if als_valid_scores else -10.0
            als_preds_clean = {k: (v if v != -np.inf else als_min) for k, v in als_preds.items()}
            norm_als = normalize_scores(als_preds_clean, eval_items)
            
            # NCF clean -inf
            ncf_valid_scores = {k: v for k, v in ncf_preds.items() if v != -np.inf}
            ncf_min = min(ncf_valid_scores.values()) if ncf_valid_scores else 0.0
            ncf_preds_clean = {k: (v if v != -np.inf else ncf_min) for k, v in ncf_preds.items()}
            norm_ncf = normalize_scores(ncf_preds_clean, eval_items)
            
            if cat_name == "Cold Start (0)":
                w = {"pop": 1.0, "content": 0.0, "als": 0.0, "ncf": 0.0}
            elif cat_name == "Very Sparse (1-5)":
                w = {"pop": 0.1, "content": 0.7, "als": 0.2, "ncf": 0.0}
            else:
                w = {"pop": 0.1, "content": 0.2, "als": 0.3, "ncf": 0.4}
                
            hybrid_preds = {}
            for item in eval_items:
                hybrid_preds[item] = (
                    w["pop"] * norm_pop[item] +
                    w["content"] * norm_content[item] +
                    w["als"] * norm_als[item] +
                    w["ncf"] * norm_ncf[item]
                )
                
            hybrid_ranked = sorted(hybrid_preds.items(), key=lambda x: x[1], reverse=True)[:10]
            hybrid_rank_list = [i for i, _ in hybrid_ranked]
            results[cat_name]["Hybrid (Adaptive)"]["hits"].append(get_hit_ratio(hybrid_rank_list, target_item))
            results[cat_name]["Hybrid (Adaptive)"]["ndcgs"].append(get_ndcg(hybrid_rank_list, target_item))

    # ── 6. COMPILE COMPARATIVE REPORT ────────────────────────────────────────
    print("\n[Report] Generating Comparative Table...")
    
    markdown_lines = []
    markdown_lines.append("# Comparative Recommendation Models Research Evaluation")
    markdown_lines.append(f"Evaluated on a random sample of {args.num_users} users using the Leave-One-Out (LOO) protocol.\n")
    
    markdown_lines.append("| User Type | Ratings | Model | Sample Size | HR@10 | NDCG@10 |")
    markdown_lines.append("|---|---|---|---|---|---|")
    
    print("\n" + "="*80)
    print("🏆 FINAL COMPARATIVE RESEARCH BENCHMARK 🏆")
    print("="*80)
    print(f"{'User Type':<18} | {'Model':<20} | {'Users':<6} | {'HR@10':<7} | {'NDCG@10':<7}")
    print("-" * 80)

    for cat in categories:
        for model in models:
            hits = results[cat][model]["hits"]
            ndcgs = results[cat][model]["ndcgs"]
            
            sample_size = len(hits)
            mean_hr = np.mean(hits) if sample_size > 0 else 0.0
            mean_ndcg = np.mean(ndcgs) if sample_size > 0 else 0.0
            
            # Format display strings
            hr_str = f"{mean_hr:.4f}"
            ndcg_str = f"{mean_ndcg:.4f}"
            
            print(f"{cat:<18} | {model:<20} | {sample_size:<6} | {hr_str:<7} | {ndcg_str:<7}")
            markdown_lines.append(f"| {cat} | {cat.split('(')[1][:-1]} | {model} | {sample_size} | {hr_str} | {ndcg_str} |")
            
        print("-" * 80)
        
    # Write report file
    report_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "evaluation_report.md")
    
    with open(report_path, "w") as f:
        f.write("\n".join(markdown_lines))
        
    print(f"\n[Report] Saved research report to: {report_path}")
    print("==============================================================\n")

if __name__ == "__main__":
    main()

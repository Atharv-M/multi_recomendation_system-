import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import logging
from pathlib import Path
import random
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

from src.config import RATINGS_DIR, DATA_DIR, MOVIES_DF_PKL_PATH, NCF_MODEL_PATH

logger = logging.getLogger(__name__)
TAGS_DIR = DATA_DIR / "raw" / "tag.csv"

class HybridImplicitDataset(Dataset):
    def __init__(self, users, items, timestamps, labels, movie_genre_matrix, movie_tags_matrix):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        # Tiny Lookup Dictionaries
        self.movie_genre_matrix = torch.tensor(movie_genre_matrix, dtype=torch.float32)
        self.movie_tags_matrix = torch.tensor(movie_tags_matrix, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        return u, i, self.movie_genre_matrix[i], self.timestamps[idx], self.movie_tags_matrix[i], self.labels[idx]

# HYBRID NeuMF MODEL

class HybridNeuMFModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, num_tags, mf_dim=16, mlp_dim=32, tag_dim=16, hidden_layers=[128, 64, 32]):
        super(HybridNeuMFModel, self).__init__()
        
        # --- GMF Brain (User x Item ID Matrix Factoring) ---
        self.embedding_user_mf = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.embedding_item_mf = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)
        
        # --- MLP Brain (Deep Content Understanding) ---
        self.embedding_user_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=mlp_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=mlp_dim)
        
        # The Secret Weapon: Tags
        self.tag_embedding = nn.EmbeddingBag(num_embeddings=num_tags, embedding_dim=tag_dim, mode='mean', padding_idx=0)
        
        # Genres
        self.genre_layer = nn.Sequential(
            nn.Linear(num_genres, 16),
            nn.ReLU()
        )
        
        # Input = user_mlp(32) + item_mlp(32) + genre(16) + tag(16) + timestamp(1) = 97
        input_dim = mlp_dim * 2 + 16 + tag_dim + 1
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2)) 
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
        # --- Final Output Layer ---
        # Concat GMF vector (16) + final MLP vector (32) = 48
        final_dim = mf_dim + hidden_layers[-1]
        self.prediction_layer = nn.Sequential(
            nn.Linear(final_dim, 1),
            nn.Sigmoid() 
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.prediction_layer[0].weight, nonlinearity='sigmoid')

    def forward(self, user_indices, item_indices, genres, timestamps, tags):
        # 1. GMF Brain
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf) 
        
        # 2. MLP Content Brain
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        
        tag_vector = self.tag_embedding(tags)
        genre_vector = self.genre_layer(genres)
        time_vector = timestamps.unsqueeze(1)
        
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp, genre_vector, time_vector, tag_vector], dim=1)
        mlp_vector = self.mlp(mlp_vector)
        
        # 3. Concatenate and predict
        neumf_vector = torch.cat([mf_vector, mlp_vector], dim=1)
        prediction = self.prediction_layer(neumf_vector)
        return prediction.squeeze(1)

# RECOMMENDER WRAPPER

class NeuralCollaborativeRecommender:
    def __init__(self, mf_dim=16, mlp_dim=32, tag_dim=16, n_epochs=6, batch_size=4096, lr=0.001):
        self.mf_dim = mf_dim
        self.mlp_dim = mlp_dim
        self.tag_dim = tag_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = None
        self.movies_df = None
        
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        
        self.tag2idx = {"<PAD>": 0} 
        self.idx2tag = {0: "<PAD>"}
        self.movie_tags = {} 
        self.max_tags_per_movie = 20
        
        self.mlb = MultiLabelBinarizer()
        self.time_scaler = MinMaxScaler()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _prepare_movie_tags(self):
        logger.info("Parsing 20k Tags for Bag Processing...")
        if not TAGS_DIR.exists():
            return
            
        tags_df = pd.read_csv(TAGS_DIR)
        tags_df['tag'] = tags_df['tag'].astype(str).str.lower().str.strip()
        
        counts = tags_df['tag'].value_counts()
        valid_tags = counts[counts >= 3].index
        tags_df = tags_df[tags_df['tag'].isin(valid_tags)]
        
        unique_tags_list = tags_df['tag'].unique()
        for idx, tag in enumerate(unique_tags_list, start=1):
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag
            
        logger.info(f"Loaded {len(self.tag2idx)} unique valid tags.")
        
        grouped = tags_df.groupby('movieId')['tag'].apply(list).reset_index()
        
        for _, row in grouped.iterrows():
            m_id = row['movieId']
            word_list = row['tag']
            
            int_list = [self.tag2idx[w] for w in word_list if w in self.tag2idx]
            int_list = list(set(int_list)) 
            
            if len(int_list) > self.max_tags_per_movie:
                int_list = int_list[:self.max_tags_per_movie]
            
            pads_needed = self.max_tags_per_movie - len(int_list)
            padded_list = int_list + [0] * pads_needed
            self.movie_tags[m_id] = padded_list

    def fit(self, train_df):
        logger.info("Initializing massive 5x Vectorized Matrix for PyTorch DataLoaders...")
        movies_df = joblib.load(MOVIES_DF_PKL_PATH)
        self.movies_df = movies_df
        
        self._prepare_movie_tags()
        
        merged = train_df.merge(movies_df[['movieId', 'genres']], on='movieId', how='left')
        merged['genres'] = merged['genres'].fillna("Unknown").str.split('|')
        self.mlb.fit(merged['genres'])
        
        if 'timestamp' in train_df.columns:
            # Safely coerce timestamps so we don't accidentally drop NA rows and break tensor shapes!
            numeric_ts = pd.to_numeric(train_df['timestamp'], errors='coerce').fillna(0).astype(np.float32)
            self.time_scaler.fit(numeric_ts.values.reshape(-1, 1))
            
        all_users = train_df['userId'].astype(str).unique()
        all_items = train_df['movieId'].astype(int).unique()
        
        self.user2idx = {str(u): i for i, u in enumerate(all_users)}
        self.idx2user = {idx: user_id for user_id, idx in self.user2idx.items()}
        self.item2idx = {int(i): idx for idx, i in enumerate(all_items)}
        self.idx2item = {idx: item_id for item_id, idx in self.item2idx.items()}
        
        user_interacted = train_df.groupby('userId')['movieId'].apply(
            lambda x: set(x.astype(int).map(self.item2idx))
        ).to_dict()
        
        n_train = len(train_df)
        n_total = n_train * 3  # 1 positive + 2 negatives (fits Kaggle 30GB RAM safe limit)

        u_arr = np.empty(n_total, dtype=np.int32)
        i_arr = np.empty(n_total, dtype=np.int32)
        y_arr = np.empty(n_total, dtype=np.float32)

        users_mapped = train_df['userId'].astype(str).map(self.user2idx).values
        items_mapped = train_df['movieId'].astype(int).map(self.item2idx).values
        
        u_arr[0:n_train] = users_mapped
        i_arr[0:n_train] = items_mapped
        y_arr[0:n_train] = 1.0
        
        num_items = len(all_items)
        idx = n_train
        random_negatives = np.random.randint(0, num_items, size=(n_train, 2))
        user_ids_original = train_df['userId'].values
        
        logger.info("Hashing Negative Samples to avoid overlapping existing interactions...")
        from tqdm import tqdm
        for i in tqdm(range(n_train), desc="Vector Collision Avoidance"):
            u_original = user_ids_original[i]
            interacted = user_interacted[u_original]
            
            negs = random_negatives[i]
            while any(n in interacted for n in negs):
                negs = np.random.randint(0, num_items, size=2)
                
            u_arr[idx:idx+2] = users_mapped[i]
            i_arr[idx:idx+2] = negs
            y_arr[idx:idx+2] = 0.0
            idx += 2

        logger.info("Initializing O(1) Memory-Efficient Dictionary Arrays...")
        num_genres = len(self.mlb.classes_)
        num_tags = len(self.tag2idx)
        
        movies_df['genres_split'] = movies_df['genres'].fillna("Unknown").str.split('|')
        all_genres_encoded = self.mlb.transform(movies_df['genres_split'])
        m_id_to_genre = {m_id: all_genres_encoded[i] for i, m_id in enumerate(movies_df['movieId'].values)}
        
        movie_genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)
        movie_tags_matrix = np.zeros((num_items, self.max_tags_per_movie), dtype=np.int32)
        
        default_pad = [0] * self.max_tags_per_movie
        for item_idx, m_id in self.idx2item.items():
            movie_tags_matrix[item_idx] = self.movie_tags.get(m_id, default_pad)
            movie_genre_matrix[item_idx] = m_id_to_genre.get(m_id, np.zeros(num_genres, dtype=np.float32))
        
        if 'timestamp' in train_df.columns:
            numeric_ts_pos = pd.to_numeric(train_df['timestamp'], errors='coerce').fillna(0).astype(np.float32)
            t_scaled = self.time_scaler.transform(numeric_ts_pos.values.reshape(-1, 1)).flatten()
        else:
            t_scaled = np.zeros(n_train)
            
        t_arr = np.zeros(n_total, dtype=np.float32)
        t_arr[0:n_train] = t_scaled
        
        train_dataset = HybridImplicitDataset(u_arr, i_arr, t_arr, y_arr, movie_genre_matrix, movie_tags_matrix)

        # num_workers=0: dataset is pure pre-computed in-memory tensors.
        # __getitem__ = tensor[idx] — instant, zero disk I/O.
        # workers=4 gave ZERO speed benefit but forked 4×3.8GB RAM copies → OOM crash.
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,   # main process only — no fork, no RAM explosion
            pin_memory=True, # still pins RAM for fast CPU→GPU transfer
        )
        
        num_genres = len(self.mlb.classes_)
        num_tags = len(self.tag2idx)
        
        self.model = HybridNeuMFModel(len(all_users), len(all_items), num_genres, num_tags, 
                                      self.mf_dim, self.mlp_dim, self.tag_dim).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        logger.info(f"Igniting massive Neural Network training across {len(u_arr)} rows on Apple {self.device}...")
        import time
        start_time = time.time()
        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            for batch_u, batch_i, batch_g, batch_t, batch_tg, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch_u, batch_i = batch_u.to(self.device), batch_i.to(self.device)
                batch_g, batch_t = batch_g.to(self.device), batch_t.to(self.device)
                batch_tg, batch_y = batch_tg.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_u, batch_i, batch_g, batch_t, batch_tg)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_u)
            logger.info(f"Epoch {epoch+1}/{self.n_epochs} - BCE Loss: {total_loss/len(train_dataset):.4f}")
        logger.info(f"NeuMF God-Tier Training Time: {time.time() - start_time:.2f} seconds")

    def save(self):
        if self.model is None:
            raise ValueError("Model not trained.")

        NCF_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        mappings = {
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'item2idx': self.item2idx,
            'idx2item': self.idx2item,
            'mf_dim': self.mf_dim,
            'mlp_dim': self.mlp_dim,
            'tag_dim': self.tag_dim,
            'mlb': self.mlb,
            'time_scaler': self.time_scaler,
            'tag2idx': self.tag2idx,
            'idx2tag': self.idx2tag,
            'movie_tags': self.movie_tags
        }
        joblib.dump(mappings, NCF_MODEL_PATH / "neumf_mappings.pkl")
        torch.save(self.model.state_dict(), NCF_MODEL_PATH / "neumf_weights.pt")
        logger.info("Hybrid NeuMF model saved.")

    def load(self):
        logger.info("Loading Hybrid NeuMF mappings and weights...")
        mappings = joblib.load(NCF_MODEL_PATH / "neumf_mappings.pkl")
        self.user2idx = mappings['user2idx']
        self.idx2user = mappings['idx2user']
        self.item2idx = mappings['item2idx']
        self.idx2item = mappings['idx2item']
        self.mf_dim = mappings['mf_dim']
        self.mlp_dim = mappings['mlp_dim']
        self.tag_dim = mappings['tag_dim']
        self.mlb = mappings['mlb']
        self.time_scaler = mappings['time_scaler']
        self.tag2idx = mappings['tag2idx']
        self.idx2tag = mappings['idx2tag']
        self.movie_tags = mappings['movie_tags']
        
        num_users = len(self.user2idx)
        num_items = len(self.item2idx)
        num_genres = len(self.mlb.classes_)
        num_tags = len(self.tag2idx)
        
        self.model = HybridNeuMFModel(num_users, num_items, num_genres, num_tags, self.mf_dim, self.mlp_dim, self.tag_dim)
        self.model.load_state_dict(torch.load(NCF_MODEL_PATH / "neumf_weights.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.movies_df = joblib.load(MOVIES_DF_PKL_PATH)

    def finetune(self, new_ratings_df, epochs=15, lr=0.005):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() before finetune()!")
            
        logger.info(f"Initiating surgical Deep Learning Fine-Tuning for {len(new_ratings_df)} Database interactions...")
        
        # 1. Map new UUIDs to __GUEST__ slots structurally
        unique_new_users = new_ratings_df['userId'].astype(str).unique()
        users_injected = 0
        for u in unique_new_users:
            if u not in self.user2idx:
                guest_keys = [k for k in self.user2idx.keys() if str(k).startswith("__GUEST_")]
                if not guest_keys:
                    logger.error("FATAL: Out of __GUEST__ padding slots! Cannot inject new users.")
                    continue
                available_guest = guest_keys[0]
                idx = self.user2idx.pop(available_guest) # Remove guest name
                # Overwrite coordinate identically!
                self.user2idx[u] = idx
                self.idx2user[idx] = u
                logger.info(f"Surgically assigned new user UUID {u} to Neural Tensor Row {idx}")
                users_injected += 1
                
        # 2. Extract Valid Rows
        users_mapped = []
        items_mapped = []
        for index, row in new_ratings_df.iterrows():
            u_str = str(row['userId'])
            m_id = int(row['movieId'])
            if u_str in self.user2idx and m_id in self.item2idx:
                users_mapped.append(self.user2idx[u_str])
                items_mapped.append(self.item2idx[m_id])
                
        n_valid = len(users_mapped)
        if n_valid == 0:
            logger.info("No valid mappings for fine-tuning. Ignoring iteration.")
            return
            
        # 3. Micro Negative Sampling
        n_total_valid = n_valid * 5
        u_arr = np.empty(n_total_valid, dtype=np.int32)
        i_arr = np.empty(n_total_valid, dtype=np.int32)
        y_arr = np.empty(n_total_valid, dtype=np.float32)
        
        u_arr[0:n_valid] = users_mapped
        i_arr[0:n_valid] = items_mapped
        y_arr[0:n_valid] = 1.0
        
        all_items_list = list(self.item2idx.values())
        idx_offset = n_valid
        for i in range(n_valid):
            negs = np.random.choice(all_items_list, size=4, replace=False)
            u_arr[idx_offset:idx_offset+4] = users_mapped[i]
            i_arr[idx_offset:idx_offset+4] = negs
            y_arr[idx_offset:idx_offset+4] = 0.0
            idx_offset += 4
            
        # 4. Deep Feature Extraction (Genres/Tags)
        items_df = pd.DataFrame({'movieId': [self.idx2item[i] for i in i_arr], 'item_idx': i_arr})
        items_df = items_df.merge(self.movies_df[['movieId', 'genres']], on='movieId', how='left')
        items_df['genres'] = items_df['genres'].fillna("Unknown").str.split('|')
        
        genres_encoded = self.mlb.transform(items_df['genres'])
        timestamps_scaled = np.ones(len(items_df))
        
        default_pad = [0] * self.max_tags_per_movie
        tags_array = items_df['movieId'].map(lambda m: self.movie_tags.get(m, default_pad)).tolist()
        tags_np = np.array(tags_array)
        
        # Deploy straight to GPU/CPU Ram Array limits bypassed!
        user_tensor = torch.tensor(u_arr, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(i_arr, dtype=torch.long).to(self.device)
        genre_tensor = torch.tensor(genres_encoded, dtype=torch.float32).to(self.device)
        time_tensor = torch.tensor(timestamps_scaled, dtype=torch.float32).to(self.device)
        tag_tensor = torch.tensor(tags_np, dtype=torch.long).to(self.device)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).to(self.device)
        
        # 5. Ignite the Optimizer Native Micro-Batch loop
        self.model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        logger.info(f"Targeting Backpropagation on {n_total_valid} User Interactions...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(user_tensor, item_tensor, genre_tensor, time_tensor, tag_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 5 == 0:
                logger.info(f"FineTune Epoch {epoch+1}/{epochs} - Micro BCE Loss: {loss.item():.4f}")
                
        logger.info(f"God-Tier Fine-Tuning execution fully complete across {users_injected} New Sign-Ups!")


    def recommend(self, user_id: str, top_k: int = 10):
        if self.model is None:
            raise ValueError("Model not loaded.")

        str_user_id = str(user_id)
        if str_user_id not in self.user2idx:
            logger.warning("User not found.")
            return pd.DataFrame(columns=["movieId", "title", "genres"])
            
        user_idx = self.user2idx[str_user_id]
        
        all_item_indices = list(self.idx2item.keys())
        all_item_ids = [self.idx2item[idx] for idx in all_item_indices]
        
        items_df = pd.DataFrame({'movieId': all_item_ids, 'item_idx': all_item_indices})
        items_df = items_df.merge(self.movies_df[['movieId', 'genres']], on='movieId', how='left')
        items_df['genres'] = items_df['genres'].fillna("Unknown").str.split('|')
        
        genres_encoded = self.mlb.transform(items_df['genres'])
        timestamps_scaled = np.ones(len(items_df)) 
        
        default_pad = [0] * self.max_tags_per_movie
        tags_array = items_df['movieId'].map(lambda m: self.movie_tags.get(m, default_pad)).tolist()
        tags_np = np.array(tags_array)
        
        user_tensor = torch.tensor([user_idx] * len(all_item_indices), dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(items_df['item_idx'].values, dtype=torch.long).to(self.device)
        genre_tensor = torch.tensor(genres_encoded, dtype=torch.float32).to(self.device)
        time_tensor = torch.tensor(timestamps_scaled, dtype=torch.float32).to(self.device)
        tag_tensor = torch.tensor(tags_np, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(user_tensor, item_tensor, genre_tensor, time_tensor, tag_tensor).cpu().numpy()
            
        item_preds = list(zip(all_item_ids, predictions))
        item_preds.sort(key=lambda x: x[1], reverse=True) 
        
        top_items = item_preds[:top_k]
        top_movie_ids = [m_id for m_id, _ in top_items]

        return self.movies_df[self.movies_df["movieId"].isin(top_movie_ids)][["movieId", "title", "genres"]]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger.info("Starting Dedicated NCF Training Script...")

    import sys
    use_full = "--use-full-data" in sys.argv

    if use_full:
        train_path = RATINGS_DIR  # full rating.csv — for production after evaluation
        logger.info("Mode: FULL DATA (production retrain after evaluation)")
    else:
        train_path = DATA_DIR / "processed" / "train_ratings.csv"  # train split only
        logger.info("Mode: TRAIN SPLIT ONLY (for honest evaluation)")

    logger.info(f"Loading ratings from {train_path}...")
    ratings = pd.read_csv(train_path)

    logger.info(f"Loaded {len(ratings):,} interactions. Starting training...")

    recommender = NeuralCollaborativeRecommender(n_epochs=5, batch_size=4096)
    recommender.fit(ratings)

    logger.info("Saving model weights and mappings...")
    recommender.save()
    logger.info("NCF training complete — model saved successfully.")


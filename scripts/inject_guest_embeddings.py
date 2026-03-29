import torch
import joblib
from src.config import NCF_MODEL_PATH
from src.models.ncf import HybridNeuMFModel

def pad_embeddings():
    print("=========================")
    print("BEGINNING DEEP LEARNING SURGERY")
    print("=========================")
    
    # 1. LOAD THE GUESTBOOK (MAPPINGS)
    mappings_path = NCF_MODEL_PATH / "neumf_mappings.pkl"
    mappings = joblib.load(mappings_path)
    
    original_num_users = len(mappings['user2idx'])
    print(f"Original User Count (Rooms Built): {original_num_users}")
    
    # 2. HACK THE GUESTBOOK WITH 10,000 FAKE NAMES
    PAD_SIZE = 10000
    for i in range(PAD_SIZE):
        fake_name = f"__GUEST_{i}__"
        fake_index = original_num_users + i
        mappings['user2idx'][fake_name] = fake_index
        mappings['idx2user'][fake_index] = fake_name
        
    print(f"Hacked User Count (New Rooms): {len(mappings['user2idx'])}")
    
    # Save the hacked guestbook permanently!
    joblib.dump(mappings, mappings_path)
    print("Guestbook perfectly overwritten.")
    
    # 3. LOAD THE PYTORCH WEIGHTS (THE GUESTS)
    weights_path = NCF_MODEL_PATH / "neumf_weights.pt"
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # Extract the user brains
    old_mf = state_dict['embedding_user_mf.weight']
    old_mlp = state_dict['embedding_user_mlp.weight']
    
    # Forge 10,000 new blank brains (Zeros)
    mf_zeros = torch.zeros((PAD_SIZE, old_mf.shape[1]), dtype=torch.float32)
    mlp_zeros = torch.zeros((PAD_SIZE, old_mlp.shape[1]), dtype=torch.float32)
    
    # Glue them to the bottom of the original matrix structurally!
    new_mf = torch.cat([old_mf, mf_zeros], dim=0)
    new_mlp = torch.cat([old_mlp, mlp_zeros], dim=0)
    
    # Overwrite the dictionary with the massive matrices
    state_dict['embedding_user_mf.weight'] = new_mf
    state_dict['embedding_user_mlp.weight'] = new_mlp
    
    # 4. SURGICALLY UPDATE THE PREDICTION LAYER BATCHNORM (IF NECESSARY)
    # Note: Only User Embeddings control the ID dimension size. Everything else is globally invariant!
    
    # Save the huge PyTorch binary!
    torch.save(state_dict, weights_path)
    print("PyTorch Weights successfully injected with 10,000 blank vectors.")
    
    print("\n--- VALIDATION TEST ---")
    from src.models.ncf import NeuralCollaborativeRecommender
    test_recommender = NeuralCollaborativeRecommender()
    test_recommender.load()
    print("SUCCESS: The ncf.py Recommender Engine booted flawlessly on the Hacked Weights without crashing!")
    print("=========================")

if __name__ == "__main__":
    pad_embeddings()

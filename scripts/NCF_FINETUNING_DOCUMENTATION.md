# Online Incremental Learning (Fine-Tuning) Architecture for NeuMF

This document serves as the official technical outline for how this repository achieves true **Real-Time Online Learning** for our fully-compiled 100-Million-Row PyTorch Deep Learning Model without having to re-run expensive 3-hour cluster compilations.

## 1. The Core Problem: "Cold-Start" Embeddings
Our production Collaborative Filtering model (NeuMF) relies fundamentally on a mathematical PyTorch structure called `nn.Embedding`. During our Kaggle GPU training phase, PyTorch instantiated a matrix exactly sized to the historical users available in `rating.csv` (`138,493` Users).

When a brand-new user signs up on our React/FastAPI frontend, Supabase issues them a fresh `UUID` string. If we fed this UUID to the Kaggle-trained model, PyTorch would trigger a fatal `IndexError: index out of bounds` because the matrix was physically enclosed at exactly `138,493` slots.

To recommend movies to new users, we needed a way to structurally train new vectors specifically for these new UUIDs without breaking the architecture.

## 2. Phase 1: Deep Learning State-Dict Surgery
Instead of throwing away the 3 hours of Kaggle GPU mathematics and retraining from scratch, we constructed a localized PyTorch surgery script: `scripts/inject_guest_embeddings.py`.

What the script executed:
1. **Model Extraction:** It successfully cracked open the compiled `15.0 MB` `neumf_weights.pt` binary.
2. **Computational Padding:** It programmatically elongated the `embedding_user_mf` and `embedding_user_mlp` parameter matrices by slicing `10,000` extra empty zero-matrices (`torch.zeros`) directly onto the bottom of the structure.
3. **Dictionary Spoofing:** To prevent the `ncf.py` script from crashing when initializing a model against our bloated matrix, we intercepted the `neumf_mappings.pkl` serial dictionary. We injected `10,000` fake placeholder keys (e.g., `"__GUEST_0__"`, `"__GUEST_1__"`) bridging the numerical gap up to `148,493`.
4. The `.pt` file was safely re-saved back to the hard drive holding completely dormant and mathematically empty AI brains.

## 3. Phase 2: Live Supabase Extraction
With `10,000` empty variables dynamically sitting inside the PyTorch state, we engineered `scripts/finetune_ncf.py` to seamlessly connect your production web interface to the AI engine.

1. **Extraction:** The script boots up the Supabase PostgreSQL API and physically downloads every single raw rating submitted from the Web UI's Onboarding Modal.
2. **Translation:** It takes the Supabase SQL Rows and reformats them perfectly into Pandas data frames natively accepted by the NeuMF PyTorch engine.

## 4. Phase 3: Surgical `.train()` Replacements (Preventing Catastrophic Forgetting)
Inside the core `ncf.py` logic, we developed a highly advanced `finetune()` function that achieves surgical Online Learning:

### The Slot Machine Mapping:
When the `new_ratings_df` is passed to PyTorch, the script isolates the unique Supabase `UUIDs`. It checks if they exist in the model's physical mapping file. If they don't, it systematically deletes `"__GUEST_X__"` from the dictionary file, replaces it with the real UUID string, and safely locks that newly registered user into their own dedicated row of the deep learning architecture forever.

### 4-to-1 Mini Batch Sampling:
For the 20 or 50 micro interactions pulled from the Supabase Database, the function spins up identical `Vector Collision Avoidance` processing loops. For every positive movie rating the user gave, it forcefully assigns 4 random unwatched negative mappings.

### Why We Train on the Whole Table Every Time:
As the Supabase `ratings` database grows, we deliberately re-extract the entire history of the UUID when fine-tuning instead of just isolating the "most recently clicked" movies. This intentionally protects the PyTorch architecture from a severe biological AI regression known as **Catastrophic Forgetting**, where a neural network severely overwrites historical user tastes in favor of their most recent micro-batch inputs. Training all vectors dynamically perfectly balances the gradients spanning the user's total personality.

## 5. Deployment Overview
This pipeline guarantees the recommendations system can perpetually organically scale without ever relying on heavy Cloud GPU hardware.
- Time to extract, construct matrices, fine-tune for 25 epochs natively using Mac Silicon, and statically deploy the binaries sequentially onto the filesystem: **~4.50 seconds**.

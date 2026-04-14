# Kaggle NCF Training & ALS Evaluation Log

## 1. The Strategy: Fixing Kaggle RAM Crashes
We identified that Kaggle's 30GB RAM limit crashed when evaluating **4 negatives per positive** out of a 20M dataset (producing ~159M rows).
The fix was to reduce negative sampling from 4 → 2, saving ~1GB of tensor memory and reducing training time per epoch to ~40m.
We also bumped epochs from 5 → 6 to recover any accuracy dropped from using faster/smaller tensors, and added per-epoch model checkpointing so no progress is ever lost if memory does peak.

## 2. Implementing the Implicit ALS Model
We successfully migrated from `scikit-surprise` SVD to the high-performance `implicit` ALS algorithm. 
- Time dropped from 45 mins → 90 seconds.
- Storage size dropped from 80MB → 15MB.
- **Critical Code Fix applied:** In implicit 0.7+, if an item-user matrix is provided to `fit()`, the library outputs the math with swapped vector arrays. We safely fixed `CollaborativeRecommender.recommend()` so that `user_vector = self.model.item_factors` preventing IndexError crashes for our 200k users.

## 3. The Incredible Discovery: Timestamp Data Leakage
When evaluating the Kaggle NCF weights downloaded to the local machine, the output was completely flat (`[0.000, 0.000...]`). Debugging revealed:
1. When passed timestamps of `0.0`, the model predicted 0%.
2. When passed timestamps of `1.0`, the model predicted 100%.

**What Happened? (Shortcut Learning)**
During Kaggle training data prep:
- Positive samples were given their real scaled timestamps (`[0-1]`).
- The 2 negative samples were just assigned `0.0`.
Because the Neural Network is incredibly smart, it completely ignored the complex User IDs, Target Movies, Genres, and Tags. It just deduced: *“If timestamp is 0, it’s negative. If timestamp > 0, it’s positive.”* 

Because my evaluation script provided an arbitrary timestamp of `1.0` to the 99 random negative items and the 1 positive target, the model thought ALL 100 items were positive targets, predicting exactly `100%` for all of them. Python's stable array sort then artificially preserved the target item at rank #1, resulting in a theoretically "perfect" (but utterly fake) HR@10 of `1.000`.

## 4. The Action Plan for Retraining
Because the model bypassed learning user taste, we must **retrain the NCF model on Kaggle**.

**The Fix:**
We must **DROP the timestamp feature completely** from the Kaggle NCF implementation. 
Neural Collaborative Filtering architectures rarely benefit from timestamps as static features anyway, and removing it completely removes the risk of data leakage.

Once the model is retrained without timestamps, it will be forced to rely on the actual User/Item Embeddings, the genres, and the tags to learn true user taste.

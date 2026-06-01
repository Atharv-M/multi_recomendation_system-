# 🔬 Recommendation Systems: Comparative Evaluation & Hybridization Research Report

**Prepared for:** Academic & Presentation Review  
**Dataset:** MovieLens (Post-2000 Filtered, 31.7M+ training ratings)  
**Evaluation Protocol:** Leave-One-Out (LOO) with 1 target vs. 99 random negatives  
**Evaluation Sample:** 50,000 sampled users (resulting in ~150,000 total evaluations across states)  

---

## 1. Introduction & Research Objectives

In recommendation systems engineering, no single algorithm is optimal across all user states. Collaborative filtering excels with dense user histories but fails under cold-start or sparse conditions. Content-based filtering is highly effective for niche tastes but suffers from over-specialization and lacks serendipity. 

This research project conducts a comparative benchmark of **four distinct algorithms** and evaluates a **Unified Adaptive Hybrid model** designed to combine their strengths dynamically based on user history density. 

The models benchmarked are:
1. **Popularity-Based Recommender** (Global Baseline)
2. **Content-Based Recommender** (TF-IDF + Cosine Similarity on genres and tags)
3. **Collaborative Filtering** (Implicit ALS Matrix Factorization)
4. **Neural Collaborative Filtering** (PyTorch NeuMF - Deep Learning)
5. **Hybrid (Adaptive) Recommender** (Unified Score Fusion)

---

## 2. Evaluation Protocol & Methodology

### 2.1 Leave-One-Out (LOO) Protocol
For each user, the **most recent rating (chronologically)** is held out as the **validation set** (`val_ratings.csv`). All prior ratings form the **training set** (`train_ratings.csv`). 

During evaluation:
1. A user is selected.
2. The user's held-out validation movie is designated as the **target item**.
3. **99 negative items** (movies the user has never interacted with in either train or val) are randomly sampled from the catalog.
4. The models are presented with this list of 100 candidate movies, rank them in descending order of preference, and output a Top-10 recommended list.

### 2.2 Metrics
* **Hit Rate at 10 (HR@10)**: Measures the fraction of users for whom the target movie is successfully recommended within the Top-10 list.
  $$\text{HR@10} = \begin{cases} 1 & \text{if } m_{\text{target}} \in \text{Top-10} \\ 0 & \text{otherwise} \end{cases}$$
* **Normalized Discounted Cumulative Gain at 10 (NDCG@10)**: Evaluates the ranking quality by rewarding the model for placing the target item closer to the #1 spot.
  $$\text{NDCG@10} = \frac{\log(2)}{\log(\text{rank} + 2)}$$
  *(where $\text{rank}$ is the 0-indexed position of the target item in the Top-10 list, returning 0 if it is not in the Top-10).*

### 2.3 Eliminating Positional Bias
To ensure rigorous evaluation, the 100 candidate items are **randomly shuffled** before scoring. This prevents Python's stable sorting from artificially inflating the metrics of models that output tie scores (e.g., scoring all items as `0.0` under Cold Start).

---

## 3. Sparsity Simulation Strategy

The MovieLens dataset naturally filters out users with fewer than 20 ratings. To benchmark our system across the entire user lifecycle, we **simulated** Cold Start and Very Sparse user behaviors inside our evaluation loop by hiding parts of the training data:

1. **Cold Start (0 Ratings)**: Hides the user's entire training history. Evaluates how the models perform for brand-new signups.
2. **Very Sparse (1-5 Ratings)**: Hides all but a random subset of **3 ratings** from the user's training history. Evaluates performance after a user rates their first few movies.
3. **Natural State**: Uses the user's full training history. Users are partitioned into three natural categories based on training counts:
   * **Sparse (5-20)**: Users who have rated 5 to 20 movies (2,380 users in sample).
   * **Medium (20-50)**: Users who have rated 20 to 50 movies (16,462 users in sample).
   * **Power User (50+)**: Users who have rated 50+ movies (31,158 users in sample).

---

## 4. Model Architectures & Advanced Evaluation Techniques

### 4.1 Popularity-Based (Global Baseline)
Scores items using an IMDb-style Bayesian Weighted Average:
$$\text{Score} = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$$
*(where $v$ is the movie's rating count, $m$ is the 90th percentile threshold of ratings, $R$ is the movie's average rating, and $C$ is the global average rating).*

### 4.2 Content-Based Recommender (TF-IDF + Cosine Similarity)
Constructs a user profile vector from their training history. Candidate items are scored by calculating their cosine similarity against the profile vector.
* **Scoring Formula**:
  $$\text{Score}(u, j) = \max_{i \in \text{history}(u)} \text{CosineSimilarity}(i, j)$$
* **Optimization**: To handle 100 candidates against $N$ history movies in milliseconds, we normalize the entire feature matrix ($V$) once and perform a single matrix multiplication:
  $$\text{Similarity Matrix} = V_{\text{candidates}} \cdot V_{\text{history}}^T$$
  The candidate score is the maximum value along the row.

### 4.3 Collaborative Filtering (Implicit ALS)
Learns 50-dimensional latent user factors ($U$) and item factors ($V$) using Alternating Least Squares on implicit feedback.
* **Simulation Technique**: For Very Sparse users (3 ratings), we cannot use the user's pre-computed latent factor vector (since it contains full history). Instead, we construct a temporary CSR matrix containing only the 3 ratings and dynamically compute a new user factor vector on the fly using:
  $$\mathbf{u}_{\text{recalc}} = \text{als\_model.recalculate\_item}(0, \mathbf{r}_{\text{sparse}})$$
  Candidates are scored by taking the dot product $\mathbf{u}_{\text{recalc}} \cdot \mathbf{v}_j$.

### 4.4 Neural Collaborative Filtering (NCF / NeuMF)
Pytorch-based neural network combining Generalized Matrix Factorization (GMF) and Deep Multi-Layer Perceptrons (MLP) with genre and tag embedding bag layers.
* **Fallback Strategy**: For Cold Start and simulated Very Sparse users, the NCF model falls back to the Popularity-based ranking, representing real-world behavior before a live fine-tuning background job is executed.

---

## 5. Unified Adaptive Hybrid Recommender Design

To combine the predictions of all four models, we implement a **Normalized Weighted Ensemble Recommender**:

### 5.1 Min-Max Score Normalization
Since model scores are on different scales, candidate scores for each model are normalized to $[0, 1]$ before fusion:
$$S^{\text{norm}}_{\text{model}}(j) = \frac{S_{\text{model}}(j) - \min(S_{\text{model}})}{\max(S_{\text{model}}) - \min(S_{\text{model}}) + 10^{-9}}$$

### 5.2 Score Fusion & Dynamic Weights
The hybrid score is computed as:
$$S_{\text{Hybrid}}(u, j) = w_{\text{pop}} \cdot S^{\text{norm}}_{\text{pop}} + w_{\text{content}} \cdot S^{\text{norm}}_{\text{content}} + w_{\text{ALS}} \cdot S^{\text{norm}}_{\text{ALS}} + w_{\text{NCF}} \cdot S^{\text{norm}}_{\text{NCF}}$$

The weights ($w$) adapt dynamically depending on the user's category:

| User Category | Popularity Weight ($w_{\text{pop}}$) | Content Weight ($w_{\text{content}}$) | ALS Weight ($w_{\text{ALS}}$) | NCF Weight ($w_{\text{NCF}}$) |
|---|---|---|---|---|
| **Cold Start (0)** | 1.0 | 0.0 | 0.0 | 0.0 |
| **Very Sparse (1-5)** | 0.1 | 0.7 | 0.2 | 0.0 |
| **Sparse / Medium / Power** | 0.1 | 0.2 | 0.3 | 0.4 |

---

## 6. Final Comparative Results (50,000 Users)

The comparative benchmark was executed on a random sample of 50,000 users.

| User Type | Ratings | Model | Sample Size | HR@10 | NDCG@10 |
|---|---|---|---|---|---|
| **Cold Start (0)** | 0 (Simulated) | Popularity | 50000 | 0.8356 | 0.5643 |
| **Cold Start (0)** | 0 (Simulated) | Content-Based | 50000 | 0.0985 | 0.0455 |
| **Cold Start (0)** | 0 (Simulated) | Collaborative (ALS) | 50000 | 0.8356 | 0.5643 |
| **Cold Start (0)** | 0 (Simulated) | NCF (NeuMF) | 50000 | 0.8356 | 0.5643 |
| **Cold Start (0)** | 0 (Simulated) | **Hybrid (Adaptive)** | 50000 | 0.8356 | 0.5643 |
| **Very Sparse (1-5)** | 3 (Simulated) | Popularity | 50000 | 0.8356 | 0.5643 |
| **Very Sparse (1-5)** | 3 (Simulated) | Content-Based | 50000 | 0.9518 | 0.6960 |
| **Very Sparse (1-5)** | 3 (Simulated) | Collaborative (ALS) | 50000 | 0.7143 | 0.5901 |
| **Very Sparse (1-5)** | 3 (Simulated) | NCF (NeuMF) | 50000 | 0.8356 | 0.5643 |
| **Very Sparse (1-5)** | 3 (Simulated) | **Hybrid (Adaptive)** | 50000 | **0.9120** | **0.6765** |
| **Sparse (5-20)** | 5-20 (Natural) | Popularity | 2380 | 0.8643 | 0.5912 |
| **Sparse (5-20)** | 5-20 (Natural) | Content-Based | 2380 | 0.9718 | 0.7114 |
| **Sparse (5-20)** | 5-20 (Natural) | Collaborative (ALS) | 2380 | 0.9319 | 0.8391 |
| **Sparse (5-20)** | 5-20 (Natural) | NCF (NeuMF) | 2380 | 0.9866 | 0.8426 |
| **Sparse (5-20)** | 5-20 (Natural) | **Hybrid (Adaptive)** | 2380 | **0.9874** | **0.8779** |
| **Medium (20-50)** | 20-50 (Natural) | Popularity | 16462 | 0.8540 | 0.5931 |
| **Medium (20-50)** | 20-50 (Natural) | Content-Based | 16462 | 0.9700 | 0.7129 |
| **Medium (20-50)** | 20-50 (Natural) | Collaborative (ALS) | 16462 | 0.9439 | 0.8569 |
| **Medium (20-50)** | 20-50 (Natural) | NCF (NeuMF) | 16462 | 0.9895 | 0.8615 |
| **Medium (20-50)** | 20-50 (Natural) | **Hybrid (Adaptive)** | 16462 | **0.9870** | **0.8879** |
| **Power User (50+)** | 50+ (Natural) | Popularity | 31158 | 0.8236 | 0.5470 |
| **Power User (50+)** | 50+ (Natural) | Content-Based | 31158 | 0.9367 | 0.6377 |
| **Power User (50+)** | 50+ (Natural) | Collaborative (ALS) | 31158 | 0.9579 | 0.8142 |
| **Power User (50+)** | 50+ (Natural) | NCF (NeuMF) | 31158 | 0.9807 | 0.7988 |
| **Power User (50+)** | 50+ (Natural) | **Hybrid (Adaptive)** | 31158 | **0.9792** | **0.8282** |

---

## 7. Key Research Insights & Discussion

### 7.1 The Hybrid Breakthrough
The Hybrid (Adaptive) model achieved the highest ranking quality (**NDCG@10**) across all active user categories, outperforming the single-algorithm SOTA baseline (NCF / NeuMF) by **3.5%** in the Sparse group, **2.6%** in the Medium group, and **2.9%** in the Power User group. This confirms that collaborative signal ensembling with genre and tag metadata similarity reduces individual model noise and yields superior, highly personalized ranking.

### 7.2 Sparsity Isolation & Performance
* **Cold Start Baseline**: Shuffling candidates successfully isolated the cold-start behavior. Content-Based fell to its true random baseline (`HR@10 = 0.0985` or ~10%), while ALS and NCF correctly fell back to Popularity (`0.8356`).
* **Content-Based Dominance in Sparse Settings**: For simulated Very Sparse users (3 ratings), the Content-Based model achieves `0.9518` HR and `0.6960` NDCG, outperforming ALS's mathematical recalculation (`0.7143` HR). This illustrates why content-based profiles are essential during user onboarding before collaborative graphs converge.
* **Collaboration & Scale**: As the ratings count increases from Sparse to Power User, NCF and ALS performance dominates Content-Based, which falls off from `0.9718` to `0.9367` HR. Collaborative filters excel as interaction matrices densify, capturing latent stylistic relationships that metadata-only vectors miss.

---
*Report compiled successfully. Saved to project root as EVALUATION_REPORT.md.*

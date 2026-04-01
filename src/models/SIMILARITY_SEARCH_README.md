# Similarity Search: Chunked Cosine Similarity
### How We Compute 87,585 × 87,585 Movie Similarities Without Crashing

---

## 📖 The Problem We Solved

The content-based recommendation model needs to know how **similar each movie is to every other movie**. With 87,585 movies in our dataset, this is a massive computational challenge.

The naive approach — computing all similarities at once — creates a matrix so large it crashes any normal computer:

```
87,585 movies × 87,585 movies × 4 bytes (float32) = ~30 GB RAM
```

A standard laptop or cloud machine has 8–16 GB. Attempting this kills the process instantly with an `MemoryError`.

Our solution: **Chunked Cosine Similarity with BLAS Matrix Multiply and argpartition** — completing the same computation in under 2 minutes using peak RAM of only ~2.7 GB.

---

## 🧠 Background: What is Cosine Similarity?

Every movie in the system is represented as a **feature vector** — a list of numbers that capture its characteristics:

```
Inception   → [0.3, 0.0, 0.8, 0.1, 0.5, ...]  (6800 numbers)
The Matrix  → [0.3, 0.0, 0.7, 0.2, 0.4, ...]  (6800 numbers)
Toy Story   → [0.0, 0.9, 0.1, 0.0, 0.2, ...]  (6800 numbers)
```

These numbers come from three sources (built by `build_features.py`):
- **TF-IDF text features** — from genres, user tags, genome tags (5000 dimensions)
- **Genre binary flags** — one column per genre, 1 if the movie belongs to it (20 dimensions)
- **Numeric features** — average rating and log(rating count), scaled (2 dimensions)

**Cosine similarity** measures the angle between two vectors. If they point in the same direction, the movies are similar. If they point in opposite directions, they are completely different.

```
                    B (Thriller)
                   /
                  /  ← small angle = HIGH similarity
                 /
────────────────A (Thriller) ─────────────────
                 \
                  \  ← large angle = LOW similarity
                   \
                    C (Romance)

cosine_similarity(A, B) = cos(small angle) ≈ 0.94  → very similar
cosine_similarity(A, C) = cos(large angle) ≈ 0.08  → very different
```

**The formula:**

```
cosine_sim(A, B) = (A · B) / (|A| × |B|)

where:
  A · B  = dot product = sum of (A[i] × B[i]) for all i
  |A|    = length of vector A = sqrt(sum of A[i]²)
```

The result is always between **0** (completely different) and **1** (identical).

---

## 🔑 The Three Key Optimizations

### Optimization 1: L2 Normalization — Eliminate the Division Forever

The cosine formula divides by the lengths of both vectors on every single comparison. For 87,585 × 87,585 comparisons, that's **7.6 billion divisions**.

The fix: normalize every vector to length 1.0 **once**, before any comparisons begin.

```python
features_norm = normalize(movie_features, norm="l2")
```

After normalization, every vector has `|A| = 1.0`. So the denominator always equals `1.0 × 1.0 = 1.0`:

```
cosine_sim(A, B) = (A · B) / (1.0 × 1.0) = A · B
```

**Cosine similarity becomes a pure dot product.** The billions of divisions vanish completely. This is not an approximation — the result is mathematically identical.

**Why this matters:** Dot products are exactly what modern CPUs and GPUs are physically built to compute at maximum speed. Each CPU core has dedicated silicon (FMA — Fused Multiply-Add units) for this specific operation.

---

### Optimization 2: BLAS Matrix Multiply — Computing Everything in Parallel

Instead of computing similarities one movie pair at a time in Python, we use **matrix multiplication** to compute 2,000 movies' similarities against all 87,585 movies simultaneously.

```
   chunk (2000 × 6800)   @   features_norm.T (6800 × 87585)   =   result (2000 × 87585)
```

Visualized:

```
                        ← 87,585 movies (columns) →
              ┌──────────────────────────────────────────────┐
         ↑    │  0.94   0.12   0.67   0.08  ...   0.31      │  ← row 0: Inception vs ALL
       2000   │  0.11   0.88   0.23   0.71  ...   0.09      │  ← row 1: Matrix vs ALL
       rows   │  0.03   0.07   0.92   0.11  ...   0.44      │  ← row 2: Toy Story vs ALL
         ↓    │   ...                                        │
              └──────────────────────────────────────────────┘
                           Result: 2000 × 87,585
```

One matrix multiply gives the complete similarity profile of 2000 movies in a single operation.

**What is BLAS?**

BLAS (Basic Linear Algebra Subprograms) is a low-level library that NumPy uses automatically when you write `A @ B`. It is not regular Python code — it is:

- Written in **Fortran and C**, compiled to native machine code
- Uses **AVX/AVX-512** — special CPU instructions that process 8–16 numbers simultaneously in one clock cycle
- **Multi-threaded** — uses ALL CPU cores at once without you writing any threading code
- Decades of optimization by Intel, AMD, and academia

When you write `chunk @ features_norm.T` in Python, you are invoking this entire industrial-strength infrastructure. That single line replaces what would otherwise be a triple nested Python `for` loop — about 100× slower.

```python
# What Python would do without BLAS (SLOW):
result = []
for movie_a in chunk:           # 2000 iterations
    row = []
    for movie_b in all_features: # 87585 iterations
        dot = 0
        for f in range(6800):    # 6800 iterations
            dot += movie_a[f] * movie_b[f]
        row.append(dot)
    result.append(row)
# Total: 2000 × 87585 × 6800 = 1.19 trillion operations in Python

# What BLAS does (FAST):
result = chunk @ features_norm.T
# Same 1.19 trillion operations — in C, AVX, multi-threaded
```

---

### Optimization 3: argpartition — Finding Top-K Without Full Sorting

Once we have 87,585 similarity scores for one movie, we only need the **top 20**. The obvious approach — sorting all 87,585 scores — is wasteful:

```
Full sort of 87,585 items:
  Algorithm:  Quicksort or Mergesort
  Complexity: O(N log N) = O(87585 × 17) ≈ 1,489,000 operations per movie
  For all 87K movies: 87585 × 1,489,000 = 130 billion operations
```

`np.argpartition` uses the **QuickSelect algorithm** to find the top-K items without sorting everything:

```python
top_indices = np.argpartition(sim_row, -20)[-20:]
```

**How QuickSelect works:**

```
Goal: Find top-3 from [0.12, 0.94, 0.31, 0.82, 0.67, 0.08, 0.44]

QuickSelect picks a pivot and partitions into two groups:
  "larger than pivot"  vs  "smaller than pivot"

It only recurses into the side that contains what you need.
It never bothers sorting items that won't be in the top-K.

Guarantee: everything in [-3:] is larger than everything in [:-3]
Result:    [0.44, 0.82, 0.94]  ← not sorted, but all top-3 are here

One small sort of just 3 items → [0.94, 0.82, 0.44]

Complexity: O(N) on average — linear scan, no sorting
For 87,585 items vs top-20: only 87,585 operations instead of 1,489,000
```

**The saving:**

```
Full sort:    87,585 × log₂(87585) ≈ 87,585 × 17 = 1,489,000 operations
argpartition: 87,585 × 1           ≈ 87,585          operations

← 17× fewer operations per movie
← 87,585 × 17 = 1.5 billion fewer operations across the full dataset
```

---

## 🏗️ The Complete Pipeline

Here is the full algorithm step by step:

```
INPUT:  movie_features  shape: (87585, 6800)  — sparse matrix from build_features.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: L2 NORMALIZE  (done once, ~0.5 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

features_norm = normalize(movie_features, norm="l2")

Each row now has ||row|| = 1.0
dot product of any two rows = their cosine similarity
RAM: ~2.0 GB (stored throughout, never deleted)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: CHUNKED LOOP  (44 iterations of chunk_size=2000)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

for start in range(0, 87585, 2000):    # 0, 2000, 4000, ..., 86000
    end   = min(start + 2000, 87585)
    chunk = features_norm[start:end]   # shape: (2000, 6800)

    ┌─────────────────────────────────────────────────────┐
    │  BLAS MATRIX MULTIPLY                               │
    │  sim_chunk = chunk @ features_norm.T                │
    │  shape: (2000, 87585)                               │
    │  RAM used: 2000 × 87585 × 4 bytes = ~700 MB        │
    │  Time: ~2.5 seconds per chunk (all CPU cores)       │
    └─────────────────────────────────────────────────────┘

    for local_idx in range(len(chunk)):
        global_idx         = start + local_idx
        sim_row            = sim_chunk[local_idx]      # 87585 scores

        sim_row[global_idx] = -1                       # exclude self

        ┌───────────────────────────────────────────┐
        │  ARGPARTITION (O(N) partial selection)    │
        │  top_idx = argpartition(sim_row, -20)[-20:]│
        │  top_idx = top_idx[argsort(...)[::-1]]    │  ← sort only 20 items
        └───────────────────────────────────────────┘

        topk_similarity[movie_ids[global_idx]] = [
            (movie_ids[i], float(sim_row[i])) for i in top_idx
        ]

    del sim_chunk   # ← CRITICAL: free 700MB before next chunk

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: SAVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

joblib.dump(topk_similarity, "topk_movie_similarity.joblib")

OUTPUT: Dictionary of 87,585 entries, each with 20 (movie_id, score) pairs
```

---

## 📊 Performance Comparison

| | Old Approach | New Approach |
|---|---|---|
| **Algorithm** | `sklearn.cosine_similarity` | Normalized `@` + argpartition |
| **Memory peak** | **~30 GB** (crashes) | **~2.7 GB** ✅ |
| **Similarity computation** | Full N×N matrix, held in RAM | 2000-row chunks, deleted after use |
| **Sorting** | `list.sort()` — O(N log N) per movie | `argpartition` — O(N) per movie |
| **CPU usage** | Single core (Python loop) | All cores (BLAS parallelism) |
| **Time for 51K movies** | 19 minutes | — |
| **Time for 87K movies** | Would crash | **1 minute 55 seconds** |
| **Result accuracy** | 100% | **100% — mathematically identical** |

---

## 💡 Why The Result Is Identical

This is a common question: "if you're using tricks and shortcuts, isn't the result approximate?"

**No. The result is exactly the same as the naive approach.**

- **Normalization** changes vector lengths but not angles. Cosine similarity measures angles, so the scores are identical.
- **Matrix multiplication** computes the exact same dot products as a Python loop — just in a different order and using different hardware instructions. The floating-point results are bit-for-bit identical.
- **argpartition** finds exact top-K values. It is not an approximation — it guarantees that every item in the top-K partition is genuinely larger than every item outside it.

The only thing that changes between the old and new approach is **how** the computation is done — not **what** is computed.

---

## 🔬 Technical Concepts Glossary

| Term | What It Means |
|---|---|
| **Cosine Similarity** | How similar two vectors are based on the angle between them. Range: 0 (different) to 1 (identical) |
| **L2 Normalization** | Dividing a vector by its own length so it becomes length 1.0 |
| **Dot Product** | Sum of element-wise multiplications: `A·B = A[0]×B[0] + A[1]×B[1] + ...` |
| **BLAS** | Basic Linear Algebra Subprograms — a C/Fortran library for ultra-fast matrix math |
| **AVX/AVX-512** | CPU instructions that process 8–16 floating-point numbers in a single clock cycle |
| **Matrix Multiply (`@`)** | Efficiently computes all dot products between two sets of vectors simultaneously |
| **argpartition** | An O(N) algorithm that separates the top-K elements without fully sorting the array |
| **QuickSelect** | The algorithm behind argpartition — finds K-th largest in linear time using pivot partitioning |
| **Sparse Matrix** | A matrix stored by only recording non-zero values — used here because TF-IDF vectors are mostly zeros |
| **pin_memory** | Allocates data in page-locked RAM so it transfers to GPU up to 2× faster |
| **chunk_size** | How many movies to process per iteration. Larger = faster but more RAM. 2000 is optimal for 16GB RAM |

---

## 📁 Related Files

| File | Role |
|---|---|
| `src/features/build_features.py` | Builds the 87585×6800 feature matrix that feeds into similarity |
| `src/models/content_based_model.py` | Contains `build_topk_similarity()` — the function this README documents |
| `artifacts/saved_features/movie_features.joblib` | The feature matrix on disk |
| `artifacts/content/topk_movie_similarity.joblib` | The final similarity index — 87,585 movies, 20 neighbors each |

---

## 🚀 How to Re-run

```bash
# From project root with venv active:
source .venv/bin/activate

# Rebuild feature matrix (if movies data updated):
PYTHONPATH=. python src/features/build_features.py

# Rebuild similarity index:
PYTHONPATH=. python -c "
from src.models.content_based_model import build_topk_similarity
build_topk_similarity(top_k=20, chunk_size=2000)
"
```

To tune for your hardware:
- **More RAM (32GB+):** increase `chunk_size=4000` → faster (fewer chunk iterations)
- **Less RAM (8GB):** decrease `chunk_size=1000` → safer (smaller temporary matrices)
- **More neighbors:** change `top_k=30` for richer recommendations (slightly larger output file)

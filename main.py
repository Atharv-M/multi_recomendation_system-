# from src.models.popularity_model import build_popularity_model

# model = build_popularity_model()
# model.fit()

# recommendations = model.recommend(top_k=10)
# print(recommendations)

# from src.models.content_based_model import ContentBasedRecommender

# model = ContentBasedRecommender()
# model.fit()

# # Example: recommendations similar to movieId = 1
# model.recommend(movie_id=1, top_k=10)

# from src.models.collaborative_filtering import CollaborativeFilter

# cf_model = CollaborativeFilter()
# cf_model.fit()

# cf_model.recommend(user_id=1, top_k=10)

from src.models.hybrid_recomender import HybridRecommender


def main():
    hybrid = HybridRecommender()
    hybrid.fit()

    print("\n--- Cold Start (No user, no movie) ---")
    print(hybrid.recommend(top_k=5))

    print("\n--- Content-Based (Movie Similarity) ---")
    print(hybrid.recommend(movie_id=1, top_k=5))

    print("\n--- Personalized (Collaborative Filtering) ---")
    print(hybrid.recommend(user_id=1, top_k=5))

if __name__ == "__main__":
    main()


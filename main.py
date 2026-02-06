# from src.models.popularity_model import build_popularity_model

# model = build_popularity_model()
# model.fit()

# recommendations = model.recommend(top_k=10)
# print(recommendations)

from src.models.content_based_model import ContentBasedRecommender

model = ContentBasedRecommender()
model.fit()

# Example: recommendations similar to movieId = 1
model.recommend(movie_id=1, top_k=10)

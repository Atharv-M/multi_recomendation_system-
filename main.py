from src.models.popularity_model import build_popularity_model

model = build_popularity_model()
model.fit()

model.recommend(top_k=10)

import logging 
import pandas as pd
from src.models.popularity_model import PopularityRecommender
from src.models.collaborative_filtering import CollaborativeRecommender
from src.models.content_based_model import ContentBasedRecommender
from src.models.ncf import NeuralCollaborativeRecommender


logger = logging.getLogger(__name__)

class HybridRecommender():
    def __init__(self,alpha=0.7):
        self.alpha = alpha
        self.popularity_model =PopularityRecommender()
        self.content_model = ContentBasedRecommender()  
        self.cf_model = CollaborativeRecommender()
        self.ncf_model = NeuralCollaborativeRecommender()

        self._is_fitted = False
    
    def fit(self):
        logger.info("Starting Training of Hybrid Model")
        self.popularity_model.fit()
        self.content_model.fit()
        self.cf_model.fit()
        self.ncf_model.fit()
        self._is_fitted = True
        logger.info("Hybrid model Trained Succefully")
    
    def save(self):
        self.popularity_model.save()
        self.content_model.save()
        self.cf_model.save()
        self.ncf_model.save()
        logger.info("Hybrid model saved successfully")
    
    def load(self):
        self.popularity_model.load()
        self.content_model.load()
        self.cf_model.load()
        self.ncf_model.load()
        self._is_fitted = True
        logger.info("Hybrid model loaded successfully")
    
    def recommend_dual(self, user_id=None, top_k=10):
        if not self._is_fitted:
            raise ValueError("Model is Not Trained. Call fit() First")
        
        logger.info(f"Computing Dual Intelligence Vector Comparison for User {user_id}")
        svd_df = self.cf_model.recommend(user_id=user_id, top_k=top_k)
        ncf_df = self.ncf_model.recommend(user_id=user_id, top_k=top_k)
        return svd_df, ncf_df

    def recommend(
        self,
        user_id=None,
        movie_id=None,
        top_k=10
    ):
        if not self._is_fitted:
            raise ValueError("Model is Not Trained . Call fit() First ")
        
        ## Movie Context Detected 
        if movie_id is not None:
            logger.info("Movie Context Detected, Using Content Based Model")
            return self.content_model.recommend(movie_id,top_k)
    
        ## Active User Detected 
        if user_id is not  None:
            logger.info("Active User Detected, Using Collaborative Filtering")
            return self.cf_model.recommend(user_id,top_k)
        
        ## Cold Start Movie Detected 
        logger.info("Cold-start movie detected. Using popularity model ")
        return self.popularity_model.recommend(top_k)
import logging
import pandas as pd
from src.models.hybrid_recomender import HybridRecommender
from app.config import LINKS_DATASET_PATH

logger = logging.getLogger(__name__)


hybrid_model = None
links_df = None

def get_hybrid_model():
    global hybrid_model
    if hybrid_model is None:
        logger.info("Initializing Hybrid Model...")
        model = HybridRecommender()
        try:
            model.load()
            hybrid_model = model
            logger.info("Hybrid Model Loaded Successfully")
        except Exception as e:
            logger.error(f"Failed to load Hybrid Model: {e}")
            raise e
    return hybrid_model

def get_links_df():
    global links_df
    if links_df is None:
        links_df = pd.read_csv(LINKS_DATASET_PATH)
        logger.info("Links Dataset Loaded Successfully")
    return links_df
    
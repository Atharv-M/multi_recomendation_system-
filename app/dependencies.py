import logging
import pandas as pd
import joblib
from src.models.hybrid_recomender import HybridRecommender
from src.config import LINKS_PKL_PATH

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

def reload_hybrid_model():
    """
    Clears the in-memory model cache so the next call to get_hybrid_model()
    reloads from the latest weights on disk.

    Call this after finetune_all.py completes to make new weights live
    without restarting the entire server.
    """
    global hybrid_model
    hybrid_model = None
    logger.info("Hybrid model cache cleared — will reload from disk on next request.")

def get_links_df():
    global links_df
    if links_df is None:
        links_df = joblib.load(LINKS_PKL_PATH)
        logger.info("Links Dataset Loaded Successfully")
    return links_df
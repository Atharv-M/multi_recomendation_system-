import logging
from src.models.hybrid_recomender import HybridRecommender

logger = logging.getLogger(__name__)


hybrid_model =None

def get_hybrid_model():
    global hybrid_model
    if hybrid_model is None:
        hybrid_model = HybridRecommender()
        hybrid_model.load()
        logger.info("Hybrid Model Loaded Successfully")
    return hybrid_model
    
import os
import pickle
import logging
import numpy as np

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Config

# Configure Logging
logger = logging.getLogger("SubcategoryClassification")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# Cache to store loaded models
_model_cache = {}

CATEGORY_MODEL_MAP = {
    'beauty': 'beauty_clusters_results.pkl',
    'comedy': 'comedy_clusters_results.pkl',
    'cooking': 'cooking_clusters_results.pkl',
    'education': 'education_clusters_results.pkl',
    'finance': 'finance_clusters_results.pkl',
    'fitness': 'fitness_clusters_results.pkl',
    'gaming': 'gaming_clusters_model_and_interpretations.pkl',
    'music': 'music_clusters_results.pkl',
    'news': 'news_clusters_results.pkl',
    'sports': 'sports_clusters_results.pkl',
    'technology': 'tech_clusters_results.pkl',
    'travel': 'travel_clusters_results.pkl'
}

def get_subcategory(category: str, video_embeddings: np.ndarray):
    print(f"[DEBUG] Inside get_subcategory with category: {category}")
    if not category:
        print("[DEBUG] Category is empty.")
        return None
        
    category_key = category.lower().strip()
    print(f"[DEBUG] category_key: {category_key}")
    
    # 1. Map to filename
    vocab_key = CATEGORY_MODEL_MAP.get(category_key)
    print(f"[DEBUG] vocab_key: {vocab_key}")
    if not vocab_key:
        print(f"[DEBUG] No mapping for {category_key}")
        logger.warning(f"No corresponding subcategory model file mapping for '{category_key}'")
        return None

    # 2. Check cache or load
    if category_key in _model_cache:
        model_data = _model_cache[category_key]
    else:
        model_path = os.path.join(Config.BASE_DIR, 'models', 'subcategory_models', vocab_key)
        
        if not os.path.exists(model_path):
            logger.warning(f"Subcategory model not found at {model_path}")
            return None
            
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # ✅ Fix the cluster centers dtype stored inside the model itself
            kmeans = model_data.get('kmeans_model')
            if kmeans is not None and hasattr(kmeans, 'cluster_centers_'):
                kmeans.cluster_centers_ = np.ascontiguousarray(
                    kmeans.cluster_centers_, dtype=np.float32
                )
                
            _model_cache[category_key] = model_data
            logger.info(f"Loaded subcategory model for '{category}'")
        except Exception as e:
            logger.error(f"Failed to load subcategory model for {category_key}: {e}")
            return None
            
    # 3. Process with model
    try:
        kmeans_model = model_data.get('kmeans_model')
        cluster_interpretations = model_data.get('cluster_interpretations')
        
        if not kmeans_model or not cluster_interpretations:
             logger.error(f"Invalid model structure for category {category_key}")
             return None
             
        # ✅ Ensure float32 AND C-contiguous memory layout before any reshaping
        video_embeddings = np.ascontiguousarray(video_embeddings, dtype=np.float32)
        
        if video_embeddings.ndim == 1:
            video_embeddings = video_embeddings.reshape(1, -1)
            
        predicted_label = kmeans_model.predict(video_embeddings)[0]
        
        # Get subcategory name and keywords
        subcategory_info = cluster_interpretations.get(predicted_label)
        
        if subcategory_info:
            if isinstance(subcategory_info, dict):
                return subcategory_info.get('name', f"Cluster {predicted_label}")
            elif isinstance(subcategory_info, str):
                return subcategory_info  # gaming model stores name directly as string
        return f"Cluster {predicted_label}"
            
    except Exception as e:
        logger.error(f"Error predicting subcategory for {category_key}: {e}")
        return None

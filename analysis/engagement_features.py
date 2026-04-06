import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, List, Dict, Any
import joblib
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Global model cache
_models_cache = {}

def _load_models():
    """Load all pre-trained models from disk (lazy loading with caching)."""
    global _models_cache
    if _models_cache:
        return _models_cache
    
    try:
        _models_cache["views_model"] = joblib.load(str(MODELS_DIR / "views_model.joblib"))
        _models_cache["likes_model"] = joblib.load(str(MODELS_DIR / "likes_model.joblib"))
        _models_cache["comments_model"] = joblib.load(str(MODELS_DIR / "comments_model.joblib"))
        _models_cache["clip_pca"] = joblib.load(str(MODELS_DIR / "pca_clip_model.joblib"))
        _models_cache["text_pca"] = joblib.load(str(MODELS_DIR / "pca_text_model.joblib"))
        print("Models and PCA objects loaded successfully from disk.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found in {MODELS_DIR}: {e}")
    except Exception as e:
        raise Exception(f"Error loading models: {e}")
    
    return _models_cache

def apply_pca_to_features(embeddings: Union[List[float], np.ndarray], pca_model: Any, prefix: str) -> Dict[str, float]:
    """
    Applies a pre-fitted PCA model to raw embeddings and returns a dictionary of named features.
    
    Args:
        embeddings: List or array of raw embeddings (e.g., 512 dims or 384 dims).
        pca_model: Pre-fitted PCA object.
        prefix: Prefix for feature names (e.g., 'clip_pca' or 'text_pca').
        
    Returns:
        Dictionary with features named {prefix}_0 to {prefix}_63.
    """
    emb_arr = np.array(embeddings).reshape(1, -1)
    pca_features = pca_model.transform(emb_arr)[0]
    
    return {f'{prefix}_{j}': float(val) for j, val in enumerate(pca_features)}

def predict_video_performance(
    clip_embeddings: Union[List[float], np.ndarray],
    text_embeddings: Union[List[float], np.ndarray],
    upload_time: Union[str, datetime],
    avg_views_recent: float,
    avg_likes_recent: float,
    avg_comments_recent: float,
    category: int,
    title_length: int,
    num_tags: int,
) -> Dict[str, Any]:
    """
    Predicts video performance metrics (views, likes, comments) and derived scores
    using pre-trained models and PCA transformations.
    
    Args:
        clip_embeddings: List or array of clip embeddings (typically 512 dims).
        text_embeddings: List or array of text embeddings (typically 384 or 768 dims).
        upload_time: ISO format string or datetime object.
        avg_views_recent: Recent average views.
        avg_likes_recent: Recent average likes.
        avg_comments_recent: Recent average comments.
        category: Integer ID for the category (0-11).
        title_length: Length of the video title.
        num_tags: Number of tags.
        
    Returns:
        Structured JSON-like dictionary with predictions and metrics.
    """
    # Load models from disk (cached after first load)
    models = _load_models()
    views_model = models["views_model"]
    likes_model = models["likes_model"]
    comments_model = models["comments_model"]
    clip_pca = models["clip_pca"]
    text_pca = models["text_pca"]
    
    # 1. Preprocess time
    if isinstance(upload_time, str):
        try:
            # Handle standard ISO formats, removing 'Z' if present
            upload_time = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
        except ValueError:
            # Fallback to current time if parsing fails
            upload_time = datetime.utcnow()
            
    upload_hour = upload_time.hour
    upload_day = upload_time.weekday()  # Monday is 0, Sunday is 6
    is_weekend = 1 if upload_day >= 5 else 0

    # Handle defaults for missing features
    avg_views_recent = float(avg_views_recent) if avg_views_recent is not None else 0.0
    avg_likes_recent = float(avg_likes_recent) if avg_likes_recent is not None else 0.0
    avg_comments_recent = float(avg_comments_recent) if avg_comments_recent is not None else 0.0
    title_length = int(title_length) if title_length is not None else 0
    num_tags = int(num_tags) if num_tags is not None else 0

    # Derive interaction ratios
    like_view_ratio = (avg_likes_recent / avg_views_recent) if avg_views_recent > 0 else 0.0
    comment_view_ratio = (avg_comments_recent / avg_views_recent) if avg_views_recent > 0 else 0.0

    # 2. Apply PCA via helper functions
    clip_pca_features = apply_pca_to_features(clip_embeddings, clip_pca, 'clip_pca')
    text_pca_features = apply_pca_to_features(text_embeddings, text_pca, 'text_pca')
    
    # 3. One-hot encode category (0 to 11)
    category_encoded = np.zeros(12)
    if 0 <= category < 12:
        category_encoded[category] = 1.0
        
    # 4. Construct final feature vector EXACTLY in order:
    features = {
        'avg_views_recent': avg_views_recent,
        'avg_likes_recent': avg_likes_recent,
        'avg_comments_recent': avg_comments_recent,
        'like_view_ratio': like_view_ratio,
        'comment_view_ratio': comment_view_ratio,
        'upload_hour': upload_hour,
        'upload_day': upload_day,
        'is_weekend': is_weekend,
        'title_length': title_length,
        'num_tags': num_tags
    }
    
    # Extend with PCA generated from helpers
    features.update(text_pca_features)
    features.update(clip_pca_features)
        
    for j in range(12):
        features[f'category_{j}'] = float(category_encoded[j])
    
    # 5. Convert to Pandas DataFrame (prevents LGBMRegressor warnings and strictly enforces name mapping)
    X = pd.DataFrame([features])
    
    # Predictions
    # Using [0] to extract the scalar prediction value from array-like output
    views_multiplier = float(views_model.predict(X)[0])
    likes_multiplier = float(likes_model.predict(X)[0])
    comment_ratio = float(comments_model.predict(X)[0])

    # -----------------------------
    # EXTENSION: Convert multipliers → actual predictions
    # -----------------------------
    avg_views_recent = max(0.0, avg_views_recent)
    avg_likes_recent = max(0.0, avg_likes_recent)
    avg_comments_recent = max(0.0, avg_comments_recent)

    pred_views = max(0.0, views_multiplier * avg_views_recent)
    pred_likes = max(0.0, likes_multiplier * avg_likes_recent)
    pred_comments = max(0.0, comment_ratio * pred_views)

    # -----------------------------
    # EXTENSION: R²-based uncertainty ranges
    # -----------------------------
    r2_views = 0.276
    r2_likes = 0.23
    r2_comments = 0.18

    views_range_pct = max(0.2, min(0.5, 1 - r2_views))
    likes_range_pct = max(0.15, min(0.4, 1 - r2_likes))
    comments_range_pct = max(0.3, min(0.6, 1 - r2_comments))

    views_lower = max(0.0, pred_views * (1 - views_range_pct))
    views_upper = max(0.0, pred_views * (1 + views_range_pct))

    likes_lower = max(0.0, pred_likes * (1 - likes_range_pct))
    likes_upper = max(0.0, pred_likes * (1 + likes_range_pct))

    comments_lower = max(0.0, pred_comments * (1 - comments_range_pct))
    comments_upper = max(0.0, pred_comments * (1 + comments_range_pct))
    
    # Derived metrics
    raw_completion_score = 0.7 * likes_multiplier + 0.3 * comment_ratio
    raw_share_score = 0.5 * views_multiplier + 0.3 * likes_multiplier + 0.2 * comment_ratio
    
    # Normalize both scores to range [0, 1] using arbitrary reasonable bounds for scaling
    # Assuming practical max for multiplier is around 3.0 and ratio around 0.1 for scaling purposes
    # Adjust these min/max scaling constants depending on your actual data distribution
    completion_score = min(max(raw_completion_score / 3.0, 0.0), 1.0)
    share_score = min(max(raw_share_score / 3.0, 0.0), 1.0)
    
    # Post-processing
    views_growth = (views_multiplier - 1.0) * 100.0
    
    # Define labels
    if views_multiplier > 1.3:
        views_level = "High"
    elif views_multiplier > 1.0:
        views_level = "Moderate"
    else:
        views_level = "Low"
        
    if likes_multiplier > 1.2:
        engagement_level = "High"
    elif likes_multiplier > 1.0:
        engagement_level = "Good"
    else:
        engagement_level = "Low"
        
    if comment_ratio > 0.02:
        discussion_level = "High"
    elif comment_ratio > 0.005:
        discussion_level = "Medium"
    else:
        discussion_level = "Low"

    # Return JSON structure
    return {
        "views": {
            "prediction": round(pred_views, 2),
            "lower_bound": round(views_lower, 2),
            "upper_bound": round(views_upper, 2),
            "multiplier": round(views_multiplier, 4),
            "growth_percent": round(views_growth, 2),
            "level": views_level
        },
        "likes": {
            "prediction": round(pred_likes, 2),
            "lower_bound": round(likes_lower, 2),
            "upper_bound": round(likes_upper, 2),
            "multiplier": round(likes_multiplier, 4),
            "engagement": engagement_level
        },
        "comments": {
            "prediction": round(pred_comments, 2),
            "lower_bound": round(comments_lower, 2),
            "upper_bound": round(comments_upper, 2),
            "ratio": round(comment_ratio, 4),
            "discussion": discussion_level
        },
        "completion": {
            "score": round(completion_score, 4)
        },
        "share": {
            "score": round(share_score, 4)
        }
    }
import os
import sys
import json
import logging
import joblib
import lightgbm  # Explicit import for unpickling
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config
try:
    from clip_embedder import ClipEmbedder
except ImportError:
    # Fallback if run from root
    try:
        sys.path.append(os.getcwd())
        from clip_embedder import ClipEmbedder
    except ImportError:
        logging.error("Could not import clip_embedder. Ensure it is in the project root.")

logger = logging.getLogger("ClipAnalysis")

# Category Mapping (Label -> ID)
LABEL_TO_ID = {
    'Beauty': 0, 'Comedy': 1, 'Cooking': 2, 'Education': 3, 'Finance': 4,
    'Fitness': 5, 'Gaming': 6, 'Music': 7, 'News': 8, 'Sports': 9,
    'Technology': 10, 'Travel': 11
}
# Create ID -> Label mapping for decoding predictions
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Global variables for caching model and embedder
_model = None
_embedder = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(Config.BASE_DIR, 'models', 'lightgbm_model.joblib')
        if os.path.exists(model_path):
            logger.info(f"Loading LightGBM model from {model_path}")
            try:
                _model = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.warning(f"Model not found at {model_path}")
    return _model

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Initializing CLIP Embedder...")
        try:
            _embedder = ClipEmbedder()
        except Exception as e:
            logger.error(f"Failed to initialize CLIP Embedder: {e}")
    return _embedder

def analyze_content(video_path, frame_folder):
    """
    Analyzes video content using CLIP embeddings and LightGBM model.
    Generates embedding from frames, normalizes it, and predicts class label.
    Results are cached in the frame folder.
    """
    # 1. Check cache
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Use distinct cache dir to survive frame cleanup
    cache_dir = os.path.join(Config.OUTPUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{video_name}_clip_analysis.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached_result = json.load(f)
            logger.info(f"Loaded cached CLIP analysis from {cache_path}")
            return cached_result
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    # 2. Get resources
    model = get_model()
    embedder = get_embedder()
    
    if model is None:
        return {"error": "LightGBM model not found or failed to load"}
    if embedder is None:
        return {"error": "CLIP Embedder failed to initialize"}

    # 3. Load frames directly from video (Sampling: Every 12th frame)
    # The original model was trained on every 12th frame, so we replicate this sampling.
    frames = list()
    SAMPLING_INTERVAL = 12
    RESIZE_DIM = (224, 224) # Resize early to save memory
    
    logger.info(f"Extracting frames from video (every {SAMPLING_INTERVAL}th frame)...")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             return {"error": "Could not open video for CLIP analysis"}
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit processing to first 120 seconds for niche analysis
        fps = cap.get(cv2.CAP_PROP_FPS)
        MAX_ANALYSIS_SECONDS = 120
        max_frame_limit = int(fps * MAX_ANALYSIS_SECONDS) if fps > 0 else total_frames
        
        while True:
            # Check duration limit
            if frame_idx > max_frame_limit:
                 logger.info(f"Reached {MAX_ANALYSIS_SECONDS}s limit ({frame_idx} frames) for niche analysis.")
                 break

            # We can optimize by grabbing frames and only retrieving (decoding) the 12th one.
            # ret = cap.grab()
            # if not ret: break
            # if frame_idx % SAMPLING_INTERVAL == 0:
            #     _, frame = cap.retrieve()
            #     ...
            # But grab() + retrieve() vs read() depends on codec.
            # For simplicity and reliability with most codecs, we use read() with conditional processing.
            
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % SAMPLING_INTERVAL == 0:
                # Resize to target dimension (e.g., 224x224) to save memory before accumulating
                # CLIP uses 224x224 but processor handles it. We resize for RAM efficiency.
                try:
                    frame_resized = cv2.resize(frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frames.append(img_rgb)
                except Exception as resize_err:
                     logger.warning(f"Failed to process frame {frame_idx}: {resize_err}")

            frame_idx += 1
            
        cap.release()
                
        if not frames:
            return {"error": "Failed to load any frames from video"}
            
        logger.info(f"Collected {len(frames)} frames for CLIP analysis.")
            
    except Exception as e:
        logger.error(f"Error loading frames: {e}")
        return {"error": f"Frame loading error: {str(e)}"}

    # 4. Generate & Normalize Embedding
    try:
        # Generate raw embedding (512,)
        embedding = embedder.get_embedding(frames) 
        if embedding is None:
             return {"error": "Failed to generate embedding"}
             
        # Reshape for normalization (sklearn expects 2D array [n_samples, n_features])
        embedding_2d = embedding.reshape(1, -1)
        
        # Normalize
        normalized_embedding = normalize(embedding_2d)
        
        # 5. Predict
        # LightGBM predict returns the predicted class (or probabilities)
        prediction = model.predict(normalized_embedding)
        label = prediction[0]
        
        # Convert numpy types to native python types for JSON serialization
        if hasattr(label, 'item'):
            label = int(label.item())
        else:
            label = int(label)

        result = {
            "predicted_class": label,
            "predicted_label": ID_TO_LABEL.get(label, "unknown"),
            "embedding_shape": list(embedding.shape)
        }
        
        # 6. Cache result
        try:
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Cached CLIP analysis to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
            
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

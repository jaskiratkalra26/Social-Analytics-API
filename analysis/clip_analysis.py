import os
import sys
import json
import logging
import joblib
import glob
import lightgbm  # Explicit import for unpickling
import cv2
import numpy as np
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

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

def preload_models():
    """
    Explicitly loads models into memory.
    Call this on application startup to avoid latency on first request.
    """
    logger.info("Preloading models...")
    get_model()
    get_embedder()
    logger.info("Models preloaded successfully.")

def analyze_content(video_path: str, frame_folder: str, frames: list = None):
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

    # 3. Load frames 
    processing_frames = list()
    RESIZE_DIM = (224, 224) # Resize early to save memory
    
    # OPTIMIZATION: Use memory frames if available
    if frames is not None and len(frames) > 0:
        logger.info(f"Using {len(frames)} in-memory frames for CLIP analysis...")
        for img in frames:
            try:
                frame_resized = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_AREA)
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                processing_frames.append(img_rgb)
            except Exception:
                pass
    elif frame_folder and os.path.exists(frame_folder):
        jpg_files = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")))
        if jpg_files:
            logger.info(f"Using {len(jpg_files)} pre-extracted frames from disk for CLIP analysis...")
            for fpath in jpg_files:
                try:
                    img = cv2.imread(fpath)
                    if img is not None:
                        frame_resized = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_AREA)
                        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        processing_frames.append(img_rgb)
                except Exception:
                    pass

    # If still no frames, fallback to video capture
    if not processing_frames:
         # Fallback to video capture
        SAMPLING_INTERVAL = 12
        logger.info(f"Extracting frames from video (every {SAMPLING_INTERVAL}th frame)...")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                pass
            else:
                frame_idx = 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                MAX_ANALYSIS_SECONDS = 120
                max_frame_limit = int(fps * MAX_ANALYSIS_SECONDS) if fps > 0 else total_frames
                
                while True:
                    if frame_idx > max_frame_limit: break
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if frame_idx % SAMPLING_INTERVAL == 0:
                         try:
                            frame_resized = cv2.resize(frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
                            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                            processing_frames.append(img_rgb)
                         except: pass
                    frame_idx += 1
                cap.release()
        except Exception: pass
         
    if not processing_frames:
        return {"error": "No frames extracted or processing failed"}
        
    avg_embedding = embedder.get_embedding(processing_frames)
    if avg_embedding is None:
         return {"error": "Failed to generate embedding from frames"}

    # Ensure 2D for model
    feature_vector = avg_embedding.reshape(1, -1)
    
    # Get Probability
    probs = model.predict_proba(feature_vector)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    
    results = []
    for idx in top_indices:
        label = ID_TO_LABEL.get(idx, "Unknown")
        score = float(probs[idx])
        results.append({"label": label, "score": score})
    
    final_output = {
        "predicted_label": results[0]["label"] if results else "Unknown",
        "video_embeddings": avg_embedding.tolist()  # list for JSON cache compatibility
    }
    
    # Cache
    try:
        with open(cache_path, 'w') as f:
            json.dump(final_output, f)
    except: pass
    
    return final_output

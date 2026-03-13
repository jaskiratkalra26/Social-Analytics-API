import os
import sys
import json
import logging
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config
from pipeline import video_loader
from pipeline import frame_extractor
from pipeline import scene_detector
from features import visual_features
from features import text_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_pacing(video_path: str, frame_folder: str = None, frames: list = None) -> dict:
    """
    Analyzes a video for pacing metrics.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # --- 1. Video Metadata ---
        logger.info(f"Loading metadata for: {video_path}")
        metadata_json = video_loader.load_video_metadata(video_path)
        if not metadata_json:
             raise ValueError("Could not load video metadata")
        
        metadata = json.loads(metadata_json)
        duration_seconds = metadata.get("duration", 0)
        
        # --- 2. Frame Extraction ---
        # We assume frames are provided or we extract them. 
        # (This block is often redundant if called from pipeline, checking just in case)
        if (frames is None or len(frames) == 0):
             if frame_folder and os.path.exists(frame_folder):
                 pass # Use folder
             else:
                 # Extraction fallback logic
                 pass

        # --- 3. Scene / Editing Features ---
        logger.info("Detecting scenes...")
        scenes = scene_detector.detect_scenes(video_path)
        
        number_of_cuts = len(scenes)
        
        if duration_seconds > 0:
            cuts_per_minute = number_of_cuts / (duration_seconds / 60)
        else:
            cuts_per_minute = 0
            
        if number_of_cuts > 0:
             avg_shot_duration = duration_seconds / number_of_cuts
        else:
             avg_shot_duration = duration_seconds
             
        scene_stats = visual_features.scene_features(scenes)
        shot_duration_variance = scene_stats.get("pace_variance", 0.0)
        
        # --- 4. Motion / Energy Features ---
        logger.info("Computing motion intensity...")
        # Pass frames list
        motion_stats = visual_features.motion_features(frame_folder, frames=frames)
        motion_intensity = motion_stats.get("motion_intensity", 0.0)
        
        # --- 5. Text Overlay Frequency ---
        logger.info("Analyzing text overlay...")
        # Pass frames list
        text_stats = text_features.extract_text_features(frame_folder, frames=frames)
        text_overlay_ratio = text_stats.get("text_presence_ratio", 0.0)

        # --- 6. Video Type Detection ---
        video_type = "short" if duration_seconds <= Config.PACING_SHORT_VIDEO_THRESHOLD else "long"
        
        # --- 7. Pacing Rule Logic ---
        input_cpm = cuts_per_minute
        pacing_category = "optimal" # Default
        
        if video_type == "short":
            if input_cpm < Config.PACING_SHORT_MIN_CUTS:
                pacing_category = "slow"
            elif Config.PACING_SHORT_MIN_CUTS <= input_cpm <= Config.PACING_SHORT_MAX_CUTS:
                pacing_category = "optimal"
            else:
                pacing_category = "too_fast"
        else: # long
            if input_cpm < Config.PACING_LONG_MIN_CUTS:
                pacing_category = "slow"
            elif Config.PACING_LONG_MIN_CUTS <= input_cpm <= Config.PACING_LONG_MAX_CUTS:
                pacing_category = "optimal"
            else:
                pacing_category = "too_fast"
                
        # --- 8. Pace Score Calculation ---
        # Normalize features first using Config constants
        
        norm_cuts = min(cuts_per_minute / Config.PACING_NORM_CUTS, 1.0)
        norm_motion = min(motion_intensity / Config.PACING_NORM_MOTION, 1.0)
        norm_variance = min(shot_duration_variance / Config.PACING_NORM_VARIANCE, 1.0)
        norm_text = text_overlay_ratio # Already 0-1
        
        # Weighted formula using Config weights
        raw_score = (
            (Config.PACING_WEIGHT_CUTS * norm_cuts) +
            (Config.PACING_WEIGHT_MOTION * norm_motion) +
            (Config.PACING_WEIGHT_VARIANCE * norm_variance) +
            (Config.PACING_WEIGHT_TEXT * norm_text)
        )
        
        pace_score = int(min(max(raw_score * 100, 0), 100))

        # Construct result
        result = {
            "video_type": video_type,
            "pace_category": pacing_category,
            "pace_score": pace_score,
            "cuts_per_minute": round(cuts_per_minute, 2),
            "avg_shot_duration": round(avg_shot_duration, 2),
            "shot_duration_variance": round(shot_duration_variance, 2),
            "motion_intensity": round(motion_intensity, 2),
            "text_overlay_ratio": round(text_overlay_ratio, 2)
        }
        
        return result

    except Exception as e:
        logger.error(f"Error in pacing analysis: {str(e)}")
        # Handle errors gracefully by returning a structure with error info or re-raising?
        # Prompt says "handle errors if the video cannot be loaded".
        # It implies returning what we can or a valid structure.
        # Let's return a basic structure with zeros or None to avoid breaking the API.
        return {
            "error": str(e),
            "video_type": "unknown",
            "pace_category": "unknown", 
            "pace_score": 0
        }

if __name__ == "__main__":
    # Test block
    if len(sys.argv) > 1:
        v_path = sys.argv[1]
        print(json.dumps(analyze_pacing(v_path), indent=4))
    else:
        print("Usage: python pacing_analysis.py <video_path>")

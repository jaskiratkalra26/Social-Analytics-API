import os
import sys
import json
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import Config
from pipeline import video_loader, frame_extractor, audio_extractor
from features import hook_analysis, pacing_analysis, lighting_analysis, text_overlay_analysis, clip_analysis, popular_hashtags, platform_recommendation

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SocialAnalytics")

def analyze_video(video_path: str) -> dict:
    """
    Runs the complete Social Analytics Pipeline on a video.
    Returns aggregated JSON results including:
    - Metadata
    - Hook Analysis
    - Pacing Analysis
    - Lighting Analysis
    - Text Analysis
    """
    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}
        
    logger.info(f"Starting analysis for: {video_path}")
    
    results = {
        "video_path": video_path,
        "metadata": {},
        "hook_analysis": {},
        "pacing_analysis": {},
        "lighting_analysis": {},
        "text_analysis": {},
        "content_classification": {},
        "platform_recommendation": {},
        "popular_hashtags": []
    }
    
    # 1. Metadata
    try:
        meta_json = video_loader.load_video_metadata(video_path)
        if meta_json:
            results["metadata"] = json.loads(meta_json)
    except Exception as e:
        logger.error(f"Metadata error: {e}")

    # 2. Frame Extraction (Centralized)
    # Extract at TARGET_FPS (e.g. 1 FPS is usually enough for general analysis, maybe 2 for better pacing/hook)
    extract_fps = getattr(Config, 'TARGET_FPS', 1)
    logger.info(f"Extracting frames at {extract_fps} FPS...")
    try:
        frame_folder = frame_extractor.extract_frames(video_path, fps_sampling=extract_fps)
        logger.info(f"Frames extracted to: {frame_folder}")
    except Exception as e:
        logger.error(f"Frame extraction error: {e}")
        return {"error": "Frame extraction failed"}

    # 3. Audio Extraction (Centralized)
    logger.info("Extracting audio...")
    try:
        # returns path or None
        audio_path = audio_extractor.extract_audio(video_path)
        if audio_path:
             logger.info(f"Audio extracted to: {audio_path}")
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")

    # 4. Run Analysis Modules
    
    # Hook Analysis
    logger.info("Running Hook Analysis...")
    try:
        results["hook_analysis"] = hook_analysis.analyze_hook(video_path, frame_folder)
    except Exception as e:
        logger.error(f"Hook analysis error: {e}")
        results["hook_analysis"] = {"error": str(e)}

    # Pacing Analysis
    logger.info("Running Pacing Analysis...")
    try:
        results["pacing_analysis"] = pacing_analysis.analyze_pacing(video_path, frame_folder=frame_folder)
    except Exception as e:
        logger.error(f"Pacing analysis error: {e}")
        results["pacing_analysis"] = {"error": str(e)}

    # Lighting Analysis
    logger.info("Running Lighting Analysis...")
    try:
        # Lighting analysis normally downsamples, but passing full folder is safer than extracting again 
        # and overwriting if we just accept it might process more frames.
        results["lighting_analysis"] = lighting_analysis.analyze_lighting(video_path, frame_folder=frame_folder)
    except Exception as e:
        logger.error(f"Lighting analysis error: {e}")
        results["lighting_analysis"] = {"error": str(e)}

    # Text Overlay Analysis
    logger.info("Running Text Overlay Analysis...")
    try:
        results["text_analysis"] = text_overlay_analysis.analyze_text_overlay(frame_folder)
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        results["text_analysis"] = {"error": str(e)}

    # Content Classification
    logger.info("Running Content Classification (CLIP+LightGBM)...")
    try:
        results["content_classification"] = clip_analysis.analyze_content(video_path, frame_folder)
    except Exception as e:
        logger.error(f"Content classification error: {e}")
        results["content_classification"] = {"error": str(e)}

    # Platform Recommendation & Hashtags
    logger.info("Running Platform Recommendation & Hashtag Generation...")
    try:
        # Prepare features for recommendation
        video_features = {
            "duration_seconds": results["metadata"].get("duration", 0),
            "hook_score": results["hook_analysis"].get("hook_score", 0),
            "pacing_score": results["pacing_analysis"].get("pace_score", 0),
            "lighting_score": results["lighting_analysis"].get("lighting_score", 0),
            "text_score": results["text_analysis"].get("text_score", 0),
            "niche": results["content_classification"].get("predicted_label", "")
        }
        
        # Get Recommendation
        results["platform_recommendation"] = platform_recommendation.recommend_platform(video_features)
        
        # Get Hashtags & Upload Time based on niche
        niche = video_features["niche"]
        if niche:
            results["popular_hashtags"] = popular_hashtags.get_popular_hashtags_by_category(niche)
            results["upload_schedule"] = {
                "best_time": platform_recommendation.get_best_upload_time(niche),
                "timezone": "User Local Time" 
            }
            
    except Exception as e:
        logger.error(f"Platform recommendation error: {e}")
        results["platform_recommendation"] = {"error": str(e)}

    # 5. Calculate Overall Score (Simple Average Strategy)
    try:
        hook_score = results["hook_analysis"].get("hook_score", 0)
        pace_score = results["pacing_analysis"].get("pace_score", 0)
        light_score = results["lighting_analysis"].get("lighting_score", 0)
        text_score = results["text_analysis"].get("text_score", 0)
        
        # Weighted average
        total_score = (hook_score * 0.35) + (pace_score * 0.25) + (text_score * 0.25) + (light_score * 0.15)
        results["viral_score"] = round(total_score, 1)
        
    except Exception:
        results["viral_score"] = 0

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Social Analytics Pipeline")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    
    final_results = analyze_video(args.video_path)
    
    # Save final results to JSON file
    output_filename = os.path.splitext(os.path.basename(args.video_path))[0] + "_results.json"
    output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
    

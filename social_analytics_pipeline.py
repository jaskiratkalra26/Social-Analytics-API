import os
import sys
import json
import logging
import time
import concurrent.futures

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import Config
from pipeline import video_loader, frame_extractor, audio_extractor, scene_detector
from analysis import hook_analysis, pacing_analysis, lighting_analysis, text_overlay_analysis, clip_analysis, platform_recommendation, subject_presentation
from analysis.storytelling_clarity import compute_storytelling_clarity
from features import popular_hashtags, visual_features

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SocialAnalytics")

def run_subject_analysis(frame_folder, frames):
    """
    Helper to run subject analysis in parallel.
    Consisting of:
    1. Extract visual_features (subject_features)
    2. Compute subject_presentation score
    """
    try:
        raw_features = visual_features.subject_features(frame_folder, frames=frames)
        return subject_presentation.compute_subject_presentation(raw_features)
    except Exception as e:
        logger.error(f"Subject analysis error: {e}")
        return {"error": str(e)}

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
        "subject_presentation": {},
        "storytelling_clarity": {},
        "platform_recommendation": {},
        "popular_hashtags": []
    }
    
    # --- STAGE 1: Extraction & Metadata (Parallel) ---
    logger.info("Starting extraction phase...")
    
    frame_folder = None
    frames_list = [] # List to hold in-memory frames
    audio_path = None
    scene_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Define tasks
        future_meta = executor.submit(video_loader.load_video_metadata, video_path)
        
        extract_fps = getattr(Config, 'TARGET_FPS', 1)
        # Note: We rely on frame_extractor returning (path, list) now
        future_frames = executor.submit(frame_extractor.extract_frames, video_path, fps_sampling=extract_fps)
        
        future_audio = executor.submit(audio_extractor.extract_audio, video_path)
        
        future_scenes = executor.submit(scene_detector.detect_scenes, video_path)

        # Wait for results
        try:
            meta_json = future_meta.result()
            if meta_json:
                results["metadata"] = json.loads(meta_json)
        except Exception as e:
            logger.error(f"Metadata error: {e}")

        try:
            # Unpack the tuple (folder, list)
            result = future_frames.result()
            if isinstance(result, tuple) and len(result) == 2:
                frame_folder, frames_list = result
            else:
                frame_folder = result # Fallback if someone reverted the change
                frames_list = [] # No memory optimization
                
            logger.info(f"Frames extracted to: {frame_folder} (Count: {len(frames_list)})")
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return {"error": "Frame extraction failed"}

        try:
            audio_path = future_audio.result()
            if audio_path:
                 logger.info(f"Audio extracted to: {audio_path}")
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            
        try:
            scene_list = future_scenes.result()
            logger.info(f"Detected {len(scene_list)} scenes")
        except Exception as e:
             logger.error(f"Scene detection error: {e}")

    # --- STAGE 2: Analysis Modules (Parallel) ---
    logger.info("Starting analysis phase...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Pass frames_list to all functions
        future_hook = executor.submit(hook_analysis.analyze_hook, video_path, frame_folder, frames=frames_list, scenes=scene_list)
        future_pacing = executor.submit(pacing_analysis.analyze_pacing, video_path, frame_folder=frame_folder, frames=frames_list, scenes=scene_list)
        future_lighting = executor.submit(lighting_analysis.analyze_lighting, video_path, frame_folder=frame_folder, frames=frames_list)
        future_text = executor.submit(text_overlay_analysis.analyze_text_overlay, frame_folder, frames=frames_list)
        future_clip = executor.submit(clip_analysis.analyze_content, video_path, frame_folder, frames=frames_list)
        future_subject = executor.submit(run_subject_analysis, frame_folder, frames_list)
        
        # Collect results as they complete

        try:
            results["hook_analysis"] = future_hook.result()
        except Exception as e:
            logger.error(f"Hook analysis error: {e}")
            results["hook_analysis"] = {"error": str(e)}
            
        try:
            results["pacing_analysis"] = future_pacing.result()
        except Exception as e:
            logger.error(f"Pacing analysis error: {e}")
            results["pacing_analysis"] = {"error": str(e)}

        try:
            results["lighting_analysis"] = future_lighting.result()
        except Exception as e:
            logger.error(f"Lighting analysis error: {e}")
            results["lighting_analysis"] = {"error": str(e)}

        try:
            results["text_analysis"] = future_text.result()
        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            results["text_analysis"] = {"error": str(e)}

        try:
            results["content_classification"] = future_clip.result()
        except Exception as e:
            logger.error(f"Content classification error: {e}")
            results["content_classification"] = {"error": str(e)}

        try:
            results["subject_presentation"] = future_subject.result()
        except Exception as e:
            logger.error(f"Subject presentation error: {e}")
            results["subject_presentation"] = {"error": str(e)}

    # --- Storytelling Clarity Analysis ---
    logger.info("Running Storytelling Clarity Analysis...")
    try:
        # Extract metrics safely with defaults
        pacing_data = results.get("pacing_analysis", {})
        text_data = results.get("text_analysis", {})
        hook_metrics = results.get("hook_analysis", {}).get("hook_metrics", {})
        text_metrics = text_data.get("text_metrics", {})
        
        clarity_metrics = {
            "pace_score": pacing_data.get("pace_score", 0) / 100.0,
            
            # Normalize motion (0-10 -> 0-1 range approx)
            "motion_flow_score": min(pacing_data.get("motion_intensity", 0) / Config.PACING_NORM_MOTION, 1.0),
            
            # Text presence ratio (already 0-1)
            "text_support_score": text_metrics.get("text_presence_ratio", 0),
            
            # Scene consistency: Inverse of variance (normalized)
            "scene_consistency_score": max(0, 1.0 - (pacing_data.get("shot_duration_variance", 0) / Config.PACING_NORM_VARIANCE)),
            
            # Text readability: Approximation via font size score or general text score
            "text_readability_score": text_metrics.get("font_size_score", 0) / 100.0, 

            # Hook text ratio
            "hook_text_ratio": hook_metrics.get("hook_text_ratio", 0)
        }
        
        results["storytelling_clarity"] = compute_storytelling_clarity(clarity_metrics)
        
    except Exception as e:
        logger.error(f"Storytelling clarity error: {e}")
        results["storytelling_clarity"] = {"error": str(e)}


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
        results["platform_recommendation"] = {}
        results["platform_recommendation"] = platform_recommendation.recommend_platform(video_features)
        
        # Get Hashtags & Upload Time based on niche
        niche = video_features["niche"]
        results["popular_hashtags"] = [] 
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
        subject_score = results["subject_presentation"].get("presentation_score", 0)
        
        # Weighted average (Adjusted weights)
        # Hook: 30%, Pacing: 20%, Text: 20%, Light: 15%, Subject: 15%
        total_score = (hook_score * 0.30) + (pace_score * 0.20) + (text_score * 0.20) + (light_score * 0.15) + (subject_score * 0.15)
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
    

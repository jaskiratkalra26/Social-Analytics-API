import os
import sys
import json
import logging
import time
import numpy as np
import redis
import concurrent.futures

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import Config
from Config import SUBCATEGORY_IDS
from pipeline import video_loader, frame_extractor, audio_extractor, scene_detector
from pipeline.trend_builder import get_trend_from_redis
from analysis import hook_analysis, pacing_analysis, lighting_analysis, text_overlay_analysis, clip_analysis, platform_recommendation, subject_presentation, viral_analysis, audio_analysis
from analysis.storytelling_clarity import compute_storytelling_clarity
from analysis.subcategory_classification import get_subcategory
from features import popular_hashtags, visual_features, audio_features

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

def run_audio_analysis(audio_path):
    """
    Helper to run audio analysis.
    1. Extract audio features
    2. Compute audio analysis
    """
    if not audio_path or not os.path.exists(audio_path):
        return {"error": "Audio file missing"}
        
    try:
        features = audio_features.extract_audio_features(audio_path)
        return audio_analysis.compute_audio_analysis(features)
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
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
        "audio_analysis": {},
        "content_classification": {},
        "subcategory": {},
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
        future_pacing = executor.submit(pacing_analysis.analyze_pacing, video_path, frame_folder=frame_folder, frames=frames_list, scenes=scene_list)
        future_lighting = executor.submit(lighting_analysis.analyze_lighting, video_path, frame_folder=frame_folder, frames=frames_list)
        future_text = executor.submit(text_overlay_analysis.analyze_text_overlay, frame_folder, frames=frames_list)
        future_clip = executor.submit(clip_analysis.analyze_content, video_path, frame_folder, frames=frames_list)
        future_subject = executor.submit(run_subject_analysis, frame_folder, frames_list)
        future_audio = executor.submit(run_audio_analysis, audio_path)
        
        # Collect results as they complete

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
            results["audio_analysis"] = future_audio.result()
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            results["audio_analysis"] = {"error": str(e)}

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

    # --- STAGE 2.5: Hook Analysis (Sequential) ---
    logger.info("Running Hook Analysis (using cached text and audio metrics)...")
    try:
        # Pass text_analysis directly so it never runs duplicate OCR
        results["hook_analysis"] = hook_analysis.analyze_hook(
            video_path, 
            frame_folder, 
            frames=frames_list, 
            scenes=scene_list, 
            text_data=results.get("text_analysis")
        )
    except Exception as e:
        logger.error(f"Hook analysis error: {e}")
        results["hook_analysis"] = {"error": str(e)}

    # --- SUBCATEGORY CLASSIFICATION & COMPETITION DATA ---
    logger.info("Running Subcategory Classification & Fetching Competition Data...")
    try:
        predicted_category = results["content_classification"].get("predicted_label")
        raw_embeddings     = results["content_classification"].get("video_embeddings")

        if predicted_category and raw_embeddings is not None:
            embeddings_np = np.array(raw_embeddings, dtype=np.float32)

            # get_subcategory returns the readable name
            subcat_name = get_subcategory(predicted_category, embeddings_np)
            
            # Look up the global integer ID (e.g., 65)
            subcat_id = SUBCATEGORY_IDS.get(subcat_name) if subcat_name else None

            results["subcategory"] = {
                "name": subcat_name,
                "id": subcat_id,
            }
            results["competition_data"] = {}
            comp_data_found = False

            if subcat_id is not None:
                # 1. FAST REDIS FETCH (Exactly by ID)
                try:
                    redis_client = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
                    redis_key = f"competition:{subcat_id}"
                    raw_comp = redis_client.get(redis_key)
                    if raw_comp:
                        results["competition_data"] = json.loads(raw_comp)
                        comp_data_found = True
                        logger.info(f"Loaded competition data directly from Redis -> {redis_key}")
                except Exception as e:
                    logger.warning(f"Failed to fetch from Redis, checking fallback: {e}")

            # 2. FALLBACK JSON FETCH (Exactly by ID mapping)
            if not comp_data_found and subcat_id is not None:
                try:
                    json_path = os.path.join(Config.OUTPUT_DIR, "fallback", "competition_data.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            fallback_data = json.load(f)

                        # The new JSON format puts everything cleanly under 'data' using string id keys!
                        item = fallback_data.get("data", {}).get(str(subcat_id))
                        if item:
                            results["competition_data"] = item
                            comp_data_found = True
                            logger.info(f"Loaded competition data from JSON fallback -> id: {subcat_id}")
                except Exception as e:
                    logger.warning(f"Failed to load from JSON fallback: {e}")

            if not comp_data_found:
                logger.warning(f"Competition data not found for id: {subcat_id}")

        else:
            results["subcategory"] = {"name": None, "id": None}
            results["competition_data"] = {}

    except Exception as e:
        logger.error(f"Subcategory classification error: {e}")
        results["subcategory"] = {"error": str(e)}
        results["competition_data"] = {}
    finally:
        # Always strip embeddings — they're internal, not part of the output
        results["content_classification"].pop("video_embeddings", None)

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
            "pace_category": pacing_data.get("pace_category", "optimal"),
            
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
    
    # --- TREND ALIGNMENT CHECK ---
    logger.info("Checking Trend Alignment...")
    try:
        # Fetch trend data from Redis (or fallback to defaults)
        trend_data = get_trend_from_redis()
        
        trend_info = {
            "status": "not_trending",
            "description": "Trend data unavailable or video category unknown.",
            "rank": None,
            "trend_category": None,
            "trend_score": 0.0,
            "trending_keywords": []
        }
        
        if trend_data and "top_categories" in trend_data:
            predicted_niche = results.get("content_classification", {}).get("predicted_label", None)
            
            if predicted_niche:
                matched_category = None
                matched_rank = None
                pred_clean = predicted_niche.lower().strip()
                
                # Try finding a match
                for rank_idx, category in enumerate(trend_data.get("top_categories", []), 1):
                    cat_name = category["category"].lower().strip()
                    cat_words = set(cat_name.split())
                    
                    is_match = False
                    
                    # 1. Exact or Substring match
                    if pred_clean == cat_name or pred_clean in cat_name or cat_name in pred_clean:
                        is_match = True
                        
                    # 2. Word intersection
                    if not is_match and pred_clean in cat_words:
                         is_match = True
                    
                    # 3. Specific Aliases
                    aliases = {
                        "technology": ["science", "computing"],
                        "gaming": ["entertainment"],
                        "entertainment": ["movies", "tv", "comedy", "music"],
                        "movies": ["entertainment", "film"],
                        "music": ["entertainment"]
                    }
                    if not is_match and pred_clean in aliases:
                        if any(alias in cat_name for alias in aliases[pred_clean]):
                            is_match = True

                    if is_match:
                        matched_category = category
                        matched_rank = rank_idx
                        break

                if matched_category:
                    trend_info = {
                        "status": "active_trend",
                        "description": f"The category '{matched_category['category']}' is currently trending #{matched_rank} on YouTube.",
                        "rank": matched_rank,
                        "trend_category": matched_category["category"],
                        "trend_score": matched_category["score"],
                        "trending_keywords": matched_category["keywords"][:5] # Top 5 trending titles/keywords
                    }
                else:
                     trend_info["status"] = "niche_content"
                     trend_info["description"] = f"Topic '{predicted_niche}' is not currently in the top 3 trends."
                     trend_info["trend_category"] = predicted_niche
            else:
                trend_info["description"] = "Could not predict video category to compare with trends."
        else:
            trend_info["description"] = "Trend data unavailable in Redis (run pipeline.trend_builder first)."
            
        results["trend_alignment"] = trend_info
        
    except Exception as e:
        logger.error(f"Trend alignment error: {e}")
        results["trend_alignment"] = {"error": str(e)}

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


    # 5. Viral Pattern Analysis (Final Aggregation)
    logger.info("Running Viral Pattern Analysis...")
    try:
        pacing_data = results.get("pacing_analysis", {})
        text_metrics = results.get("text_analysis", {}).get("text_metrics", {})
        story_data = results.get("storytelling_clarity", {})
        
        viral_metrics = {
            "hook_score": results.get("hook_analysis", {}).get("hook_score", 0) / 100.0,
            "pace_score": pacing_data.get("pace_score", 0) / 100.0,
            "pace_category": pacing_data.get("pace_category", "optimal"),
            "motion_flow_score": min(pacing_data.get("motion_intensity", 0) / Config.PACING_NORM_MOTION, 1.0),
            "text_support_score": text_metrics.get("text_presence_ratio", 0),
            "subject_presentation_score": results.get("subject_presentation", {}).get("presentation_score", 0) / 100.0,
            "storytelling_clarity_score": story_data.get("clarity_score", 0) / 100.0
        }
        
        results["viral_analysis"] = viral_analysis.compute_viral_analysis(viral_metrics)
        # results["viral_score"] is now available inside results["viral_analysis"]["viral_score"]
        
    except Exception as e:
        logger.error(f"Viral analysis error: {e}")
        results["viral_analysis"] = {"viral_score": 0, "error": str(e)}

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
    

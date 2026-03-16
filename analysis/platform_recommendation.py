"""
Platform Recommendation Module

This module determines the most suitable social media platform for a video 
based on extracted features such as duration, hook score, pacing, and visual quality.
It uses a rule-based approach to ensure transparency and speed.
"""

import os
import sys

# Add project root to sys.path to allow importing Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config

def recommend_platform(video_features: dict) -> dict:
    """
    Analyzes video features to recommend the best platform for posting.

    Args:
        video_features (dict): A dictionary containing video analytics metrics.
                               Expected keys:
                               - duration_seconds (float/int)
                               - hook_score (float/int)
                               - pacing_score (float/int)
                               - lighting_score (float/int)
                               - text_score (float/int)
                               - niche (str)

    Returns:
        dict: A dictionary containing:
              - video_type (str): "short" or "long"
              - recommended_platform (str): Name of the recommended platform
              - reason (str): Explanation for the recommendation
              - platform_scores (dict, optional): Scores for each platform (only for short videos)
    """
    # Safely retrieve features with default values
    duration = video_features.get("duration_seconds", 0)
    hook_score = video_features.get("hook_score", 0)
    pacing_score = video_features.get("pacing_score", 0)
    lighting_score = video_features.get("lighting_score", 0)
    text_score = video_features.get("text_score", 0)
    niche = video_features.get("niche", "").lower()

    # Video Type Detection
    if duration > Config.VIDEO_TYPE_SHORT_THRESHOLD:
        return {
            "video_type": "long",
            "recommended_platform": "YouTube",
            "reason": "Long-form videos perform best on YouTube"
        }

    # Short Video Platform Scoring
    video_type = "short"
    tiktok_score = 0
    reels_score = 0
    shorts_score = 0

    # TikTok Scoring Rules
    if hook_score > Config.TIKTOK_HOOK_THRESHOLD:
        tiktok_score += Config.TIKTOK_HOOK_BONUS
    if pacing_score > Config.TIKTOK_PACING_THRESHOLD:
        tiktok_score += Config.TIKTOK_PACING_BONUS
    if duration < Config.TIKTOK_DURATION_THRESHOLD_LOW:
        tiktok_score += Config.TIKTOK_DURATION_BONUS
    if text_score > Config.TIKTOK_TEXT_THRESHOLD:
        tiktok_score += Config.TIKTOK_TEXT_BONUS
    
    # TikTok Penalty
    if duration > Config.TIKTOK_PENALTY_THRESHOLD:
        tiktok_score -= Config.TIKTOK_PENALTY_VALUE

    # Instagram Reels Scoring Rules
    if lighting_score > Config.REELS_LIGHTING_THRESHOLD:
        reels_score += Config.REELS_LIGHTING_BONUS
    if text_score > Config.REELS_TEXT_THRESHOLD:
        reels_score += Config.REELS_TEXT_BONUS
    if hook_score > Config.REELS_HOOK_THRESHOLD:
        reels_score += Config.REELS_HOOK_BONUS
    if Config.REELS_PACING_RANGE_MIN <= pacing_score <= Config.REELS_PACING_RANGE_MAX:
        reels_score += Config.REELS_PACING_BONUS

    # YouTube Shorts Scoring Rules
    if duration > Config.SHORTS_DURATION_THRESHOLD:
        shorts_score += Config.SHORTS_DURATION_BONUS
    if hook_score > Config.SHORTS_HOOK_THRESHOLD:
        shorts_score += Config.SHORTS_HOOK_BONUS
    if text_score > Config.SHORTS_TEXT_THRESHOLD:
        shorts_score += Config.SHORTS_TEXT_BONUS
    if niche == "education":
        shorts_score += Config.SHORTS_NICHE_EDUCATION_BONUS

    # Platform Selection
    platform_scores = {
        "tiktok": tiktok_score,
        "instagram_reels": reels_score,
        "youtube_shorts": shorts_score
    }

    # Find the platform with the highest score
    # In case of a tie, the first one encountered in max iteration is returned, 
    # but practically max on a dict uses keys or values depending on implementation. 
    # Here we specify key to return the platform name.
    recommended_platform_key = max(platform_scores, key=platform_scores.get)

    # Map keys to display names and Generate Reasons
    reason = ""
    recommended_platform_name = ""

    if recommended_platform_key == "tiktok":
        recommended_platform_name = "TikTok"
        reason = "Strong hook and fast pacing perform best on TikTok"
    elif recommended_platform_key == "instagram_reels":
        recommended_platform_name = "Instagram Reels"
        reason = "High visual quality and balanced pacing suit Instagram Reels"
    elif recommended_platform_key == "youtube_shorts":
        recommended_platform_name = "YouTube Shorts"
        reason = "Slightly longer short-form content works well on YouTube Shorts"
    else:
        # Fallback (should not be reached with current logic)
        recommended_platform_name = recommended_platform_key
        reason = "Best matched platform based on scoring rules"

    return {
        "video_type": video_type,
        "recommended_platform": recommended_platform_name,
        "reason": reason,
        "platform_scores": platform_scores
    }

def get_best_upload_time(niche: str) -> str:
    """
    Determines the best upload time based on the content niche.
    
    Args:
        niche (str): The identified category of the video.
        
    Returns:
        str: The recommended time window.
    """
    if not niche:
        return "Unknown"
        
    niche_lower = niche.lower()
    
    # Try exact match first
    if niche_lower in Config.CATEGORY_UPLOAD_TIME:
        return Config.CATEGORY_UPLOAD_TIME[niche_lower]
        
    # Try to find key in niche text (e.g. "finance tips" contains "finance")
    for category, time_window in Config.CATEGORY_UPLOAD_TIME.items():
        if category in niche_lower:
            return time_window
            
    return "Unknown"

import os
import sys
import json
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config
from features import visual_features, text_features, audio_features
from pipeline import scene_detector, video_loader

def analyze_hook(video_path: str, frame_folder: str, frames: list = None, scenes: list = None) -> dict:
    """
    Analyzes the hook (first 3 seconds) of a video.
    Returns hook score and insights.
    
    Args:
        video_path (str): Path to the input video.
        frame_folder (str): Folder containing extracted frames from the video.
        frames (list, optional): List of loaded frames to use instead of reading from disk.
        
    Returns:
        dict: Hook analysis result containing score, category, issues, and metrics.
    """
    
    if not os.path.exists(video_path):
        return {"error": f"Video path not found: {video_path}"}
        
    # Standard output structure
    hook_data = {
        "hook_score": 100,
        "hook_category": "",
        "issues_detected": [],
        "improvement_suggestions": [],
        "hook_metrics": {}
    }

    # 1. Load Video Metadata for duration/fps
    try:
        meta_json_str = video_loader.load_video_metadata(video_path)
        if meta_json_str:
            metadata = json.loads(meta_json_str)
        else:
            metadata = {}
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = {}

    # 2. Determine Hook Duration (3 seconds) in frames
    extract_fps = getattr(Config, 'TARGET_FPS', 1) 
    hook_duration_sec = Config.HOOK_DURATION
    hook_frames_count = int(hook_duration_sec * extract_fps)
    if hook_frames_count < 2: 
        hook_frames_count = 2

    # 3. Visual Features (Motion)
    try:
        # Pass frames list to motion_features
        motion_data = visual_features.motion_features(frame_folder, max_frames=hook_frames_count, frames=frames)
        motion_intensity = motion_data.get("motion_intensity", 0.0)
    except Exception as e:
        print(f"Error calculating motion: {e}")
        motion_intensity = 0.0

    # 4. Text Features (Hook Text Ratio)
    try:
        # Pass frames list to extract_text_features
        text_data = text_features.extract_text_features(frame_folder, frames=frames)
        hook_text_ratio = text_data.get("hook_text_ratio", 0.0)
    except Exception as e:
        print(f"Error extracting text features: {e}")
        hook_text_ratio = 0.0

    # 5. Audio Features
    # Construct audio path based on pipeline convention
    video_filename = os.path.basename(video_path)
    audio_filename = os.path.splitext(video_filename)[0] + ".wav"
    audio_path = os.path.join(Config.AUDIO_OUTPUT_DIR, audio_filename)
    
    audio_energy = 0.0
    speech_presence = False
    
    if os.path.exists(audio_path):
        try:
            audio_data = audio_features.extract_audio_features(audio_path)
            # Use 'hook_audio_intensity' if available (specific to first 3s)
            raw_hook_energy = audio_data.get("hook_audio_intensity", 0.0)
            
            # If hook energy is 0 (e.g. failure), fall back to global audio_energy,
            # but ideally we trust hook_audio_intensity
            audio_energy = raw_hook_energy
            
            # Speech presence logic
            # Using 'speech_clarity' as a proxy. 
            # Threshold > 0.5 chosen arbitrarily based on 'clarity' concept (mean > 0.5*std).
            # This is a heuristic.
            speech_clarity = audio_data.get("speech_clarity", 0.0)
            speech_presence = speech_clarity > 0.5 
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
    else:
        # Audio file missing implies silence or error
        # Penalties will apply naturally due to 0 values
        pass

    # 6. Scene Detector (Cuts in first 3s)
    scene_cuts_first_3s = 0
    try:
        # Detect scenes returns list of (start, end)
        if scenes is None:
            scenes = scene_detector.detect_scenes(video_path)
            
        # Count scenes starting between 0 (exclusive) and Config.HOOK_DURATION
        # A cut is a transition. The first scene starts at 0.0 (usually).
        # Subsequent starts imply cuts.
        count = 0
        for start_sec, end_sec in scenes:
            if 0.0 < start_sec < float(Config.HOOK_DURATION):
                count += 1
        scene_cuts_first_3s = count
        
    except Exception as e:
        print(f"Error detecting scenes: {e}")

    # Populate Hook Metrics
    hook_metrics = {
        "motion_intensity": float(round(motion_intensity, 2)),
        "hook_text_ratio": float(round(hook_text_ratio, 2)),
        "audio_energy": float(round(audio_energy, 2)),
        "speech_presence": bool(speech_presence),
        "scene_cuts_first_3s": int(scene_cuts_first_3s)
    }

    # Algorithm Scoring
    score = 100
    
    # Rule 1 - Low Visual Motion
    if motion_intensity < Config.HOOK_MIN_MOTION:
        score -= Config.HOOK_PENALTY_LOW_MOTION
        hook_data["issues_detected"].append("low visual motion in opening")
        hook_data["improvement_suggestions"].append("add faster visual movement in first 3 seconds")

    # Rule 2 - Slow Scene Start
    # 0 cuts in first 3 seconds
    if scene_cuts_first_3s == 0:
        score -= Config.HOOK_PENALTY_STATIC_SCENE
        hook_data["issues_detected"].append("opening scene remains static too long")
        hook_data["improvement_suggestions"].append("introduce faster cuts or visual changes in the opening")

    # Rule 3 - Weak Hook Text
    if hook_text_ratio < Config.HOOK_MIN_TEXT_RATIO:
        score -= Config.HOOK_PENALTY_WEAK_TEXT
        hook_data["issues_detected"].append("weak or missing hook text")
        hook_data["improvement_suggestions"].append("include a strong hook text overlay")

    # Rule 4 - Low Audio Energy
    if audio_energy < Config.HOOK_MIN_AUDIO_ENERGY: 
        score -= Config.HOOK_PENALTY_LOW_AUDIO
        hook_data["issues_detected"].append("low audio energy in opening")
        hook_data["improvement_suggestions"].append("start with engaging sound or voice")

    # Rule 5 - No Speech
    if not speech_presence:
        score -= Config.HOOK_PENALTY_NO_SPEECH
        hook_data["issues_detected"].append("no speech detected in opening hook")
        hook_data["improvement_suggestions"].append("consider starting with narration or voice introduction")

    # Normalization
    score = max(0, min(100, score))
    hook_data["hook_score"] = score
    
    # Category
    if score > Config.HOOK_SCORE_STRONG:
        cat = "strong"
    elif score >= Config.HOOK_SCORE_MODERATE:
        cat = "moderate"
    elif score >= Config.HOOK_SCORE_WEAK:
        cat = "weak"
    else:
        cat = "very_weak"
    hook_data["hook_category"] = cat
    
    hook_data["hook_metrics"] = hook_metrics
    
    return hook_data

if __name__ == "__main__":
    # Test block
    print("Hook Analysis Module Loaded")

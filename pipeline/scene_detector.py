import os
import sys

# Add the project root directory to Python path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

import json

def detect_scenes(video_path: str, threshold: float = None):
    """
    Detects scenes in a video using content-aware detection.
    
    Args:
        video_path (str): Path to the video file.
        threshold (float): Threshold for content detector. If None, uses value from Config.
        
    Returns:
        list: A list of tuples (start, end) in seconds.
    """
    if threshold is None:
        threshold = Config.SCENE_THRESHOLD
        
    # Check cache
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = Config.METADATA_OUTPUT_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{video_name}_scenes.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                print(f"Loading cached scenes from {cache_path}")
                return json.load(f)
        except Exception:
            pass

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    # Improve processing speed by downscaling before processing.
    video.downscale_factor = Config.SCENE_DOWNSCALE_FACTOR

    # Start video from 0 so we don't miss first scene
    # We can pass duration argument to detect_scenes if needed, but current usage is whole video
    # If the user wants only first 3 seconds, we might need a way to limit detection or filter results.
    scene_manager.detect_scenes(video, show_progress=Config.SCENE_SHOW_PROGRESS)
    
    # Returns a list of (FrameTimecode, FrameTimecode) 
    scene_list = scene_manager.get_scene_list()
    
    # Convert to seconds tuples [(start, end), ...]
    scenes_in_seconds = [
        (start.get_seconds(), end.get_seconds()) 
        for start, end in scene_list
    ]
    
    # Save to cache
    try:
        with open(cache_path, 'w') as f:
            json.dump(scenes_in_seconds, f)
            print(f"Cached scenes to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache scenes: {e}")
        
    return scenes_in_seconds

if __name__ == "__main__":
    # Example usage
    # scenes = detect_scenes("path/to/video.mp4")
    # print(scenes)
    pass

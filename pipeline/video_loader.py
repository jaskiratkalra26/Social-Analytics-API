import cv2
import os
import json
import sys

# Add the project root directory to Python path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config

def load_video_metadata(video_path: str):
    """
    Loads video metadata using OpenCV, saves it to a JSON file, and returns the JSON string.
    
    Args:
        video_path (str): The path to the video file.
        
    Returns:
        str: A JSON string containing video metadata.
        Returns None if video cannot be opened.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    # Calculate additional metadata
    video_name = os.path.basename(video_path)
    file_size = os.path.getsize(video_path)
    resolution = f"{width}x{height}"
    aspect_ratio = width / height if height > 0 else 0
    
    metadata = {
        "video_name": video_name,
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
        "file_size_bytes": file_size,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio
    }
    
    # Save metadata to JSON file in data/json folder
    os.makedirs(Config.METADATA_OUTPUT_DIR, exist_ok=True)
    json_filename = os.path.splitext(video_name)[0] + ".json"
    json_path = os.path.join(Config.METADATA_OUTPUT_DIR, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to: {json_path}")
    
    return json.dumps(metadata)

# Example usage
if __name__ == "__main__":
    # Create a dummy video file or point to an existing one to test
    pass

import cv2
import os
import sys

# Add the project root directory to Python path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config

def extract_frames(video_path, fps_sampling=None, duration=None):
    """
    Extracts frames from a video at a specified frame rate (FPS).
    Saves the frames to a subdirectory named after the video ID in the directory specified in Config.py.

    Args:
        video_path (str): Path to the input video file.
        fps_sampling (int, optional): Frames per second to extract. Defaults to Config.TARGET_FPS.
        duration (int, optional): Duration in seconds to extract frames from. If None, processes entire video.

    Returns:
        str: Path to the directory containing extract frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(Config.FRAMES_OUTPUT_DIR, video_name)
    
    # Check if frames already exist to avoid re-extraction
    if os.path.exists(output_dir):
        existing_frames = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if existing_frames:
            print(f"Frames already extracted at: {output_dir}")
            return output_dir
            
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return output_dir

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = fps_sampling if fps_sampling is not None else Config.TARGET_FPS
    
    # Calculate interval to skip frames based on target FPS
    if target_fps > 0:
        frame_interval = int(video_fps / target_fps)
    else:
        frame_interval = 1
    
    # Ensure interval is at least 1
    frame_interval = max(1, frame_interval)

    frame_count = 0
    saved_count = 0
    
    max_frames = int(duration * video_fps) if duration else float('inf')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count >= max_frames:
            break

        # Check if this frame should be saved
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    # print(f"Extracted {saved_count} frames from {video_path} into {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Example usage
    # extract_frames("path/to/video.mp4")
    pass

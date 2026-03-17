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
        existing_frames_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if existing_frames_files:
            print(f"Frames already exist at: {output_dir}. Loading into memory...")
            frames_cache = []
            for f in existing_frames_files:
                img_path = os.path.join(output_dir, f)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize on load if needed, just in case cached frames are large
                    if Config.FRAME_MAX_WIDTH and img.shape[1] > Config.FRAME_MAX_WIDTH:
                         scale = Config.FRAME_MAX_WIDTH / img.shape[1]
                         img = cv2.resize(img, (Config.FRAME_MAX_WIDTH, int(img.shape[0] * scale)))
                    frames_cache.append(img)
            return output_dir, frames_cache
            
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return output_dir, []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = fps_sampling if fps_sampling is not None else Config.TARGET_FPS
    
    # Calculate interval to skip frames based on target FPS
    if target_fps > 0:
        frame_interval = int(video_fps / target_fps)
    else:
        frame_interval = 1
        
    extracted_frames = []
    frame_count = 0
    saved_count = 0

    # Calculate max frames based on duration if provided
    max_frame_limit = float('inf')
    if duration:
        max_frame_limit = int(duration * video_fps)

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_count > max_frame_limit:
            break

        if fps_sampling:
            target_time = frame_count / video_fps
            fps_target_time = saved_count / fps_sampling
            if target_time >= fps_target_time:
                # Resize if needed for speed/storage
                if Config.FRAME_MAX_WIDTH and frame.shape[1] > Config.FRAME_MAX_WIDTH:
                    scale = Config.FRAME_MAX_WIDTH / frame.shape[1]
                    # Keep aspect ratio
                    frame = cv2.resize(frame, (Config.FRAME_MAX_WIDTH, int(frame.shape[0] * scale)))
                
                extracted_frames.append(frame)
                
                # Save frame to disk
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
        
        elif frame_count % frame_interval == 0:
            # Resize if needed for speed/storage
            if Config.FRAME_MAX_WIDTH and frame.shape[1] > Config.FRAME_MAX_WIDTH:
                scale = Config.FRAME_MAX_WIDTH / frame.shape[1]
                # Keep aspect ratio
                frame = cv2.resize(frame, (Config.FRAME_MAX_WIDTH, int(frame.shape[0] * scale)))
            
            extracted_frames.append(frame)
            
            # Save frame to disk
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return output_dir, extracted_frames


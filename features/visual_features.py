import cv2
import numpy as np
import os
import glob

def get_sorted_frame_paths(frame_folder):
    """
    Helper to get sorted list of frame paths from folder.
    Supports .jpg, .jpeg, .png, .bmp formats.
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    
    # Check if folder exists
    if not os.path.exists(frame_folder):
        return []

    for ext in extensions:
        # Use glob to find files matching pattern
        # os.path.join handles directory separators correctly
        search_path = os.path.join(frame_folder, ext)
        files.extend(glob.glob(search_path))
    
    # Sort files to ensure chronological order based on filename
    # Assuming filenames are padded numbers like frame_0001.jpg
    return sorted(files)

def scene_features(scene_list):
    """
    Computes scene and editing signals from a list of scene tuples.
    
    Args:
        scene_list: List of tuples (start_time, end_time)
        
    Returns:
        Dictionary with keys:
        - cut_frequency: Number of scenes / total duration
        - avg_scene_duration: Mean duration of scenes
        - pace_variance: Variance of scene durations
    """
    if not scene_list:
        return {
            "cut_frequency": 0.0,
            "avg_scene_duration": 0.0,
            "pace_variance": 0.0
        }

    # Calculate duration for each scene
    durations = [end - start for start, end in scene_list]
    
    # Calculate total video duration based on the end time of the last scene.
    # We sort the scene list by start time to find the true end.
    sorted_scenes = sorted(scene_list, key=lambda x: x[0])
    # Use the end time of the last scene as total duration
    total_duration = sorted_scenes[-1][1] if sorted_scenes else 0.0
    
    # Avoid division by zero
    if total_duration <= 0:
        total_duration = 1.0  # Fallback to avoid crash, though 0 duration seems invalid
        
    num_scenes = len(scene_list)
    
    cut_frequency = num_scenes / total_duration
    avg_scene_duration = np.mean(durations) if durations else 0.0
    pace_variance = np.var(durations) if durations else 0.0
    
    return {
        "cut_frequency": float(cut_frequency),
        "avg_scene_duration": float(avg_scene_duration),
        "pace_variance": float(pace_variance)
    }

def motion_features(frame_folder, max_frames=None, frames=None):
    """
    Estimates motion intensity using optical flow.
    
    Args:
        frame_folder: Path to folder containing extracted frames
        max_frames: Optional int, max number of frames to process from start
        frames: Optional list of numpy.ndarray frames. If provided, skips loading from disk.
        
    Returns:
        Dictionary with key:
        - motion_intensity: Mean magnitude of motion vectors
    """
    if frames is not None and len(frames) > 0:
        # Use provided memory frames
        if max_frames and max_frames > 0:
            processing_frames = frames[:max_frames]
        else:
            processing_frames = frames
            
        if len(processing_frames) < 2:
            return {"motion_intensity": 0.0}
            
        magnitudes = []
        prev_frame = processing_frames[0]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(processing_frames)):
            curr_frame = processing_frames[i]
            # Resize for speed (e.g. to 320 width)
            h, w = curr_frame.shape[:2]
            if w > 320:
                scale = 320 / w
                new_h = int(h * scale)
                # Note: This modifies the frame for calculation only, 
                # be careful not to modify the shared frame object if it matters elsewhere.
                # cv2.resize returns a new image, so it's safe.
                curr_frame_resized = cv2.resize(curr_frame, (320, new_h))
                if i == 1:
                     prev_frame_resized = cv2.resize(prev_frame, (320, new_h))
                     prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame_resized, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(np.mean(magnitude))
            prev_gray = curr_gray
            
        motion_intensity = np.mean(magnitudes) if magnitudes else 0.0
        return {"motion_intensity": float(motion_intensity)}

    # Fallback to loading from disk
    files = get_sorted_frame_paths(frame_folder)
    
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    
    # If only one frame exists or folder is empty, return 0
    if len(files) < 2:
        return {"motion_intensity": 0.0}
    
    magnitudes = []
    
    # Read first frame
    prev_frame = cv2.imread(files[0])
    if prev_frame is None:
        return {"motion_intensity": 0.0}
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Iterate through subsequent frames
    for i in range(1, len(files)):
        curr_frame = cv2.imread(files[i])
        if curr_frame is None:
            continue
            
        # Resize for speed (e.g. to 320 width)
        h, w = curr_frame.shape[:2]
        if w > 320:
            scale = 320 / w
            new_h = int(h * scale)
            curr_frame = cv2.resize(curr_frame, (320, new_h))
            if i == 1: # also resize prev_frame if first iteration
                prev_frame = cv2.resize(prev_frame, (320, new_h))
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Compute magnitude of flow vectors
        # flow has shape (H, W, 2)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Mean magnitude for this frame pair
        magnitudes.append(np.mean(magnitude))
        
        # Update previous frame
        prev_gray = curr_gray
        
    motion_intensity = np.mean(magnitudes) if magnitudes else 0.0
    
    return {
        "motion_intensity": float(motion_intensity)
    }

def quality_features(frame_folder):
    """
    Computes visual quality signals for frames.
    
    Args:
        frame_folder: Path to folder containing extracted frames
        
    Returns:
        Dictionary with keys:
        - brightness_mean
        - contrast_mean
        - blur_score
    """
    files = get_sorted_frame_paths(frame_folder)
    
    if not files:
        return {
            "brightness_mean": 0.0,
            "contrast_mean": 0.0,
            "blur_score": 0.0
        }
        
    brightness_values = []
    contrast_values = []
    blur_values = []
    
    for f in files:
        frame = cv2.imread(f)
        if frame is None:
            continue
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1) Brightness: Mean grayscale intensity
        brightness = np.mean(gray_frame)
        brightness_values.append(brightness)
        
        # 2) Contrast: Standard deviation of grayscale intensity
        contrast = np.std(gray_frame)
        contrast_values.append(contrast)
        
        # 3) Blur score: Laplacian variance
        blur = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        blur_values.append(blur)
        
    return {
        "brightness_mean": float(np.mean(brightness_values)) if brightness_values else 0.0,
        "contrast_mean": float(np.mean(contrast_values)) if contrast_values else 0.0,
        "blur_score": float(np.mean(blur_values)) if blur_values else 0.0
    }

def subject_features(frame_folder):
    """
    Detects presence of human faces using Haar Cascade.
    
    Args:
        frame_folder: Path to folder containing extracted frames
        
    Returns:
        Dictionary with key:
        - face_ratio: number_of_frames_with_faces / total_frames
    """
    files = get_sorted_frame_paths(frame_folder)
    
    if not files:
        return {"face_ratio": 0.0}
    
    # Load Haar Cascade for face detection
    # Using the default xml provided by cv2
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        # Fallback to just the filename if the full path construction fails or is handled internally
        cascade_path = 'haarcascade_frontalface_default.xml'
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Check if cascade loaded successfully
    if face_cascade.empty():
        # Try loading directly if cv2.data.haarcascades was not correct environment path
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            # If still empty, we can't detect faces, return 0.0 to be safe
            return {"face_ratio": 0.0}
    
    frames_with_faces = 0
    total_frames = 0
    
    for f in files:
        frame = cv2.imread(f)
        if frame is None:
            continue
            
        total_frames += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            frames_with_faces += 1
            
    face_ratio = (frames_with_faces / total_frames) if total_frames > 0 else 0.0
    
    return {
        "face_ratio": float(face_ratio)
    }

def composition_features(frame_folder):
    """
    Estimates center focus score based on brightness of center region.
    
    Args:
        frame_folder: Path to folder containing extracted frames
        
    Returns:
        Dictionary with key:
        - center_focus_score: average mean brightness of center region
    """
    files = get_sorted_frame_paths(frame_folder)
    
    if not files:
        return {"center_focus_score": 0.0}
        
    center_scores = []
    
    for f in files:
        frame = cv2.imread(f)
        if frame is None:
            continue
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_frame.shape
        
        # Define center region: middle 50% of the image
        # Height: from 1/4 to 3/4
        # Width: from 1/4 to 3/4
        start_y, end_y = height // 4, (3 * height) // 4
        start_x, end_x = width // 4, (3 * width) // 4
        
        if start_y >= end_y or start_x >= end_x:
            # Handle edge cases where frame might be too small
            center_region = gray_frame
        else:
            center_region = gray_frame[start_y:end_y, start_x:end_x]
            
        mean_brightness = np.mean(center_region)
        center_scores.append(mean_brightness)
        
    return {
        "center_focus_score": float(np.mean(center_scores)) if center_scores else 0.0
    }

def extract_visual_features(frame_folder, scene_list=None):
    """
    Master function to aggregate all visual features.
    
    Args:
        frame_folder: Path to folder containing extracted frames
        scene_list: List of tuples (start_time, end_time). Optional.
        
    Returns:
        Dictionary containing all combined features
    """
    # Initialize result dictionary
    all_features = {}
    
    # 1. Scene & Editing Signals
    # Even if scene_list is empty, we want the zeroed-out dictionary
    if scene_list is None:
        scene_list = []
    all_features.update(scene_features(scene_list))
    
    # 2. Motion & Energy Signals
    all_features.update(motion_features(frame_folder))
    
    # 3. Visual Quality Signals
    all_features.update(quality_features(frame_folder))
    
    # 4. Subject Detection
    all_features.update(subject_features(frame_folder))
    
    # 5. Composition Signals
    all_features.update(composition_features(frame_folder))
    
    return all_features

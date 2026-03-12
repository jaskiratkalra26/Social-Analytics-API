import cv2
import pytesseract
import numpy as np
import os
import glob
import sys
import json

# Add project root to sys.path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import Config
    if hasattr(Config, 'TESSERACT_CMD') and Config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
except ImportError:
    pass # Config might not be available if used as standalone

def extract_text_features(frame_folder, verbose=False):
    """
    Extracts structured OCR features from sampled frames in a folder.
    
    Args:
        frame_folder (str): Path to the folder containing video frames.
        verbose (bool): If True, prints extracted text to console.
        
    Returns:
        dict: A dictionary containing the following features:
            - text_presence_ratio: Ratio of frames with detected text.
            - text_density: Average number of characters per sampled frame.
            - font_size_score: Average relative area of text bounding boxes.
            - context_clarity: Ratio of words with length >= 3 to total words.
            - hook_text_ratio: Ratio of text presence in the first 5 sampled frames.
            
    Raises:
        FileNotFoundError: If the frame_folder does not exist.
    """
    
    if not os.path.exists(frame_folder):
        raise FileNotFoundError(f"Frame folder not found: {frame_folder}")
        
    # Get all image files sorted by name (assuming format frame_XXXX.jpg)
    # Using simple glob pattern, but ensure consistent sorting
    all_files = glob.glob(os.path.join(frame_folder, "*.jpg"))
    if not all_files:
        # Try checking for other extensions or return empty if none found
        pass 
        
    # Sort strictly by filename to ensure timeline order
    frame_files = sorted(all_files)
    
    if not frame_files:
        return {
            "text_presence_ratio": 0.0,
            "text_density": 0.0,
            "font_size_score": 0.0,
            "context_clarity": 0.0,
            "hook_text_ratio": 0.0
        }
    
    # Check for cached results
    cache_path = os.path.join(frame_folder, "text_features_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                print(f"Loading cached text features from {cache_path}")
                return json.load(f)
        except Exception:
            pass # fallback to extraction

    # PROCESS FRAMES
    # Sampling: Process frames based on configured rate
    sample_rate = getattr(Config, 'OCR_SAMPLE_RATE', 3)
    sampled_frames = frame_files[::sample_rate]
    num_sampled = len(sampled_frames)
    
    frames_with_text_count = 0
    total_characters_detected = 0
    
    # For Font Size Prominence
    total_relative_box_area = 0.0
    total_text_boxes_count = 0
    
    # For Context Clarity
    total_words_count = 0
    valid_words_count = 0
    
    # For Hook Text Ratio
    hook_frames_limit = getattr(Config, 'OCR_HOOK_FRAMES_LIMIT', 5)
    hook_text_frames_count = 0
    
    # Minimum word length for context clarity
    min_word_length = getattr(Config, 'OCR_MIN_WORD_LENGTH', 3)

    # For Reading Speed Analysis
    prev_frame_words = set()
    prev_frame_boxes = {}  # Store word -> (x, y) centroid
    reading_speeds = []
    motion_magnitudes = []
    
    # Assuming frames are extracted at 1 FPS (Config.TARGET_FPS)
    # The time difference between samples is roughly sample_rate / TARGET_FPS
    time_delta = sample_rate / getattr(Config, 'TARGET_FPS', 1.0)
    safe_time_delta = max(0.1, time_delta)

    # We iterate through sampled frames
    print(f"Starting OCR on {num_sampled} frames...")
    for idx, frame_path in enumerate(sampled_frames):
        if idx % 5 == 0:
            print(f"Processing frame {idx}/{num_sampled}...")
            
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        height, width, _ = img.shape
        frame_area = float(width * height)
        # Diagonal for normalization
        diag_pixels = np.sqrt(height**2 + width**2)
        
        if frame_area == 0:
            continue

        # Get verbose data including boxes and text
        # We need both text string (for density/clarity) and boxes (for size)
        # image_to_data returns structured data
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract is not installed or not in your PATH. See README for installation instructions.")
        except Exception:
            # If OCR fails entirely for a frame, treat as no text
            continue

        # Valid text pieces in this frame
        frame_text_content = ""
        frame_has_text = False
        current_frame_words = set()
        current_frame_boxes = {} # word -> centroid (x,y)
        
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            # Check for non-empty text
            # conf (confidence) is generally -1 for structure items, 0-100 for words
            # We filter by text content existence
            text = data['text'][i].strip()
            
            if text:
                frame_has_text = True
                frame_text_content += text + " "
                
                # Clean text for reading speed and content analysis
                clean_word = "".join(c for c in text if c.isalnum()).lower()
                if clean_word and len(clean_word) > 1:
                    current_frame_words.add(clean_word)
                    
                    # Store centroid for motion tracking
                    cx = data['left'][i] + data['width'][i] / 2
                    cy = data['top'][i] + data['height'][i] / 2
                    # If multiple occurrences, average them or just take last (simple approach)
                    current_frame_boxes[clean_word] = (cx, cy)
                
                # Font Size Prominence Calculation
                # box area
                w = data['width'][i]
                h = data['height'][i]
                box_area = w * h
                
                # Requirements: "Compute: Average bounding box area across detected text. Normalize by frame area."
                if frame_area > 0:
                    total_relative_box_area += (box_area / frame_area)
                    total_text_boxes_count += 1
                
                # Context Clarity Calculation
                # Requirement: "Split OCR text into words."
                # image_to_data gives individual words/tokens usually
                # "Valid word: length >= min_word_length characters"
                total_words_count += 1
                if len(text) >= min_word_length:
                    valid_words_count += 1

        # Text Presence Logic - per frame
        if frame_has_text:
            frames_with_text_count += 1
            if idx < hook_frames_limit:
                hook_text_frames_count += 1
            
            if verbose:
                print(f"[Frame {idx}] Text detected: '{frame_text_content.strip()}'")

        # Reading Speed Calculation
        new_words_count = len(current_frame_words.difference(prev_frame_words))
        
        # WPM = (new_words / seconds) * 60
        wpm = (new_words_count / safe_time_delta) * 60
        if wpm > 0:
            reading_speeds.append(wpm)
            
        # Motion Magnitude Calculation
        # Compare positions of common words
        common_words = current_frame_words.intersection(prev_frame_words)
        total_dist = 0.0
        n_common = 0
        
        for word in common_words:
            if word in prev_frame_boxes and word in current_frame_boxes:
                x1, y1 = prev_frame_boxes[word]
                x2, y2 = current_frame_boxes[word]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                # Normalize dist by screen diagonal to get 0-1 magnitude relative to screen size
                if diag_pixels > 0:
                    norm_dist = dist / diag_pixels
                    total_dist += norm_dist
                    n_common += 1
        
        if n_common > 0:
            # Average motion of common words per second
            avg_norm_dist = (total_dist / n_common) / safe_time_delta
            motion_magnitudes.append(avg_norm_dist)
        
        # Update prev frame state
        prev_frame_words = current_frame_words
        prev_frame_boxes = current_frame_boxes

        # Text Density Logic
        # "Measure how much text appears."
        # "total_characters_detected / frames_sampled"
        # We aggregate char count from the re-constructed string or just sum lengths
        # Using string length from accumulated text (minus extra space at end) or sum of parts
        # Let's sum lengths of parts to be precise
        # frame_text_content has trailling space, strip it
        cleaned_frame_text = frame_text_content.replace(" ", "")
        total_characters_detected += len(cleaned_frame_text)

    # -----------------------------------------
    # COMPUTE METRICS
    # -----------------------------------------

    # 1) Text Presence Ratio
    # frames_with_text / frames_sampled
    text_presence_ratio = frames_with_text_count / num_sampled if num_sampled > 0 else 0.0

    # 2) Text Density
    # total_characters_detected / frames_sampled
    text_density = total_characters_detected / num_sampled if num_sampled > 0 else 0.0

    # 3) Font Size Prominence
    # avg_box_area / frame_area (Normalized)
    # Calculation: (Sum of (box_area/frame_area)) / Total Boxes
    font_size_score = (total_relative_box_area / total_text_boxes_count) if total_text_boxes_count > 0 else 0.0

    # 4) Context Clarity
    # valid_words / total_words
    context_clarity = valid_words_count / total_words_count if total_words_count > 0 else 0.0

    # 5) Hook Text Ratio
    # text_frames_in_first5 / 5 (or fewer if less frames)
    # "If fewer than 5 frames exist: use available frames."
    divisor_hook = min(num_sampled, hook_frames_limit)
    hook_text_ratio = hook_text_frames_count / divisor_hook if divisor_hook > 0 else 0.0

    # 6) Reading Speed (Average WPM)
    # Average only non-zero samples (moments where text appeared or changed)
    active_reading_speeds = [s for s in reading_speeds if s > 0]
    final_reading_speed = np.mean(active_reading_speeds) if active_reading_speeds else 0.0

    # Motion Magnitude (Average of detected movements)
    # 0.0 means static. 1.0 means moved entirely across screen in 1 second.
    active_motion = [m for m in motion_magnitudes if m > 0.01] # Filter tiny jitter
    final_motion_score = np.mean(active_motion) if active_motion else 0.0

    result = {
        "text_presence_ratio": float(text_presence_ratio),
        "text_density": float(text_density),
        "font_size_score": float(font_size_score),
        "context_clarity": float(context_clarity),
        "hook_text_ratio": float(hook_text_ratio),
        "reading_speed": float(final_reading_speed),
        "motion_score": float(final_motion_score)
    }
    
    # Cache result
    try:
        if not cache_path: # In case cache_path variable scope is an issue, redefine
            cache_path = os.path.join(frame_folder, "text_features_cache.json")
        with open(cache_path, 'w') as f:
            print(f"Caching text features to {cache_path}")
            json.dump(result, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not cache text features: {e}")
        
    return result


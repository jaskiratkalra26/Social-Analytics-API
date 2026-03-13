import cv2
import numpy as np
import os
import glob
import sys
import json
import easyocr
import threading
import torch

# Add project root to sys.path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import Config
except ImportError:
    pass # Config might not be available if used as standalone

# Global Singleton for Reader
_reader = None
_reader_lock = threading.Lock()
_ocr_processing_lock = threading.Lock()

def get_reader():
    """Lazy initialization of EasyOCR Reader"""
    global _reader
    with _reader_lock:
        if _reader is None:
            try:
                use_gpu = torch.cuda.is_available()
                print(f"Initializing EasyOCR (GPU={use_gpu})...")
                _reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            except Exception as e:
                print(f"Warning: Failed to initialize EasyOCR: {e}")
                _reader = None
    return _reader

def extract_text_features(frame_folder, verbose=False, frames=None):
    """
    Extracts structured OCR features from sampled frames.
    """
    if frame_folder and os.path.exists(frame_folder):
        cache_path = os.path.join(frame_folder, "text_features_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except: pass

    frames_to_process = []
    if frames is not None and len(frames) > 0:
        frames_to_process = frames
    elif frame_folder and os.path.exists(frame_folder):
        frames_to_process = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")))

    if not frames_to_process:
        return {
            "text_presence_ratio": 0.0, "text_density": 0.0, "font_size_score": 0.0,
            "context_clarity": 0.0, "hook_text_ratio": 0.0, "reading_speed": 0.0, "motion_score": 0.0
        }

    with _ocr_processing_lock:
        if frame_folder and os.path.exists(frame_folder):
             # Double check cache
             cache_path = os.path.join(frame_folder, "text_features_cache.json")
             if os.path.exists(cache_path):
                 try:
                    with open(cache_path, 'r') as f: return json.load(f)
                 except: pass

        sample_rate = getattr(Config, 'OCR_SAMPLE_RATE', 3)
        sampled_indices = range(0, len(frames_to_process), sample_rate)
        num_sampled = len(sampled_indices)
        
        frames_with_text_count = 0
        total_characters_detected = 0
        total_relative_box_area = 0.0
        total_text_boxes_count = 0
        total_words_count = 0
        valid_words_count = 0
        hook_frames_limit = getattr(Config, 'OCR_HOOK_FRAMES_LIMIT', 5)
        hook_text_frames_count = 0
        min_word_length = getattr(Config, 'OCR_MIN_WORD_LENGTH', 3)
        
        reading_speeds = []
        prev_frame_words = set()
        time_delta = sample_rate / getattr(Config, 'TARGET_FPS', 1.0)
        safe_time_delta = max(0.1, time_delta)

        reader = get_reader()
        print(f"Starting OCR on {num_sampled} frames using EasyOCR...")

        for i, idx in enumerate(sampled_indices):
            item = frames_to_process[idx]
            if isinstance(item, str):
                img = cv2.imread(item)
            else:
                img = item # ndarray
            
            if img is None: continue
            
            height, width = img.shape[:2]
            frame_area = float(width * height)
            if frame_area == 0: continue

            try:
                result = reader.readtext(img) if reader else []
            except Exception as e:
                if verbose: print(f"OCR Error: {e}")
                result = []

            has_valid_text = False
            frame_char_count = 0
            current_frame_words = set()

            for (bbox, text, prob) in result:
                if prob < 0.4: continue
                clean_text = text.strip()
                if not clean_text: continue
                
                has_valid_text = True
                frame_char_count += len(clean_text)
                
                (tl, tr, br, bl) = bbox
                box_w = np.linalg.norm(np.array(tr) - np.array(tl))
                box_h = np.linalg.norm(np.array(bl) - np.array(tl))
                total_relative_box_area += ((box_w * box_h) / frame_area)
                total_text_boxes_count += 1
                
                for w in clean_text.split():
                    w_lower = w.lower()
                    current_frame_words.add(w_lower)
                    total_words_count += 1
                    if len(w) >= min_word_length:
                        valid_words_count += 1

            if has_valid_text:
                frames_with_text_count += 1
                total_characters_detected += frame_char_count
                if idx < hook_frames_limit:
                    hook_text_frames_count += 1
            
            new_words = len(current_frame_words - prev_frame_words)
            if new_words > 0:
                reading_speeds.append((new_words / safe_time_delta) * 60)
            prev_frame_words = current_frame_words

        # Results
        res = {
            "text_presence_ratio": frames_with_text_count / num_sampled if num_sampled else 0,
            "text_density": total_characters_detected / num_sampled if num_sampled else 0,
            "font_size_score": min(100, ((total_relative_box_area / total_text_boxes_count) / 0.10 * 100) if total_text_boxes_count else 0),
            "context_clarity": valid_words_count / total_words_count if total_words_count else 0,
            "hook_text_ratio": hook_text_frames_count / sum(1 for x in sampled_indices if x < hook_frames_limit) if any(x < hook_frames_limit for x in sampled_indices) else 0,
            "reading_speed": float(np.mean(reading_speeds)) if reading_speeds else 0.0,
            "motion_score": 0.0 
        }

        if frame_folder and os.path.exists(frame_folder):
             # Ensure cache path logic is sound
             try:
                cache_path = os.path.join(frame_folder, "text_features_cache.json")
                with open(cache_path, 'w') as f: json.dump(res, f)
             except: pass
            
        return res


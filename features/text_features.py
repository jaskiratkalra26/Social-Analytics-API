import cv2
import numpy as np
import os
import glob
import sys
import json
import easyocr
import threading
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to sys.path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import Config
except ImportError:
    Config = None # Config might not be available if used as standalone

_CACHE_VERSION = "v2"

# Global Singleton for Reader
_reader = None
_reader_lock = threading.Lock()

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

# Helper function _do_ocr removed as concurrency was dropped for PyTorch safety.

def extract_text_features(frame_folder, verbose=False, frames=None):
    """
    Extracts structured OCR features from sampled frames.
    """
    if frame_folder and os.path.exists(frame_folder):
        cache_path = os.path.join(frame_folder, "text_features_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if data.get("_cache_version") == _CACHE_VERSION:
                        return data
                    else:
                        print("Cache version mismatch, running OCR fresh.")
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

    target_fps = getattr(Config, 'TARGET_FPS', 1.0) if Config else 1.0
    hook_duration_sec = getattr(Config, 'HOOK_DURATION', 3.0) if Config else 3.0
    hook_frames_limit = max(1, int(hook_duration_sec * target_fps))

    sample_rate = getattr(Config, 'OCR_SAMPLE_RATE', 3) if Config else 3
    max_frames = getattr(Config, 'OCR_MAX_FRAMES', 60) if Config else 60 # Max 60 frames to rescue massive videos

    # 1. Guarantee EVERY frame of the actual hook is analyzed so we never miss instant hook text
    hook_indices = list(range(0, min(len(frames_to_process), hook_frames_limit)))
    if not hook_indices and frames_to_process:
        hook_indices = [0]

    
    # 2. Sparsely sample the rest of the video
    rest_indices = list(range(hook_frames_limit, len(frames_to_process), sample_rate))
    
    # 3. Apply the 20-minute video upper limit protective cap uniformly
    if len(hook_indices) + len(rest_indices) > max_frames:
        allowed_rest = max_frames - len(hook_indices)
        if allowed_rest > 0:
            step_size = len(rest_indices) / allowed_rest
            rest_indices = [rest_indices[int(i * step_size)] for i in range(allowed_rest)]
        else:
            rest_indices = []

    sampled_indices = hook_indices + rest_indices
    num_sampled = len(sampled_indices)
    
    reader = get_reader()
    print(f"Starting OCR on {num_sampled} frames using EasyOCR...")

    # Phase 1: Sequential Pre-pass to identify identical frames
    similarity_threshold = getattr(Config, 'OCR_SIMILARITY_THRESHOLD', 0.01) if Config else 0.01
    prev_img_small = None
    frame_cache = {}
    tasks_to_run = []

    for idx in sampled_indices:
        item = frames_to_process[idx]
        if isinstance(item, str):
            img = cv2.imread(item)
        else:
            img = item
            
        if img is None: continue
        
        height, width = img.shape[:2]
        frame_area = float(width * height)
        if frame_area == 0: continue
        
        frame_cache[idx] = {'area': frame_area, 'should_run_ocr': True}
        
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_small = cv2.resize(img_gray, (64, 64))
            
            if prev_img_small is not None:
                diff = np.mean(np.abs(img_small.astype("float32") - prev_img_small.astype("float32"))) / 255.0
                if diff < similarity_threshold:
                    frame_cache[idx]['should_run_ocr'] = False
            
            if frame_cache[idx]['should_run_ocr']:
                prev_img_small = img_small
        except Exception:
            pass

        if frame_cache[idx]['should_run_ocr']:
            if img.shape[1] > 800:
                scale_ocr = 800 / img.shape[1]
                img_ocr = cv2.resize(img, (800, int(img.shape[0] * scale_ocr)))
            else:
                img_ocr = img
                
            tasks_to_run.append((idx, img_ocr, reader))

    # Phase 2: Sequential OCR Inference (Thread pool removed for PyTorch safety)
    ocr_results = {}
    for task in tasks_to_run:
        idx, img_ocr, r = task
        try:
            res = r.readtext(img_ocr) if r else []
        except Exception as e:
            print(f"OCR Exception on frame {idx}: {e}")
            res = []
        ocr_results[idx] = res

    # Phase 3: Sequential processing of OCR results for temporal metrics
    frames_with_text_count = 0
    total_characters_detected = 0
    total_relative_box_area = 0.0
    total_text_boxes_count = 0
    total_words_count = 0
    valid_words_count = 0
    hook_text_frames_count = 0
    min_word_length = getattr(Config, 'OCR_MIN_WORD_LENGTH', 3) if Config else 3
    
    reading_speeds = []
    prev_frame_words = set()
    time_delta = sample_rate / target_fps
    safe_time_delta = max(0.1, time_delta)
    
    prev_ocr_result = []

    for idx in sampled_indices:
        if idx not in frame_cache:
            continue
            
        fc = frame_cache[idx]
        frame_area = fc['area']
        
        if fc['should_run_ocr']:
            result = ocr_results.get(idx, [])
            prev_ocr_result = result
        else:
            result = prev_ocr_result
            
        has_valid_text = False
        frame_char_count = 0
        current_frame_words = set()

        for (bbox, text, prob) in result:
            if prob < 0.4: continue
            clean_text = text.strip()
            if not clean_text: continue
            if len(bbox) < 4: continue
            
            has_valid_text = True
            frame_char_count += len(clean_text)
            
            tl, tr, br, bl = bbox[0], bbox[1], bbox[2], bbox[3]
            # OPTIMISED MATH: direct absolute differences instead of np array instantiations
            box_w = abs(tr[0] - tl[0])
            box_h = abs(bl[1] - tl[1])
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

    # Results Formulation
    ideal_area = getattr(Config, 'OCR_FONT_IDEAL_AREA', 0.10) if Config else 0.10
    avg_relative_area = (total_relative_box_area / total_text_boxes_count) if total_text_boxes_count > 0 else 0.0

    res = {
        "_cache_version": _CACHE_VERSION,
        "text_presence_ratio": frames_with_text_count / num_sampled if num_sampled else 0,
        "text_density": total_characters_detected / num_sampled if num_sampled else 0,
        "font_size_score": min(100, (avg_relative_area / ideal_area * 100) if ideal_area > 0 else 0),
        "context_clarity": valid_words_count / total_words_count if total_words_count else 0,
        "hook_text_ratio": hook_text_frames_count / sum(1 for x in sampled_indices if x < hook_frames_limit) if any(x < hook_frames_limit for x in sampled_indices) else 0,
        "reading_speed": float(np.mean(reading_speeds)) if reading_speeds else 0.0,
        "motion_score": 0.0  # Placeholder for future implementation. Currently never computed.
    }

    if frame_folder and os.path.exists(frame_folder):
         try:
            cache_path = os.path.join(frame_folder, "text_features_cache.json")
            with open(cache_path, 'w') as f: json.dump(res, f)
         except: pass
        
    return res


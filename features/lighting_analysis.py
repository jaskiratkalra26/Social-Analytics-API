import os
import sys
import numpy as np
import cv2
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config
from pipeline import video_loader
from pipeline import frame_extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_lighting(video_path: str, frame_folder: str = None, frames: list = None) -> dict:
    """
    Analyzes the lighting quality of a video.
    """
    
    # ... Metadata loading ...
    if not os.path.exists(video_path):
        return {"error": "Video not found"}
        
    # Standard metadata loading... (omitted for brevity in prompt, assume kept or unrelated to frames)
    
    # --- 2. Frame Extraction (Sampling) ---
    logger.info("Computing Lighting Features...")
    
    processing_frames = []
    
    if frames is not None and len(frames) > 0:
        # User provided frames in memory
        # We need to sample them if there are too many, to simulate the logic of "sampling_fps"
        # Config.LIGHTING_SAMPLE_COUNT = 15
        target_count = getattr(Config, 'LIGHTING_SAMPLE_COUNT', 15)
        total_frames = len(frames)
        
        if total_frames > target_count:
            step = total_frames // target_count
            processing_frames = frames[::step]
            # Ensure we don't exceed target count too much
            processing_frames = processing_frames[:target_count]
        else:
            processing_frames = frames
            
    elif frame_folder and os.path.exists(frame_folder):
        # Fallback to loading from disk
        all_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')])
        target_count = getattr(Config, 'LIGHTING_SAMPLE_COUNT', 15)
        if len(all_files) > target_count:
            step = len(all_files) // target_count
            processing_frames = all_files[::step][:target_count]
        else:
            processing_frames = all_files
    else:
        # Should not happen if pipeline is correct
        return {"error": "No frames available"}

    # --- 3. Compute Lighting Features ---
    brightness_means = []
    contrasts = []
    dark_pixel_ratios = []
    bright_pixel_ratios = []

    for item in processing_frames:
        if isinstance(item, str):
            img = cv2.imread(item)
        else:
            img = item
            
        if img is None: continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Average Brightness
        mean_brightness = np.mean(gray)
        brightness_means.append(mean_brightness)
        
        # 3. Contrast (std dev of pixel values)
        contrast = np.std(gray)
        contrasts.append(contrast)
        
        # Total pixels
        total_pixels = gray.size
        
        # 4. Dark Pixel Ratio (< 30)
        dark_pixels = np.sum(gray < Config.LIGHTING_DARK_PIXEL_THRESH)
        dark_ratio = dark_pixels / total_pixels
        dark_pixel_ratios.append(dark_ratio)
        
        # 5. Bright Pixel Ratio (> 225)
        bright_pixels = np.sum(gray > Config.LIGHTING_BRIGHT_PIXEL_THRESH)
        bright_ratio = bright_pixels / total_pixels
        bright_pixel_ratios.append(bright_ratio)

    if not brightness_means:
        return {"error": "Could not process any frames"}

    # Aggregate Metrics
    avg_brightness = float(np.mean(brightness_means))
    # 2. Brightness Variance (variance across frames)
    brightness_variance = float(np.var(brightness_means))
    avg_contrast = float(np.mean(contrasts))
    avg_dark_ratio = float(np.mean(dark_pixel_ratios))
    avg_bright_ratio = float(np.mean(bright_pixel_ratios))

    metrics = {
        "avg_brightness": round(avg_brightness, 2),
        "brightness_variance": round(brightness_variance, 2),
        "contrast": round(avg_contrast, 2),
        "dark_pixel_ratio": round(avg_dark_ratio, 2),
        "bright_pixel_ratio": round(avg_bright_ratio, 2)
    }

    # --- 4. Rule-Based Analysis ---
    issues = []
    suggestions = []
    score_penalties = 0

    # Constants from Config
    THRESH_LOW_BRIGHT = Config.LIGHTING_LOW_BRIGHTNESS
    THRESH_HIGH_BRIGHT = Config.LIGHTING_HIGH_BRIGHTNESS # for good/over boundary? 
    # Use explicit overexposure threshold 200 from prompt/config
    THRESH_OVEREXPOSED = Config.LIGHTING_OVEREXPOSED_THRESHOLD
    
    THRESH_LOW_CONTRAST = Config.LIGHTING_CONTRAST_LOW
    THRESH_HIGH_CONTRAST = Config.LIGHTING_CONTRAST_HIGH
    
    THRESH_DARK_RATIO = Config.LIGHTING_DARK_RATIO_MAX
    THRESH_BRIGHT_RATIO = Config.LIGHTING_BRIGHT_RATIO_MAX
    THRESH_VARIANCE = Config.LIGHTING_VARIANCE_THRESHOLD

    # Rule 1: Low Lighting
    if avg_brightness < THRESH_LOW_BRIGHT:
        issues.append("Low lighting / underexposed video")
        suggestions.append("Increase front lighting or record in a brighter environment")
        score_penalties += 25

    # Rule 2: Overexposed Video
    if avg_brightness > THRESH_OVEREXPOSED:
        issues.append("Overexposed lighting")
        suggestions.append("Reduce exposure or move light source further away")
        score_penalties += 25
        
    # Rule 3: Flat Lighting
    if avg_contrast < THRESH_LOW_CONTRAST:
        issues.append("Flat lighting with low depth")
        suggestions.append("Add side lighting to create contrast and depth")
        score_penalties += 15
        
    # Rule 4: Harsh Lighting
    if avg_contrast > THRESH_HIGH_CONTRAST:
        issues.append("Harsh lighting with strong shadows")
        suggestions.append("Use diffused lighting or soft lighting setup")
        score_penalties += 15
        
    # Rule 5: Shadow Clipping
    if avg_dark_ratio > THRESH_DARK_RATIO:
        issues.append("Large shadow regions detected")
        suggestions.append("Add fill light to reduce shadow areas")
        score_penalties += 10
        
    # Rule 6: Highlight Clipping
    if avg_bright_ratio > THRESH_BRIGHT_RATIO:
        issues.append("Overexposed highlights detected")
        suggestions.append("Reduce direct lighting or lower camera exposure")
        score_penalties += 10
        
    # Rule 7: Inconsistent Lighting
    if brightness_variance > THRESH_VARIANCE:
        issues.append("Lighting flicker or inconsistent exposure")
        suggestions.append("Use stable light sources and avoid mixed lighting")
        score_penalties += 10

    # --- 5. Score Calculation ---
    final_score = max(0, 100 - score_penalties)
    
    # --- 6. Categorization ---
    if final_score > Config.LIGHTING_SCORE_EXCELLENT:
        category = "excellent"
    elif final_score >= Config.LIGHTING_SCORE_GOOD:
        category = "good" 
    elif final_score >= Config.LIGHTING_SCORE_POOR:
        category = "needs_improvement"
    else:
        category = "poor"

    return {
        "lighting_score": final_score,
        "lighting_category": category,
        "issues_detected": issues,
        "improvement_suggestions": suggestions,
        "lighting_metrics": metrics
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        v_path = sys.argv[1]
        print(json.dumps(analyze_lighting(v_path), indent=4))
    else:
        print("Usage: python lighting_analysis.py <video_path>")

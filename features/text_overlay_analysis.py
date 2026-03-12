import os
import sys

# Add project root to sys.path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config
from features.text_features import extract_text_features

def analyze_text_overlay(frame_folder: str) -> dict:
    """
    Analyzes text overlay metrics from video frames to determine readability,
    clutter, and effectiveness.

    Args:
        frame_folder (str): Path to the folder containing extracted video frames.

    Returns:
        dict: A dictionary containing the text quality score, category, detected issues,
              improvement suggestions, and the original text metrics.
    """
    
    # Check if frame folder exists
    if not os.path.exists(frame_folder):
        return {
            "text_score": 0,
            "text_category": "error",
            "issues_detected": ["Frame folder not found"],
            "improvement_suggestions": [],
            "text_metrics": {}
        }

    try:
        # Get OCR metrics
        # Setting verbose=False as per requirements for production/API usage
        text_metrics = extract_text_features(frame_folder, verbose=False)
    except Exception as e:
        return {
            "text_score": 0,
            "text_category": "error",
            "issues_detected": [f"Error extracting text features: {str(e)}"],
            "improvement_suggestions": [],
            "text_metrics": {}
        }

    # Initialize analysis variables
    score = 100
    issues = []
    suggestions = []

    # Extract metrics for easier access
    # Defaulting to 0 if key is missing to avoid crashes, though extract_text_features should return them
    text_presence_ratio = text_metrics.get("text_presence_ratio", 0)
    text_density = text_metrics.get("text_density", 0)
    font_size_score = text_metrics.get("font_size_score", 0)
    context_clarity = text_metrics.get("context_clarity", 0)
    hook_text_ratio = text_metrics.get("hook_text_ratio", 0)
    reading_speed = text_metrics.get("reading_speed", 0)
    motion_score = text_metrics.get("motion_score", 0)


    # --- Rule Engine ---

    # Fallback: No Text Detected / Negligible Text
    # If practically no text is found (less than 5% presence), we treat it as missing text
    # to avoid false positives on small noise artifacts being flagged as "small font".
    if text_presence_ratio < 0.05:
        score = 50  # Start at 'Needs Improvement' level given social video expectations
        issues.append("Very little to no text overlay detected")
        suggestions.append("Consider adding captions or text overlays to improve engagement")
        
        # Apply Hook Text Penalty if missing (which it is likely)
        if hook_text_ratio < 0.1: # Threshold internal to logic if not in Config
            # We already penalized score start, but ensure hook message is clear
            if "Weak hook text" not in issues: # Avoid dups if logic changes
                issues.append("Weak hook text presence in the opening frames")
                suggestions.append("Add a strong hook text in the first few seconds")
            
    else:
        # Rule 1 — Text Too Small
        # Only check size if we have enough confident text to measure
        if font_size_score < 0.01: # Check against hard lower bound for noise
             # If score is extremely low despite presence > 0.05, it might still be noise
             # But if it is real text, it is indeed too small.
             pass

        if font_size_score < Config.FONT_SIZE_LOW:
            score -= Config.TEXT_PENALTY_SMALL_FONT
            issues.append("Text too small to read on most screens")
            suggestions.append("Increase font size or enlarge text overlay")

        # Rule 2 — Low Context Clarity
        if context_clarity < Config.CONTEXT_CLARITY_LOW:
            score -= Config.TEXT_PENALTY_LOW_CLARITY
            issues.append("OCR detected many unclear or fragmented words")
            suggestions.append("Use simpler fonts and avoid decorative typography")

        # Rule 3 — Too Much Text (Clutter)
        if text_density > Config.TEXT_DENSITY_HIGH:
            score -= Config.TEXT_PENALTY_HIGH_DENSITY
            issues.append("Too much text on screen causing clutter")
            suggestions.append("Reduce the amount of text per frame")

        # Rule: Text Moving Too Fast (WPM)
        if reading_speed > Config.READING_SPEED_HIGH:
            score -= Config.TEXT_PENALTY_FAST_TEXT
            issues.append(f"Text content changes too fast ({int(reading_speed)} WPM) to be easily read")
            suggestions.append("Increase duration of text overlays or reduce text amount per screen")

        # Rule: Text Scrolling Too Fast
        if motion_score > Config.MOTION_SCORE_HIGH:
            score -= Config.TEXT_PENALTY_FAST_SCROLL
            issues.append(f"Text scrolling speed is too high (Magnitude: {motion_score:.2f})")
            suggestions.append("Reduce scrolling speed by 20–30% for better readability")

        # Rule 4 — Too Little Text
        # Only if Rule 5 doesn't apply (mutually exclusive concepts usually, but logic allows both check)
        if text_presence_ratio < Config.TEXT_PRESENCE_LOW:
            score -= Config.TEXT_PENALTY_LOW_PRESENCE
            issues.append("Very little text appears throughout the video")
            suggestions.append("Add text overlays to highlight important information")

        # Rule 5 — Excessive Text Overlays
        if text_presence_ratio > Config.TEXT_PRESENCE_HIGH:
            score -= Config.TEXT_PENALTY_HIGH_PRESENCE
            issues.append("Text appears in most frames which may overwhelm viewers")
            suggestions.append("Reduce constant text overlays")

        # Rule 6 — Weak Hook Text
        if hook_text_ratio < Config.HOOK_TEXT_RATIO_LOW:
            score -= Config.TEXT_PENALTY_WEAK_HOOK
            issues.append("Weak hook text presence in the opening frames")
            suggestions.append("Add a strong hook text in the first few seconds")

    # Clamp score
    score = max(0, min(100, score))

    # Determine Category
    if score > Config.TEXT_SCORE_EXCELLENT:
        category = "excellent"
    elif score >= Config.TEXT_SCORE_GOOD:
        category = "good"
    elif score >= Config.TEXT_SCORE_POOR:
        category = "needs_improvement"
    else:
        category = "poor"

    return {
        "text_score": float(score),
        "text_category": category,
        "issues_detected": issues,
        "improvement_suggestions": suggestions,
        "text_metrics": text_metrics
    }

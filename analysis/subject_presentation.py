import Config

def compute_subject_presentation(features: dict) -> dict:
    """
    Evaluates how well the subject appears in a video based on visual metrics.
    
    Args:
        features: Dictionary containing:
            - face_ratio: float
            - avg_face_size: float
            - face_centering: float
            - face_sharpness: float
            - face_brightness: float
            
    Returns:
        Dictionary with:
        - presentation_score: float (0-100)
        - issues: list[str]
        - suggestions: list[str]
    """
    
    # Check if we have any valid face data at all
    # If face_ratio is 0, practically nothing else matters much, but we need to handle it safely
    face_ratio = features.get("face_ratio", 0.0)
    avg_face_size = features.get("avg_face_size", 0.0)
    face_centering = features.get("face_centering", 0.0)
    face_sharpness = features.get("face_sharpness", 0.0)
    face_brightness = features.get("face_brightness", 0.0)
    
    # Get Config Values
    IDEAL_SIZE_MIN = getattr(Config, 'SUBJECT_IDEAL_SIZE_MIN', 0.02)
    IDEAL_SIZE_MAX = getattr(Config, 'SUBJECT_IDEAL_SIZE_MAX', 0.25)
    SHARPNESS_REF = getattr(Config, 'SUBJECT_SHARPNESS_THRESHOLD', 150.0)
    BRIGHT_MIN = getattr(Config, 'SUBJECT_BRIGHTNESS_MIN', 80)
    BRIGHT_MAX = getattr(Config, 'SUBJECT_BRIGHTNESS_MAX', 180)
    
    W_RATIO = getattr(Config, 'SUBJECT_SCORE_RATIO_WEIGHT', 0.30)
    W_SIZE = getattr(Config, 'SUBJECT_SCORE_SIZE_WEIGHT', 0.20)
    W_CENTER = getattr(Config, 'SUBJECT_SCORE_CENTERING_WEIGHT', 0.15)
    W_SHARP = getattr(Config, 'SUBJECT_SCORE_SHARPNESS_WEIGHT', 0.20)
    W_BRIGHT = getattr(Config, 'SUBJECT_SCORE_BRIGHTNESS_WEIGHT', 0.15)

    issues = []
    suggestions = []

    # --- 1. Edge Case: No Face Detected ---
    if face_ratio == 0:
        issues.append("No visible subject detected")
        suggestions.append("Ensure the subject is clearly visible in the video")
        return {
            "presentation_score": 0.0,
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {
                "face_ratio": 0.0,
                "avg_face_size": 0.0,
                "face_centering": 0.0,
                "face_sharpness": 0.0,
                "face_brightness": 0.0
            }
        }

    # --- 2. Feature Normalization ---

    # A. Face Size Score (Framing)
    if IDEAL_SIZE_MIN <= avg_face_size <= IDEAL_SIZE_MAX:
        face_size_score = 1.0
    elif avg_face_size < IDEAL_SIZE_MIN:
        # Ramp up
        face_size_score = avg_face_size / IDEAL_SIZE_MIN
    else:
        # Ramp down for too close
        face_size_score = max(0.0, 1.0 - (avg_face_size - IDEAL_SIZE_MAX))

    # B. Sharpness Score
    sharpness_score = min(face_sharpness / SHARPNESS_REF, 1.0)

    # C. Brightness Score
    if BRIGHT_MIN <= face_brightness <= BRIGHT_MAX:
        brightness_score = 1.0
    elif face_brightness < BRIGHT_MIN:
        brightness_score = face_brightness / float(BRIGHT_MIN)
    else:
        # Too bright
        brightness_score = max(0.0, 1.0 - (face_brightness - float(BRIGHT_MAX)) / float(BRIGHT_MAX))


    # --- 3. Compute Weighted Score ---
    score = (
        W_RATIO * face_ratio +
        W_SIZE * face_size_score +
        W_CENTER * face_centering +
        W_SHARP * sharpness_score +
        W_BRIGHT * brightness_score
    )
    
    presentation_score = score * 100.0


    # --- 4. Detect Issues & Suggestions ---
    
    if face_ratio < 0.5:
        issues.append("Subject appears in too few frames")
        suggestions.append("Ensure the subject is visible throughout the video")

    if avg_face_size < IDEAL_SIZE_MIN:
        # Only report if we actually have faces but they are small
        issues.append("Subject is too far from the camera")
        suggestions.append("Move closer to the camera for better framing")

    if face_centering < 0.5:
        issues.append("Subject is not centered in the frame")
        suggestions.append("Position the subject closer to the center of the frame")

    if face_sharpness < (SHARPNESS_REF * 0.4): # e.g. < 60
        issues.append("Video appears blurry")
        suggestions.append("Ensure the camera is properly focused")

    if face_brightness < (BRIGHT_MIN - 10): # e.g. < 70
        issues.append("Lighting on the subject is too dark")
        suggestions.append("Improve lighting or face a light source")


    return {
        "presentation_score": round(presentation_score, 1),
        "issues": issues,
        "suggestions": suggestions,
        "metrics": {
            "face_ratio": round(face_ratio, 2),
            "avg_face_size": round(avg_face_size, 3),
            "face_centering": round(face_centering, 2),
            "face_sharpness": round(face_sharpness, 1),
            "face_brightness": round(face_brightness, 1)
        }
    }

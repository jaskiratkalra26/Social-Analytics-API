def compute_storytelling_clarity(metrics: dict) -> dict:
    """
    Computes a clarity score for the video narrative based on pre-calculated metrics.
    
    Args:
        metrics (dict): A dictionary containing storytelling-related metrics:
            - pace_score
            - motion_flow_score
            - text_support_score
            - scene_consistency_score
            - text_readability_score
            - hook_text_ratio
            
    Returns:
        dict: A dictionary containing:
            - clarity_score (float): 0-100 score
            - issues (list[str]): List of detected issues
            - suggestions (list[str]): List of improvement suggestions
    """
    
    # Step 1: Extract Metrics
    # Using .get() with 0.0 defaults to handle missing metrics safely
    pace_score = metrics.get("pace_score", 0.0)
    pace_category = metrics.get("pace_category", "optimal")
    motion_flow_score = metrics.get("motion_flow_score", 0.0)
    text_support_score = metrics.get("text_support_score", 0.0)
    scene_consistency_score = metrics.get("scene_consistency_score", 0.0)
    text_readability_score = metrics.get("text_readability_score", 0.0)
    hook_text_ratio = metrics.get("hook_text_ratio", 0.0)

    issues = []
    suggestions = []

    # Step 2: Compute Clarity Score
    clarity_score = (
        0.30 * pace_score +
        0.20 * motion_flow_score +
        0.20 * text_support_score +
        0.15 * scene_consistency_score +
        0.10 * text_readability_score +
        0.05 * hook_text_ratio
    )

    # Scale to 0-100
    clarity_score = clarity_score * 100

    # Step 3: Detect Issues

    # Pacing problems
    if pace_score < 0.5:
        if pace_category == "too_fast":
             issues.append("Video pacing is inconsistent or too fast")
             suggestions.append("Reduce scene transition speed for better storytelling flow")
        elif pace_category == "slow":
             issues.append("Video pacing is too slow or static")
             suggestions.append("Increase scene transitions to maintain interest")
        elif pace_category == "optimal" and pace_score < 40:
             issues.append("Pacing is optimal but visual engagement is low")
             suggestions.append("Add more motion, text, or visual variety")
        else:
             issues.append("Video pacing is inconsistent")
             suggestions.append("Ensure consistent scene transitions for better storytelling flow")

    # Low motion
    if motion_flow_score < 0.4:
        issues.append("Video appears visually static")
        suggestions.append("Add more visual variation or camera movement")

    # Lack of supporting text
    if text_support_score < 0.3:
        issues.append("Lack of on-screen explanation")
        suggestions.append("Add captions or text highlights to support the story")

    # Scene inconsistency
    if scene_consistency_score < 0.4:
        issues.append("Scene transitions appear inconsistent")
        suggestions.append("Maintain more consistent shot durations")

    # Poor text readability (Score < 15 means extremely small text)
    if text_readability_score < 0.15:
        issues.append("Text overlays are difficult to read (too small)")
        suggestions.append("Increase font size for better readability")

    # Weak introduction
    if hook_text_ratio < 0.1:
        issues.append("The introduction does not clearly explain the topic")
        suggestions.append("Add explanatory text in the first few seconds")

    # Step 4: Severity Check
    if clarity_score < 40:
        issues.append("Overall storytelling clarity is poor")
        suggestions.append("Improve pacing and add clearer explanations")

    # Step 5: Limit Output Size
    # Keep only the first 3 issues/suggestions to remain concise
    issues = issues[:3]
    suggestions = suggestions[:3]

    # Step 6: Return Result
    return {
        "clarity_score": round(clarity_score, 2),
        "issues": issues,
        "suggestions": suggestions
    }

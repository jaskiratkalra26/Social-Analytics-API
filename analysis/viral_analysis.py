import logging
import Config

def compute_viral_analysis(metrics: dict) -> dict:
    """
    Evaluates how well a video matches common viral patterns.
    
    Args:
        metrics (dict): Dictionary containing normalized scores (0-1):
            - hook_score
            - pace_score
            - motion_flow_score
            - text_support_score
            - subject_presentation_score
            - storytelling_clarity_score
            
    Returns:
        dict: Viral analysis results including score, patterns, issues, and suggestions.
    """
    
    # Step 1: Extract Metrics
    hook_score = metrics.get("hook_score", 0.0)
    pace_score = metrics.get("pace_score", 0.0)
    pace_category = metrics.get("pace_category", "")  # Extract category
    motion_flow_score = metrics.get("motion_flow_score", 0.0)
    text_support_score = metrics.get("text_support_score", 0.0)
    subject_presentation_score = metrics.get("subject_presentation_score", 0.0)
    storytelling_clarity_score = metrics.get("storytelling_clarity_score", 0.0)

    patterns = []
    issues = []
    suggestions = []

    # Step 2: Compute Viral Score
    viral_score = (
        Config.VIRAL_WEIGHT_HOOK * hook_score +
        Config.VIRAL_WEIGHT_PACE * pace_score +
        Config.VIRAL_WEIGHT_MOTION * motion_flow_score +
        Config.VIRAL_WEIGHT_SUBJECT * subject_presentation_score +
        Config.VIRAL_WEIGHT_CLARITY * storytelling_clarity_score +
        Config.VIRAL_WEIGHT_TEXT * text_support_score
    )

    viral_score = viral_score * 100

    # Step 3: Detect Viral Patterns
    if hook_score > Config.VIRAL_THRESHOLD_HOOK_STRONG:
        patterns.append("Hook-driven structure")

    if pace_score > Config.VIRAL_THRESHOLD_PACE_FAST:
        patterns.append("Fast-paced editing")

    if text_support_score > Config.VIRAL_THRESHOLD_TEXT_HEAVY:
        patterns.append("Caption-driven storytelling")

    if subject_presentation_score > Config.VIRAL_THRESHOLD_SUBJECT_FOCUS:
        patterns.append("Face-focused content")

    if motion_flow_score > Config.VIRAL_THRESHOLD_MOTION_HIGH:
        patterns.append("High visual engagement")

    # Step 4: Detect Issues + Suggestions
    if hook_score < Config.VIRAL_ISSUE_HOOK_WEAK:
        issues.append("Weak opening hook")
        suggestions.append("Start with a strong attention-grabbing hook")

    if pace_score < Config.VIRAL_ISSUE_PACE_SLOW:
        if pace_category == "too_fast":
             issues.append("Pacing is too fast")
             suggestions.append("Slow down scene transitions to allow information absorption")
        elif pace_category == "optimal" and pace_score < 0.4:
            # Optimal category but low numerical score often means low motion or low text interaction
            issues.append("Pacing is technically optimal but engagement is low")
            suggestions.append("Increase visual variety or text overlays to boost engagement score")
        else:
             issues.append("Slow or inconsistent pacing")
             suggestions.append("Increase scene transitions for better engagement")

    if text_support_score < Config.VIRAL_ISSUE_TEXT_LOW:
        issues.append("Lack of supporting text")
        suggestions.append("Add captions to explain key points")

    if motion_flow_score < Config.VIRAL_ISSUE_MOTION_LOW:
        issues.append("Low visual engagement")
        suggestions.append("Add more motion or scene variation")

    if subject_presentation_score < Config.VIRAL_ISSUE_SUBJECT_POOR:
        issues.append("Poor subject presentation")
        suggestions.append("Ensure subject is clearly visible and well framed")

    # Step 5: Overall Low Score Check
    if viral_score < Config.VIRAL_ISSUE_SCORE_LOW:
        issues.append("Overall viral potential is low")
        suggestions.append("Improve hook, pacing, and engagement elements")

    # Step 6: Limit Output Size
    patterns = patterns[:3]
    issues = issues[:3]
    suggestions = suggestions[:3]

    # Step 7: Return Result
    return {
        "viral_score": round(viral_score, 2),
        "patterns": patterns,
        "issues": issues,
        "suggestions": suggestions
    }

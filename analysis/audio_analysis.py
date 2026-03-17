import Config

def compute_audio_analysis(features: dict) -> dict:

    # Extract features safely
    audio_energy = features.get("audio_energy", 0)
    beat_strength = features.get("beat_strength", 0)
    tempo_bpm = features.get("tempo_bpm", 0)
    speech_clarity = features.get("speech_clarity", 0)
    hook_audio_intensity = features.get("hook_audio_intensity", 0)
    spectrogram_variance = features.get("spectrogram_variance", 0)
    mfcc_variance = features.get("mfcc_variance", 0)

    # Normalize values to 0-1 range for scoring
    # Heuristic normalization based on typical ranges
    norm_audio_energy = min(audio_energy / Config.AUDIO_NORM_ENERGY, 1.0) # RMS often low
    norm_beat_strength = min(beat_strength / Config.AUDIO_NORM_BEAT, 1.0) # Onset often < 5
    norm_speech_clarity = min(speech_clarity / Config.AUDIO_NORM_CLARITY, 1.0) # Ratio often < 3
    norm_hook_intensity = min(hook_audio_intensity / Config.AUDIO_NORM_HOOK, 1.0)
    norm_spec_variance = min(spectrogram_variance / Config.AUDIO_NORM_SPEC, 1.0) # Guessing scale
    norm_mfcc_variance = min(mfcc_variance / Config.AUDIO_NORM_MFCC, 1.0) # Variance of MFCCs can be high

    patterns = []
    issues = []
    suggestions = []

    # Compute audio score
    audio_score = (
        Config.AUDIO_SCORE_WEIGHT_ENERGY * norm_audio_energy +
        Config.AUDIO_SCORE_WEIGHT_BEAT * norm_beat_strength +
        Config.AUDIO_SCORE_WEIGHT_CLARITY * norm_speech_clarity +
        Config.AUDIO_SCORE_WEIGHT_HOOK * norm_hook_intensity +
        Config.AUDIO_SCORE_WEIGHT_SPEC * norm_spec_variance +
        Config.AUDIO_SCORE_WEIGHT_MFCC * norm_mfcc_variance
    ) * 100

    # Detect patterns
    if beat_strength > Config.AUDIO_THRESH_BEAT_HIGH:
        patterns.append("Music-driven engagement")

    if speech_clarity > Config.AUDIO_THRESH_CLARITY_HIGH:
        patterns.append("Speech-focused content")

    # Only flag "High-energy" if there is actually audio energy present
    if tempo_bpm > Config.AUDIO_THRESH_TEMPO_HIGH and audio_energy >= Config.AUDIO_THRESH_ENERGY_LOW:
        patterns.append("High-energy audio")

    if hook_audio_intensity > Config.AUDIO_THRESH_HOOK_HIGH:
        patterns.append("Strong audio hook")
    
    # Detect issues and suggestions
    if audio_energy < Config.AUDIO_THRESH_ENERGY_LOW: # Adjusted threshold for raw RMS (typically 0.0 ~ 0.2)
        issues.append("Audio lacks energy")
        suggestions.append("Increase voice intensity or add background music")

    if speech_clarity < Config.AUDIO_THRESH_CLARITY_LOW:
        issues.append("Speech is not clear")
        suggestions.append("Improve microphone quality")

    if hook_audio_intensity < Config.AUDIO_THRESH_HOOK_LOW: # Adjusted for raw RMS
        issues.append("Weak audio in the beginning")
        suggestions.append("Start with stronger voice or sound")

    if spectrogram_variance < Config.AUDIO_THRESH_SPEC_LOW: # Adjusted placeholder
        issues.append("Audio lacks variation")
        suggestions.append("Add variation in tone or background sound")

    # Limit output
    patterns = patterns[:3]
    issues = issues[:3]
    suggestions = suggestions[:3]

    return {
        "audio_score": round(audio_score, 2),
        "patterns": patterns,
        "issues": issues,
        "suggestions": suggestions
    }

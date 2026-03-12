import librosa
import numpy as np
import os

def extract_audio_features(audio_path):
    """
    Extracts audio features from a WAV file using librosa.
    
    Args:
        audio_path (str): Path to the WAV audio file.
        
    Returns:
        dict: Dictionary containing calculated audio features.
              Returns zeros if file is empty or too short to analyze.
              
    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
    try:
        # Load audio file
        # sr=None ensures we keep original sampling rate, usually 44.1kHz or 16kHz
        y, sr = librosa.load(audio_path, sr=None)
        
        # Check if audio is empty
        if len(y) == 0:
            return _get_zero_features()
            
    except Exception:
        # If loading fails (e.g. empty file or corrupt), return zeros
        return _get_zero_features()

    # 1. Energy Level (Mean RMS)
    try:
        # Compute RMS energy (root-mean-square)
        rms = librosa.feature.rms(y=y)
        audio_energy = float(np.mean(rms))
    except Exception:
        audio_energy = 0.0

    # 2. Beat Strength (Onset Strength)
    try:
        # Compute onset envelope to track percussive events
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = float(np.mean(onset_env))
    except Exception:
        beat_strength = 0.0

    # 3. Tempo (BPM)
    try:
        # Estimate global tempo
        # beat_track returns tempo and beat frames. We only need tempo.
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa 0.10+ returns tempo as a scalar or array depending on version/settings
        # Ensure it's a float
        tempo_bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
    except Exception:
        tempo_bpm = 0.0

    # 4. Speech Clarity Approximation
    # approximated as Mean RMS / (Std Dev RMS + epsilon)
    # Higher value -> More consistent energy (likely clear speech vs noisy/music)
    try:
        if 'rms' not in locals():
            rms = librosa.feature.rms(y=y)
        
        mean_rms = np.mean(rms)
        std_rms = np.std(rms)
        small_constant = 1e-6
        
        speech_clarity = float(mean_rms / (std_rms + small_constant))
    except Exception:
        speech_clarity = 0.0

    # 5. First 3 Seconds Audio Intensity (Hook Strength)
    try:
        # Determine duration in samples for 3 seconds
        duration_sec = 3
        duration_samples = duration_sec * sr
        
        # Slice the audio array for efficiency instead of reloading file
        y_hook = y[:duration_samples]
        
        if len(y_hook) > 0:
            hook_rms = librosa.feature.rms(y=y_hook)
            hook_audio_intensity = float(np.mean(hook_rms))
        else:
            hook_audio_intensity = 0.0
    except Exception:
        hook_audio_intensity = 0.0

    # 6. Spectrogram Features
    spectrogram_dict = spectrogram_features(y, sr)

    # 7. MFCC Features
    mfcc_dict = mfcc_features(y, sr)

    features = {
        "audio_energy": audio_energy,
        "beat_strength": beat_strength,
        "tempo_bpm": tempo_bpm,
        "speech_clarity": speech_clarity,
        "hook_audio_intensity": hook_audio_intensity
    }
    
    features.update(spectrogram_dict)
    features.update(mfcc_dict)
    
    return features

def spectrogram_features(y, sr):
    """
    Computes statistical features from the magnitude spectrogram.
    """
    try:
        # Compute Short-Time Fourier Transform (STFT)
        stft = librosa.stft(y)
        spectrogram = np.abs(stft)
        
        return {
            "spectrogram_mean": float(np.mean(spectrogram)),
            "spectrogram_variance": float(np.var(spectrogram))
        }
    except Exception:
        return {"spectrogram_mean": 0.0, "spectrogram_variance": 0.0}

def mfcc_features(y, sr):
    """
    Computes statistical features from Mel-frequency cepstral coefficients (MFCCs).
    """
    try:
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            "mfcc_mean": float(np.mean(mfcc)),
            "mfcc_variance": float(np.var(mfcc))
        }
    except Exception:
        return {"mfcc_mean": 0.0, "mfcc_variance": 0.0}

def _get_zero_features():
    """Helper to return zero-filled dictionary on error/empty input."""
    return {
        "audio_energy": 0.0,
        "beat_strength": 0.0,
        "tempo_bpm": 0.0,
        "speech_clarity": 0.0,
        "hook_audio_intensity": 0.0,
        "spectrogram_mean": 0.0,
        "spectrogram_variance": 0.0,
        "mfcc_mean": 0.0,
        "mfcc_variance": 0.0
    }

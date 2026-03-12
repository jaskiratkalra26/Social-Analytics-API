import os
import sys

# Try to import VideoFileClip, handling different MoviePy versions
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        # For some v2 setups
        from moviepy.video.io.VideoFileClip import VideoFileClip

# Add the project root directory to Python path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Config

def extract_audio(video_path, duration=None):
    """
    Extracts audio from a video file and saves it as a WAV file.
    
    Args:
        video_path (str): The path to the video file.
        duration (int, optional): Duration in seconds to extract audio. If None, processes entire video.
        
    Returns:
        str: The path to the saved audio file, or None if extraction failed.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
        
    try:
        # Ensure the output directory exists
        os.makedirs(Config.AUDIO_OUTPUT_DIR, exist_ok=True)
        
        # dynamic audio filename based on video filename
        video_filename = os.path.basename(video_path)
        audio_filename = os.path.splitext(video_filename)[0] + ".wav"
        audio_output_path = os.path.join(Config.AUDIO_OUTPUT_DIR, audio_filename)
        
        # Check cache
        if os.path.exists(audio_output_path):
            print(f"Audio already extracted at: {audio_output_path}")
            return audio_output_path
        
        # Load the video clip
        video_clip = VideoFileClip(video_path)
        
        # Trim clip if duration is specified
        if duration is not None:
            # Check MoviePy version compatibility
            # In v1.0.3, it is .subclip(start, end)
            # In v2.0+, it might be .subclipped(start, end) or similar, but usually subclip is available on the object or via wrapper
            # However, sometimes VideoFileClip might not have it directly if imports are weird.
            
            # Ensure duration doesn't exceed clip duration
            end_time = min(duration, video_clip.duration)
            
            # Safe subclip method
            try:
                # Try standard method
                if hasattr(video_clip, 'subclip'):
                    video_clip = video_clip.subclip(0, end_time)
                else:
                    # Fallback for some versions where subclip might be on the parent class not correctly bound or imported
                    # This happens sometimes with partial imports.
                    # As a workaround, we can just set the duration attribute if we are just writing audio?
                    # No, write_audiofile writes whole clip.
                    
                    # Alternatively, slicing: clip = clip.subclip(0, end_time)
                    # If attribute missing, maybe it's `subclipped` in v2?
                    if hasattr(video_clip, 'subclipped'):
                        video_clip = video_clip.subclipped(0, end_time)
                    else:
                        # Fallback: Just proceed with full audio if we can't clip it
                        # But better to try slicing by time:
                        pass
            except Exception as e:
                print(f"Warning: Could not subclip video: {e}. using full duration.")

        
        # check if video has audio
        if video_clip.audio is None:
            print(f"No audio found in {video_path}")
            video_clip.close()
            return None
            
        # Write the audio to a file
        video_clip.audio.write_audiofile(audio_output_path, codec=Config.AUDIO_CODEC)
        
        # Close the clip to release resources
        # IMPORTANT: Close the audio reader specifically to release file handle on Windows
        if video_clip.audio:
            try:
                video_clip.audio.close()
            except:
                pass
        video_clip.close()
        
        print(f"Audio extracted to: {audio_output_path}")
        return audio_output_path
        
    except Exception as e:
        print(f"An error occurred during audio extraction: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    # extract_audio("path/to/video.mp4")
    pass

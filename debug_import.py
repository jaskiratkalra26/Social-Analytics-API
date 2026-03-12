import sys
import os
print("Starting script...")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from social_analytics_pipeline import analyze_video
    print("Imported analyze_video")
except Exception as e:
    print(f"Error importing: {e}")

import os
import sys
import json
import logging

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from social_analytics_pipeline import analyze_video

logging.basicConfig(level=logging.INFO)

video_path = os.path.join("test_videos", "video_03.mp4")
if not os.path.exists(video_path):
    print(f"{video_path} not found")
    sys.exit(1)

print(f"Testing single video: {video_path}")
results = analyze_video(video_path)
print("Analysis complete")
if "error" in results:
    print(f"Error: {results['error']}")
else:
    print("Success keys:", results.keys())

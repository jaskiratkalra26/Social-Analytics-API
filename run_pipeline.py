import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from social_analytics_pipeline import analyze_video

video_path = "Video_for_OCR_Model_Testing.mp4"
output_file = "output/social_analytics_results.json"

if not os.path.exists("output"):
    os.makedirs("output")

print(f"Running analysis on {video_path}...")
results = analyze_video(video_path)

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Analysis complete. Results saved to {output_file}")

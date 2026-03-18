import json
import sys
import os
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure project root is in python path so absolute imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime
from collections import Counter

from pipeline.clip_keyword_mapper import hybrid_map_keywords
from pipeline.youtube_trends_fetcher import fetch_trending_videos

# Initialize Redis client
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None

if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")

def build_trend_data():
    # Returns structured data now: [ { "title": "...", "youtube_category": "...", ... } ]
    video_items = fetch_trending_videos(region_code="IN", max_results=50)

    # Hybrid Categorization:
    # 1. Use YouTube Category if valid
    # 2. Use CLIP/Regex mapping (title + tags) if YT category is generic/missing
    
    final_categories = []
    processed_videos = []
    
    # Categories considered "Generic" or "Vague" that need refinement
    GENERIC_CATS = ["key", "Entertainment", "People & Blogs", "Howto & Style", "Nonprofits & Activism", "Life Hacks", "Lifestyle"]

    for item in video_items:
        # Check if item is a dict (real data) or string (simulation fallback)
        if not isinstance(item, dict):
             continue
             
        yt_cat = item.get("youtube_category", "Entertainment")
        
        # If the YT category is specific enough, use it
        if yt_cat and yt_cat not in GENERIC_CATS:
            final_cat = yt_cat
        else:
            # Otherwise, use our hybrid mapper on the title + tags for better accuracy
            # E.g. "Minecraft Gameplay" under "Entertainment" -> "Gaming"
            tags = item.get('tags', [])
            context_text = f"{item.get('title', '')} {' '.join(tags[:5] if tags else [])}"
            # map expecting a list
            mapped_list = hybrid_map_keywords([context_text])
            mapped = mapped_list[0] if mapped_list else None
            
            final_cat = mapped if mapped else yt_cat
            
        final_categories.append(final_cat)
        
        # Store for processing
        processed_videos.append({
            "title": item.get('title'),
            "category": final_cat
        })

    # Count frequency of CATEGORIES (not individual keywords)
    category_counts = Counter(final_categories)
    total = sum(category_counts.values())

    # Sort categories by frequency
    ranked = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    # Select top 3 categories
    top_categories = [cat for cat, _ in ranked[:3]]

    # Build result structure
    trend_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "top_categories": [],
    }

    for cat in top_categories:
        # Get videos belonging to this category
        cat_videos = [v["title"] for v in processed_videos if v["category"] == cat]

        score = category_counts[cat] / total if total > 0 else 0

        trend_data["top_categories"].append({
            "category": cat,
            "score": round(score, 3),
            "keywords": cat_videos[:5]  # top 5 video titles used as keywords
        })

    return trend_data

def save_trend_to_redis(data):
    if not redis_client:
        print("Redis client not initialized.")
        return False
    try:
        redis_client.set("trending_data", json.dumps(data))
        print("Trend data saved to Redis.")
        return True
    except Exception as e:
        print(f"Error saving to Redis: {e}")
        return False

def get_trend_from_redis():
    default_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "top_categories": [],
    }
    
    if not redis_client:
        print("Redis client not initialized.")
        return default_data

    try:
        data_str = redis_client.get("trending_data")
        if data_str:
            return json.loads(data_str)
        else:
            return default_data
    except Exception as e:
        print(f"Error reading from Redis: {e}")
        return default_data

def save_trend_data(data, path="output/trend_data.json"):
    # Ensure output directory exists (though workspace info says output/ exists)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def get_category_trend(category, trend_data=None):
    if trend_data is None:
        trend_data = get_trend_from_redis()

    for item in trend_data.get("top_categories", []):
        if item["category"] == category:
            return item

    return {
        "category": category,
        "score": 0,
        "keywords": [],
        "trend": "not trending"
    }

def main():
    try:
        trend_data = build_trend_data()
        
        # Save to Redis (Primary)
        save_trend_to_redis(trend_data)
        
        # Save to JSON (Backup)
        save_trend_data(trend_data)
        
        print("Trend data updated successfully using YouTube Data API!")
    except Exception as e:
        print(f"Error updating trend data: {e}")

if __name__ == "__main__":
    main()

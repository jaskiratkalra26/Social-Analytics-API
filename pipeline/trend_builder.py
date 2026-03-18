import json
import sys
import os
import redis
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime
from collections import Counter

from pipeline.clip_keyword_mapper import hybrid_map_keywords
from pipeline.youtube_trends_fetcher import fetch_trending_videos

from keybert import KeyBERT
import nltk
nltk.download('stopwords', quiet=True)

REDIS_URL = os.getenv("REDIS_URL")
redis_client = None

if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")

def build_trend_data():
    video_items = fetch_trending_videos(region_code="IN", max_results=50)

    final_categories = []
    video_keywords = []

    GENERIC_CATS = ["key", "Entertainment", "People & Blogs", "Howto & Style", "Nonprofits & Activism", "Life Hacks", "Lifestyle"]

    kw_model = KeyBERT()

    for item in video_items:
        if not isinstance(item, dict):
            continue

        yt_cat = item.get("youtube_category", "Entertainment")

        if yt_cat and yt_cat not in GENERIC_CATS:
            final_cat = yt_cat
        else:
            tags = item.get('tags', [])
            context_text = f"{item.get('title', '')} {' '.join(tags[:5] if tags else [])}"
            mapped_list = hybrid_map_keywords([context_text])
            mapped = mapped_list[0] if mapped_list else None
            final_cat = mapped if mapped else yt_cat

        final_categories.append(final_cat)

        title = item.get('title', '')
        tags = item.get('tags', [])
        text_for_keywords = title + ' ' + ' '.join(tags[:5] if tags else [])
        keywords = kw_model.extract_keywords(
            text_for_keywords,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=5
        )
        video_keywords.append({
            "category": final_cat,
            "keywords": [kw for kw, score in keywords]
        })

    category_counts = Counter(final_categories)
    total = sum(category_counts.values())
    top_categories = [cat for cat, _ in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]]

    trend_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "top_categories": [],
    }

    for cat in top_categories:
        cat_keywords = []
        for vkw in video_keywords:
            if vkw["category"] == cat:
                cat_keywords.extend(vkw["keywords"])

        keyword_counts = Counter(cat_keywords)
        trend_data["top_categories"].append({
            "category": cat,
            "score": round(category_counts[cat] / total, 3) if total > 0 else 0,
            "keywords": [kw for kw, _ in keyword_counts.most_common(5)]
        })

    return trend_data

def save_trend_to_redis(data):
    if not redis_client:
        print("Redis client not initialized.")
        return False
    try:
        redis_client.set("trending_data", json.dumps(data, ensure_ascii=False))
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
        return json.loads(data_str) if data_str else default_data
    except Exception as e:
        print(f"Error reading from Redis: {e}")
        return default_data

def save_trend_data(data, path="output/trend_data.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_category_trend(category, trend_data=None):
    if trend_data is None:
        trend_data = get_trend_from_redis()
    for item in trend_data.get("top_categories", []):
        if item["category"] == category:
            return item
    return {"category": category, "score": 0, "keywords": [], "trend": "not trending"}

def main():
    try:
        trend_data = build_trend_data()
        save_trend_to_redis(trend_data)
        save_trend_data(trend_data)
        print("Trend data updated successfully using YouTube Data API!")
    except Exception as e:
        print(f"Error updating trend data: {e}")

if __name__ == "__main__":
    main()
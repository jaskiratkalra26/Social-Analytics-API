try:
    from googleapiclient.discovery import build
    import Config
except ImportError:
    print("Warning: google-api-python-client not installed.")
    build = None

def fetch_trending_videos(region_code="IN", max_results=50):
    """
    Fetches trending videos from YouTube API v3.
    Returns: List of dictionaries containing video metadata (title, tags, category_id, etc.)
    """
    api_key = getattr(Config, 'YOUTUBE_API_KEY', None)
    
    if not api_key or api_key == 'YOUR_API_KEY':
        print("Warning: YouTube API Key not found in Config.py. Returning simulation data.")
        return get_simulation_data()

    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        request = youtube.videos().list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=max_results
        )
        response = request.execute()
        
        video_items = []
        for item in response.get('items', []):
            snippet = item['snippet']
            title = snippet.get('title', '')
            description = snippet.get('description', '')
            tags = snippet.get('tags', [])
            category_id = snippet.get('categoryId', '')
            
            # Map YouTube Category ID to actionable label
            yt_category = YOUTUBE_CATEGORY_MAP.get(str(category_id), "Entertainment")
            
            video_items.append({
                "title": title,
                "description": description[:200], # Trucate description
                "tags": tags[:5], # Top 5 tags
                "category_id": category_id,
                "youtube_category": yt_category
            })
                
        return video_items

    except Exception as e:
        print(f"YouTube API Error: {e}")
        return get_simulation_data()

# https://developers.google.com/youtube/v3/docs/videoCategories/list
YOUTUBE_CATEGORY_MAP = {
    '1': 'Movies & TV',
    '2': 'Automotive',
    '10': 'Music',
    '15': 'Animals',
    '17': 'Sports',
    '18': 'Movies & TV',
    '19': 'Travel',
    '20': 'Gaming',
    '22': 'Lifestyle', # People & Blogs
    '23': 'Comedy',
    '24': 'Entertainment',
    '25': 'News & Politics',
    '26': 'Life Hacks', # Howto & Style
    '27': 'Education',
    '28': 'Science', # Science & Technology
    '29': 'Activism',
    '30': 'Movies & TV',
    '31': 'Anime/Animation',
    '44': 'Movies & TV' # Trailers
}

def get_simulation_data():
    # Simulation data as last resort (Structured)
    return [
        {"title": "IPL 2026 Finals Highlights", "tags": ["cricket", "ipl"], "youtube_category": "Sports"},
        {"title": "iPhone 17 Review", "tags": ["tech", "apple"], "youtube_category": "Science"}, 
        {"title": "Latest Bollywood Trailer", "tags": ["movie", "trailer"], "youtube_category": "Movies & TV"},
        {"title": "Funny Cat Fails", "tags": ["cats", "funny"], "youtube_category": "Animals"},
        {"title": "Stock Market Crash", "tags": ["finance", "money"], "youtube_category": "News & Politics"}
    ]


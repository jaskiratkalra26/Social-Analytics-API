from typing import List, Dict, Union

# Data extracted from hashtags.txt
# Structure: Category -> List of (Hashtag, Usage %)
POPULAR_HASHTAGS_DB = {
    "Gaming": [
        {"hashtag": "#gaming", "usage": "35%"},
        {"hashtag": "#gamer", "usage": "13%"},
        {"hashtag": "#playstation", "usage": "7%"},
        {"hashtag": "#videogames", "usage": "7%"},
        {"hashtag": "#game", "usage": "6%"},
        {"hashtag": "#games", "usage": "6%"},
        {"hashtag": "#xbox", "usage": "5%"},
        {"hashtag": "#twitch", "usage": "4%"},
        {"hashtag": "#gamingcommunity", "usage": "3%"}
    ],
    "Education": [
        {"hashtag": "#education", "usage": "61%"},
        {"hashtag": "#learning", "usage": "7%"},
        {"hashtag": "#school", "usage": "6%"},
        {"hashtag": "#students", "usage": "4%"},
        {"hashtag": "#study", "usage": "3%"},
        {"hashtag": "#motivation", "usage": "3%"},
        {"hashtag": "#student", "usage": "3%"},
        {"hashtag": "#love", "usage": "3%"},
        {"hashtag": "#teacher", "usage": "3%"},
        {"hashtag": "#college", "usage": "2%"}
    ],
    "Technology": [
        {"hashtag": "#technology", "usage": "52%"},
        {"hashtag": "#tech", "usage": "14%"},
        {"hashtag": "#innovation", "usage": "6%"},
        {"hashtag": "#business", "usage": "5%"},
        {"hashtag": "#iphone", "usage": "4%"},
        {"hashtag": "#engineering", "usage": "4%"},
        {"hashtag": "#technews", "usage": "3%"},
        {"hashtag": "#science", "usage": "3%"},
        {"hashtag": "#software", "usage": "3%"},
        {"hashtag": "#gadgets", "usage": "3%"}
    ],
    "Finance": [
        {"hashtag": "#finance", "usage": "39%"},
        {"hashtag": "#money", "usage": "10%"},
        {"hashtag": "#business", "usage": "10%"},
        {"hashtag": "#investing", "usage": "7%"},
        {"hashtag": "#financialfreedom", "usage": "6%"},
        {"hashtag": "#investment", "usage": "6%"},
        {"hashtag": "#entrepreneur", "usage": "5%"},
        {"hashtag": "#stockmarket", "usage": "4%"},
        {"hashtag": "#trading", "usage": "4%"},
        {"hashtag": "#stocks", "usage": "4%"}
    ],
    "Fitness": [
        {"hashtag": "#fitness", "usage": "40%"},
        {"hashtag": "#gym", "usage": "10%"},
        {"hashtag": "#workout", "usage": "8%"},
        {"hashtag": "#fitnessmotivation", "usage": "8%"},
        {"hashtag": "#fit", "usage": "7%"},
        {"hashtag": "#motivation", "usage": "6%"},
        {"hashtag": "#bodybuilding", "usage": "5%"},
        {"hashtag": "#training", "usage": "4%"},
        {"hashtag": "#health", "usage": "4%"},
        {"hashtag": "#fitfam", "usage": "3%"}
    ],
    "Cooking": [
        {"hashtag": "#cooking", "usage": "33%"},
        {"hashtag": "#food", "usage": "13%"},
        {"hashtag": "#foodie", "usage": "9%"},
        {"hashtag": "#foodporn", "usage": "7%"},
        {"hashtag": "#instafood", "usage": "7%"},
        {"hashtag": "#foodphotography", "usage": "5%"},
        {"hashtag": "#foodlover", "usage": "5%"},
        {"hashtag": "#foodblogger", "usage": "5%"},
        {"hashtag": "#foodstagram", "usage": "5%"},
        {"hashtag": "#homemade", "usage": "5%"}
    ],
    "Travel": [
        {"hashtag": "#travel", "usage": "42%"},
        {"hashtag": "#travelphotography", "usage": "9%"},
        {"hashtag": "#photography", "usage": "8%"},
        {"hashtag": "#nature", "usage": "8%"},
        {"hashtag": "#love", "usage": "6%"},
        {"hashtag": "#travelgram", "usage": "6%"},
        {"hashtag": "#instagood", "usage": "5%"},
        {"hashtag": "#photooftheday", "usage": "5%"},
        {"hashtag": "#adventure", "usage": "3%"},
        {"hashtag": "#instagram", "usage": "3%"}
    ],
    "Music": [
        {"hashtag": "#music", "usage": "54%"},
        {"hashtag": "#love", "usage": "7%"},
        {"hashtag": "#hiphop", "usage": "6%"},
        {"hashtag": "#rap", "usage": "5%"},
        {"hashtag": "#musician", "usage": "4%"},
        {"hashtag": "#art", "usage": "4%"},
        {"hashtag": "#artist", "usage": "4%"},
        {"hashtag": "#musica", "usage": "3%"},
        {"hashtag": "#rock", "usage": "3%"}
    ],
    "Comedy": [
        {"hashtag": "#comedy", "usage": "43%"},
        {"hashtag": "#funny", "usage": "13%"},
        {"hashtag": "#memes", "usage": "9%"},
        {"hashtag": "#meme", "usage": "6%"},
        {"hashtag": "#funnymemes", "usage": "6%"},
        {"hashtag": "#lol", "usage": "4%"},
        {"hashtag": "#humor", "usage": "4%"},
        {"hashtag": "#love", "usage": "4%"},
        {"hashtag": "#fun", "usage": "4%"},
        {"hashtag": "#viral", "usage": "3%"}
    ],
    "News": [
        {"hashtag": "#news", "usage": "65%"},
        {"hashtag": "#instagram", "usage": "4%"},
        {"hashtag": "#viral", "usage": "4%"},
        {"hashtag": "#trending", "usage": "4%"},
        {"hashtag": "#india", "usage": "4%"},
        {"hashtag": "#breakingnews", "usage": "4%"},
        {"hashtag": "#media", "usage": "3%"},
        {"hashtag": "#newsupdate", "usage": "3%"},
        {"hashtag": "#love", "usage": "3%"},
        {"hashtag": "#politics", "usage": "2%"}
    ],
    "Sports": [
        {"hashtag": "#sports", "usage": "50%"},
        {"hashtag": "#football", "usage": "10%"},
        {"hashtag": "#sport", "usage": "7%"},
        {"hashtag": "#fitness", "usage": "6%"},
        {"hashtag": "#basketball", "usage": "5%"},
        {"hashtag": "#nfl", "usage": "5%"},
        {"hashtag": "#nba", "usage": "4%"},
        {"hashtag": "#soccer", "usage": "3%"},
        {"hashtag": "#baseball", "usage": "3%"},
        {"hashtag": "#gym", "usage": "3%"}
    ],
    "Beauty": [
        {"hashtag": "#beauty", "usage": "45%"},
        {"hashtag": "#makeup", "usage": "7%"},
        {"hashtag": "#love", "usage": "7%"},
        {"hashtag": "#beautiful", "usage": "7%"},
        {"hashtag": "#fashion", "usage": "7%"},
        {"hashtag": "#skincare", "usage": "5%"},
        {"hashtag": "#instagood", "usage": "4%"},
        {"hashtag": "#photography", "usage": "4%"},
        {"hashtag": "#model", "usage": "4%"},
        {"hashtag": "#style", "usage": "4%"}
    ]
}

def get_popular_hashtags_by_category(category: str) -> List[Dict[str, str]]:
    """
    Returns a list of popular hashtags and their usage percentage for a given category.

    Args:
        category (str): The category to retrieve hashtags for (e.g., "Gaming", "Education").
                        Case-insensitive.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'hashtag' and 'usage'.
                              Returns an empty list if category is not found.
    """
    if not category:
        return []
        
    # Normalize category
    normalized_category = category.strip().title()
    
    # Handle direct mapping of special cases/typos if needed
    if normalized_category == "Eduction":
        normalized_category = "Education"
        
    return POPULAR_HASHTAGS_DB.get(normalized_category, [])

if __name__ == "__main__":
    # Test execution
    test_cats = ["Gaming", "Education", "Music", "NonExistent"]
    for cat in test_cats:
        print(f"--- {cat} ---")
        results = get_popular_hashtags_by_category(cat)
        for item in results:
            print(f"{item['hashtag']} ({item['usage']})")
        print()

import torch
import open_clip

# --- Step 2: Category Mapping ---
LABEL_TO_ID = {
    'Beauty & Fashion': 0,
    'Comedy': 1,
    'Cooking & Food': 2,
    'Education': 3,
    'Finance': 4,
    'Fitness & Health': 5,
    'Gaming': 6,
    'Music': 7,
    'News & Politics': 8,
    'Sports': 9,
    'Technology': 10,
    'Travel': 11,
    'Movies & TV': 12,
    'Automotive': 13,
    'Science': 14,
    'Animals': 15,
    'Spirituality': 16,
    'Relationships': 17
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# --- Step 3: Create Descriptive Category Prompts ---
CATEGORY_PROMPTS = [
    "beauty makeup fashion style and clothing",
    "funny comedy jokes memes and pranks",
    "cooking food recipes delicious meals and baking",
    "educational tutorials learning study and history",
    "finance investing money stock market and business",
    "fitness workout gym yoga health and wellness",
    "gaming esports gameplay minecraft pubg and consoles",
    "music songs video concerts singers and lyrics",
    "news politics government world events and breaking news",
    "sports cricket football athlete matches and tournaments",
    "technology gadgets smartphones AI computers and coding",
    "travel vlogs tourism vacation destinations and nature",
    "movies cinema trailers bollywood hollywood actors and tv shows",
    "automotive cars bikes racing vehicles and driving",
    "science space physics nature biology and experiments",
    "cute animals pets dogs cats and wildlife",
    "spirituality god religion devotion and meditation",
    "relationships dating love couples and advice"
]

# --- Step 4: Load CLIP Model (Initialize Once) ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

# --- Step 5: Precompute Category Embeddings (Global) ---
def compute_category_embeddings():
    tokens = tokenizer(CATEGORY_PROMPTS).to(device)

    with torch.no_grad():
        embeddings = model.encode_text(tokens)

    # Normalize
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings

CATEGORY_EMBEDDINGS = compute_category_embeddings()

# --- Step 6: Encode Keywords in Batch ---
def encode_keywords(keywords):
    tokens = tokenizer(keywords).to(device)

    with torch.no_grad():
        embeddings = model.encode_text(tokens)

    # Normalize
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings

# --- Step 7: Map Keywords → Categories (Batch Version) ---
def map_keywords_to_categories(keywords):
    if not keywords:
        return []

    # Get keyword embeddings
    keyword_embeddings = encode_keywords(keywords)

    # Compute similarity between keywords and all categories (cosine similarity = dot product of normalized vectors)
    similarities = keyword_embeddings @ CATEGORY_EMBEDDINGS.T

    # Get index of best match
    best_indices = similarities.argmax(dim=1).cpu().numpy()

    mapped = [ID_TO_LABEL[int(k)] for k in best_indices]

    return mapped

import re

# --- Step 8: Optional Hybrid Rules (Fast + Accurate) ---
def hybrid_map_keywords(keywords):
    results = []
    
    # Expanded rule-based system for faster matching
    # Using specific terms to avoid false positives (e.g. "ai" matching "trailer")
    keyword_rules = {
        "Sports": ["ipl", "cricket", "football", "match", "athlete", "score", "tennis", "wrestling", "sport"],
        "Technology": ["ai", "chatgpt", "tech", "gadget", "mobile", "iphone", "software", "coding", "samsung", "pixel", "laptop", "computer", "android", "ios"],
        "Fitness & Health": ["gym", "weight loss", "workout", "yoga", "health", "diet", "bodybuilding", "exercise", "fitness"],
        "Cooking & Food": ["recipe", "food", "kitchen", "cook", "cake", "baking", "dinner", "pizza", "burger", "curry", "street food"],
        "Travel": ["vlog", "trip", "tour", "visit", "bali", "paris", "vacation", "flight", "hotel", "travel"],
        "Finance": ["money", "invest", "stock", "crypto", "trading", "bitcoin", "market", "economy", "finance", "business"],
        "Beauty & Fashion": ["makeup", "skin", "hair", "fashion", "style", "lipstick", "dress", "clothing", "outfit", "grwm"],
        "Gaming": ["game", "gameplay", "minecraft", "pubg", "cod", "esports", "ps5", "xbox", "fortnite", "roblox", "gta"],
        "Comedy": ["funny", "meme", "prank", "standup", "laugh", "comedy", "joke", "skit"],
        "Music": ["song", "lyrics", "concert", "singer", "band", "music video", "rapper", "dj", "remix", "lofi"],
        "Education": ["learn", "tutorial", "study", "tips", "how to", "class", "exam", "history", "science course", "lesson"],
        "News & Politics": ["news", "update", "report", "breaking", "politics", "minister", "election", "vote", "modi", "congress"],
        "Movies & TV": ["trailer", "teaser", "movie", "film", "cinema", "actor", "actress", "series", "episode", "season", "netflix", "prime video", "full movie"],
        "Automotive": ["car", "bike", "racing", "drive", "vehicle", "scooter", "tata", "mahindra", "maruti", "toyota"],
        "Science": ["space", "nasa", "physics", "chemistry", "biology", "planet", "experiment", "science"],
        "Animals": ["cat", "dog", "pet", "animal", "cute", "wildlife", "puppy", "kitten", "zoo"],
        "Spirituality": ["god", "prayer", "temple", "devotional", "bhajan", "mantra", "meditation", "spiritual"],
        "Relationships": ["love", "dating", "couple", "relationship", "marriage", "crush", "wedding"]
    }

    # Pre-compile regex patterns for efficiency
    # Matches whole words only: \b(word1|word2|...)\b
    # Note: Special handling for "ai" which is very short
    compiled_rules = {}
    for category, rules in keyword_rules.items():
        # Escape rules to be safe in regex
        escaped_rules = [re.escape(r) for r in rules]
        pattern_str = r'\b(' + '|'.join(escaped_rules) + r')\b'
        compiled_rules[category] = re.compile(pattern_str, re.IGNORECASE)

    # First pass: use regex matching
    for kw in keywords:
        match_found = None
        
        # Priority Check: First check generic rules
        # You might want to prioritize specific categories if needed
        # For now, we iterate in order
        for category, pattern in compiled_rules.items():
            if pattern.search(kw):
                match_found = category
                break
        
        results.append(match_found)

    # Identify keywords that need CLIP mapping
    remaining_indices = [i for i, r in enumerate(results) if r is None]
    remaining_keywords = [keywords[i] for i in remaining_indices]

    if remaining_keywords:
        mapped_categories = map_keywords_to_categories(remaining_keywords)

        # Fill in the gaps
        for idx, category in zip(remaining_indices, mapped_categories):
            results[idx] = category

    return results

# --- Step 9: Test ---
if __name__ == "__main__":
    test_keywords = [
        "weight loss tips",
        "ipl match today",
        "chatgpt update",
        "chicken recipe",
        "travel vlog bali",
        "how to invest in stocks",
        "funny cat videos",
        "make money online"
    ]

    print("Running hybrid mapping test...")
    categories = hybrid_map_keywords(test_keywords)

    for k, c in zip(test_keywords, categories):
        print(f"{k} -> {c}")

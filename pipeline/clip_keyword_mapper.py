import torch
import open_clip

# --- Step 2: Category Mapping ---
LABEL_TO_ID = {
    'Beauty': 0,
    'Comedy': 1,
    'Cooking': 2,
    'Education': 3,
    'Finance': 4,
    'Fitness': 5,
    'Gaming': 6,
    'Music': 7,
    'News': 8,
    'Sports': 9,
    'Technology': 10,
    'Travel': 11
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# --- Step 3: Create Descriptive Category Prompts ---
CATEGORY_PROMPTS = [
    "beauty and makeup tutorial content",
    "funny comedy entertainment content",
    "cooking recipes and food preparation",
    "educational and learning content",
    "finance investing and money advice",
    "fitness workout and health content",
    "gaming gameplay and esports content",
    "music songs and performance content",
    "news and current events reporting",
    "sports cricket football matches",
    "technology gadgets and AI content",
    "travel vlogs and tourism content"
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

# --- Step 8: Optional Hybrid Rules (Fast + Accurate) ---
def hybrid_map_keywords(keywords):
    results = []
    
    # Expanded rule-based system for faster matching
    keyword_rules = {
        "Sports": ["ipl", "cricket", "football", "match", "athlete", "score"],
        "Technology": ["ai", "chatgpt", "tech", "gadget", "mobile", "iphone", "software", "coding"],
        "Fitness": ["gym", "weight loss", "workout", "yoga", "health", "diet"],
        "Cooking": ["recipe", "food", "kitchen", "cook", "cake", "baking", "dinner"],
        "Travel": ["vlog", "trip", "tour", "visit", "bali", "paris", "vacation"],
        "Finance": ["money", "invest", "stock", "crypto", "trading", "bitcoin"],
        "Beauty": ["makeup", "skin", "hair", "fashion", "style", "lipstick"],
        "Gaming": ["game", "gameplay", "minecraft", "pubg", "cod", "esports", "ps5"],
        "Comedy": ["funny", "meme", "prank", "standup", "laugh", "comedy"],
        "Music": ["song", "lyrics", "concert", "singer", "band", "music video"],
        "Education": ["learn", "tutorial", "study", "tips", "how to", "class"],
        "News": ["news", "update", "report", "breaking", "politics"]
    }

    # First pass: use fast string matching rules
    for kw in keywords:
        k = kw.lower()
        match_found = None
        
        for category, rules in keyword_rules.items():
            if any(rule in k for rule in rules):
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

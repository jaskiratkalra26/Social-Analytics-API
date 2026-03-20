import os

# Project Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output Directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
METADATA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'metadata')
FRAMES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'frames')
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'audio')
AUDIO_CODEC = 'pcm_s16le'

# Scene Detection
SCENE_THRESHOLD = 30.0
SCENE_DOWNSCALE_FACTOR = 8
SCENE_SHOW_PROGRESS = False

# Frame Extraction
TARGET_FPS = 1
FRAME_MAX_WIDTH = 640  # Resize frames to this width for speed
OCR_SAMPLE_RATE = 3  # Process every 3rd captured frame for speed
OCR_SIMILARITY_THRESHOLD = 0.01  # Skip OCR if frame differs by less than 1% from previous

# Pacing Analysis
PACING_SHORT_VIDEO_THRESHOLD = 90

# Short Video Pacing Thresholds (Cuts per Minute)
PACING_SHORT_MIN_CUTS = 8
PACING_SHORT_MAX_CUTS = 25

# Long Video Pacing Thresholds (Cuts per Minute)
PACING_LONG_MIN_CUTS = 2
PACING_LONG_MAX_CUTS = 10

# Pacing Score Normalization (approx max values)
PACING_NORM_CUTS = 30.0
PACING_NORM_MOTION = 10.0
PACING_NORM_VARIANCE = 10.0

# Pacing Score Weights
PACING_WEIGHT_CUTS = 0.4
PACING_WEIGHT_MOTION = 0.3
PACING_WEIGHT_VARIANCE = 0.2
PACING_WEIGHT_TEXT = 0.1

# Lighting Analysis
LIGHTING_SAMPLE_COUNT = 15  # Target number of frames to sample

# Hook Analysis Configuration
HOOK_DURATION = 3

# YouTube Data API Config
import os
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")  # Loads from env if set, else fallback
HOOK_MIN_MOTION = 0.3

# Subject Presentation Analysis
SUBJECT_SCORE_RATIO_WEIGHT = 0.30
SUBJECT_SCORE_SIZE_WEIGHT = 0.20
SUBJECT_SCORE_CENTERING_WEIGHT = 0.15
SUBJECT_SCORE_SHARPNESS_WEIGHT = 0.20
SUBJECT_SCORE_BRIGHTNESS_WEIGHT = 0.15

SUBJECT_IDEAL_SIZE_MIN = 0.02
SUBJECT_IDEAL_SIZE_MAX = 0.25
SUBJECT_SHARPNESS_THRESHOLD = 150.0
SUBJECT_BRIGHTNESS_MIN = 80
SUBJECT_BRIGHTNESS_MAX = 180

HOOK_MIN_TEXT_RATIO = 0.3
HOOK_MIN_AUDIO_ENERGY = 0.1

# Hook Penalties
HOOK_PENALTY_LOW_MOTION = 20
HOOK_PENALTY_STATIC_SCENE = 15
HOOK_PENALTY_WEAK_TEXT = 20
HOOK_PENALTY_LOW_AUDIO = 15
HOOK_PENALTY_NO_SPEECH = 10

# Hook Score Categories
HOOK_SCORE_STRONG = 80
HOOK_SCORE_MODERATE = 60
HOOK_SCORE_WEAK = 40

# Lighting Thresholds
LIGHTING_LOW_BRIGHTNESS = 60.0
LIGHTING_HIGH_BRIGHTNESS = 180.0
LIGHTING_OVEREXPOSED_THRESHOLD = 200.0

LIGHTING_CONTRAST_LOW = 25.0
LIGHTING_CONTRAST_HIGH = 90.0

LIGHTING_DARK_PIXEL_THRESH = 30
LIGHTING_DARK_RATIO_MAX = 0.20

LIGHTING_BRIGHT_PIXEL_THRESH = 225
LIGHTING_BRIGHT_RATIO_MAX = 0.15

LIGHTING_VARIANCE_THRESHOLD = 50.0

# Lighting Categorization
LIGHTING_SCORE_EXCELLENT = 80
LIGHTING_SCORE_GOOD = 60
LIGHTING_SCORE_POOR = 40

# Text Overlay Analysis - Thresholds
TEXT_PRESENCE_LOW = 0.05
TEXT_PRESENCE_HIGH = 0.80

TEXT_DENSITY_HIGH = 120

FONT_SIZE_LOW = 0.005
FONT_SIZE_GOOD_MIN = 0.005
FONT_SIZE_GOOD_MAX = 0.03
FONT_SIZE_HIGH = 0.08

CONTEXT_CLARITY_LOW = 0.4
CONTEXT_CLARITY_HIGH = 0.7

HOOK_TEXT_RATIO_LOW = 0.3
HOOK_TEXT_RATIO_HIGH = 0.5

# Text Overlay Analysis - Penalties
TEXT_PENALTY_SMALL_FONT = 25
TEXT_PENALTY_LOW_CLARITY = 20
TEXT_PENALTY_HIGH_DENSITY = 15
TEXT_PENALTY_LOW_PRESENCE = 10
TEXT_PENALTY_HIGH_PRESENCE = 10
TEXT_PENALTY_WEAK_HOOK = 10

# Text Overlay Analysis - Categories
TEXT_SCORE_EXCELLENT = 80
TEXT_SCORE_GOOD = 60
TEXT_SCORE_POOR = 40

# Text Reading Speed (Words Per Minute)
READING_SPEED_IDEAL = 130
READING_SPEED_HIGH = 150
TEXT_PENALTY_FAST_TEXT = 15

# Text Motion Score (0.0 - 1.0 Normalized movement per second)
# 0.2 - 0.5 is normal reading speed for scrolling
MOTION_SCORE_HIGH = 0.5
MOTION_SCORE_EXCESSIVE = 0.8
TEXT_PENALTY_FAST_SCROLL = 15


 

# Platform Recommendation - Thresholds & Scores

# General Video Type
VIDEO_TYPE_SHORT_THRESHOLD = 90

# TikTok Recommendation Logic
TIKTOK_HOOK_THRESHOLD = 70
TIKTOK_HOOK_BONUS = 30

TIKTOK_PACING_THRESHOLD = 70
TIKTOK_PACING_BONUS = 25

TIKTOK_DURATION_THRESHOLD_LOW = 30
TIKTOK_DURATION_BONUS = 20

TIKTOK_TEXT_THRESHOLD = 60
TIKTOK_TEXT_BONUS = 10

TIKTOK_PENALTY_THRESHOLD = 60
TIKTOK_PENALTY_VALUE = 10

# Instagram Reels Recommendation Logic
REELS_LIGHTING_THRESHOLD = 75
REELS_LIGHTING_BONUS = 25

REELS_TEXT_THRESHOLD = 70
REELS_TEXT_BONUS = 20

REELS_HOOK_THRESHOLD = 60
REELS_HOOK_BONUS = 20

REELS_PACING_RANGE_MIN = 50
REELS_PACING_RANGE_MAX = 70
REELS_PACING_BONUS = 15

# YouTube Shorts Recommendation Logic
SHORTS_DURATION_THRESHOLD = 30
SHORTS_DURATION_BONUS = 20

SHORTS_HOOK_THRESHOLD = 60
SHORTS_HOOK_BONUS = 20

SHORTS_TEXT_THRESHOLD = 60
SHORTS_TEXT_BONUS = 15

SHORTS_NICHE_EDUCATION_BONUS = 20

# Upload Time Recommendations (Category-Specific)
CATEGORY_UPLOAD_TIME = {
    "beauty": "6PM - 9PM",
    "comedy": "7PM - 11PM",
    "cooking": "11AM - 2PM",
    "education": "3PM - 6PM",
    "finance": "7AM - 9AM or 6PM - 8PM",
    "fitness": "6AM - 9AM",
    "gaming": "5PM - 9PM",
    "music": "6PM - 10PM",
    "news": "7AM - 9AM",
    "sports": "5PM - 9PM",
    "technology": "6PM - 9PM",
    "travel": "10AM - 2PM"
}



# Viral Analysis Weights
VIRAL_WEIGHT_HOOK = 0.30
VIRAL_WEIGHT_PACE = 0.20
VIRAL_WEIGHT_MOTION = 0.15
VIRAL_WEIGHT_SUBJECT = 0.15
VIRAL_WEIGHT_CLARITY = 0.10
VIRAL_WEIGHT_TEXT = 0.10

# Viral Pattern Thresholds (Normalized 0-1)
VIRAL_THRESHOLD_HOOK_STRONG = 0.7
VIRAL_THRESHOLD_PACE_FAST = 0.6
VIRAL_THRESHOLD_TEXT_HEAVY = 0.3
VIRAL_THRESHOLD_SUBJECT_FOCUS = 0.7
VIRAL_THRESHOLD_MOTION_HIGH = 0.6

# Viral Issue Thresholds (Normalized 0-1)
VIRAL_ISSUE_HOOK_WEAK = 0.4
VIRAL_ISSUE_PACE_SLOW = 0.5
VIRAL_ISSUE_TEXT_LOW = 0.2
VIRAL_ISSUE_MOTION_LOW = 0.4
VIRAL_ISSUE_SUBJECT_POOR = 0.5
VIRAL_ISSUE_SCORE_LOW = 40
# Audio Analysis Config
AUDIO_SCORE_WEIGHT_ENERGY = 0.25
AUDIO_SCORE_WEIGHT_BEAT = 0.15
AUDIO_SCORE_WEIGHT_CLARITY = 0.15
AUDIO_SCORE_WEIGHT_HOOK = 0.15
AUDIO_SCORE_WEIGHT_SPEC = 0.15
AUDIO_SCORE_WEIGHT_MFCC = 0.15

# Normalization Constants (Divisors)
AUDIO_NORM_ENERGY = 0.5
AUDIO_NORM_BEAT = 5.0
AUDIO_NORM_CLARITY = 3.0
AUDIO_NORM_HOOK = 0.5
AUDIO_NORM_SPEC = 100.0
AUDIO_NORM_MFCC = 2000.0

# Thresholds
AUDIO_THRESH_BEAT_HIGH = 0.6
AUDIO_THRESH_CLARITY_HIGH = 0.7
AUDIO_THRESH_TEMPO_HIGH = 120
AUDIO_THRESH_HOOK_HIGH = 0.6

AUDIO_THRESH_ENERGY_LOW = 0.05
AUDIO_THRESH_CLARITY_LOW = 0.5
AUDIO_THRESH_HOOK_LOW = 0.05
AUDIO_THRESH_SPEC_LOW = 0.1


# Subcategory ID Lookup
# Key   → actual subcategory name (from cluster_interpretations in pickle)
# Value → unique sequential integer ID (1–59)
SUBCATEGORY_IDS = {
    # beauty (1–5)
    "Skincare & Cosmetics Products":                    1,
    "Affordable & Professional Makeup Techniques":      2,
    "Indian & Social Media Makeup Trends":              3,
    "Beauty Horoscopes & E-commerce":                   4,
    "Nature-Inspired Beauty & Scenery":                 5,
    # comedy (6–7)
    "Viral Comedy Shorts & New Content":                6,
    "Relatable POV Skits & Social Media Memes":         7,
    # cooking (8–12)
    "Restaurant-Style Dinner Recipes":                  8,
    "Fun Foodie Shorts & Chicken Recipes":              9,
    "Trending Tasty Finds & Seafood":                  10,
    "Breakfast Vlogs & Creative Treats":               11,
    "Quick & Healthy Recipes":                         12,
    # education (13–20)
    "Personal Growth & Business Motivation":           13,
    "Psychological Facts & Quotes":                    14,
    "General Knowledge & History Facts":               15,
    "Tech Tools & Smart Gadgets":                      16,
    "Exam Prep (GK) & Motivation":                     17,
    "Science Projects & School Activities":            18,
    "English Language Learning":                       19,
    "Higher Education & Career Guidance":              20,
    # finance (21–25)
    "Professional & Personal Motivation":              21,
    "Wealth Building & Personal Income":               22,
    "Financial Literacy & Educational Resources":      23,
    "Stock Market Trading & Analysis":                 24,
    "Strategic Investing & Common Pitfalls":           25,
    # fitness (26–31)
    "Dance, Martial Arts & Rhythmic Exercise":         26,
    "Mixed Casual Fitness & Funny Content":            27,
    "Running, Sports & Outdoor Fitness":               28,
    "Gym Transformations & Evolution":                 29,
    "Heavy Strength Training & Bodybuilding":          30,
    "Desi Fitness & Calisthenics Challenges":          31,
    # music (32–37)
    "AI/Curated Music & Call to Action":               32,
    "Viral Dance & Short Music Videos":                33,
    "Ringtones & Status Videos (Mixed Language)":      34,
    "New Rap & Dance Music":                           35,
    "Live Music & Guitar Tutorials":                   36,
    "Telugu/Hindi Ringtones & Status":                 37,
    # news (38–42)
    "Motivational Quotes & Educational Inspiration":   38,
    "International Relations & Policy News":           39,
    "Sports News & Cricket Updates":                   40,
    "Social Stories & Human Interest Clips":           41,
    "Breaking News & Top Regional Headlines":          42,
    # sports (43–47)
    "Badminton & Racket Sports":                       43,
    "International Sports Highlights (Cricket)":       44,
    "Motivation & Combat Sports Training":             45,
    "Casual Team Sports & Social Media Trends":        46,
    "North American Collegiate & Ice Hockey":          47,
    # tech (48–53)
    "Mobile Ecosystems & OS":                          48,
    "Viral Gadgets & Gadget Compilations":             49,
    "Flagship Devices & Consumer Tech":                50,
    "Industrial Tech & Creative Engineering":          51,
    "AI & Future Technology":                          52,
    "Tech Education & Productivity Tools":             53,
    # travel (54–59)
    "Urban Views & International Cities":              54,
    "Nature Landscapes & Rail Travel":                 55,
    "Cultural Sites & Solo Exploration":               56,
    "Scenic Destinations & Beautiful Places":          57,
    "Travel Entertainment & Novelty Content":          58,
    "Romantic European Getaways":                      59,
    # gaming (60–67)
    "Minecraft Content":                                       60,
    "Competitive Shooters (Fortnite, Valorant, Freefire)":     61,
    "DOP Game Challenges & Puzzle Solutions":                  62,
    "Viral Gaming Content (Social Media Style)":               63,
    "Roblox War & Gameplay":                                   64,
    "Mobile Gaming Shorts & Horror":                           65,
    "GTA, Valorant, and Indian Driving Games":                 66,
    "Diverse Roblox & Meme-centric Gameplay":                  67,
}

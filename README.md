# Social Analytics API

A high-performance, FastAPI-based service designed to deeply analyze social media videos (TikTok, Reels, Shorts) and extract highly actionable insights. This machine learning pipeline processes visual, textual, and audio content to evaluate engagement potential, technical quality, platform suitability, category clusters, and real-time viral trends.

## 🚀 Key Features

The API processes uploaded videos through a comprehensive **multi-stage parallel architecture**, leveraging both CPU and GPU resources:

*   **Hook Analysis**: Evaluates the first 3 seconds of the video using visual motion and OCR text presence to determine how effectively it grabs attention.
*   **Pacing Analysis**: Measures cut rate (scenes per minute) and visual energy to score whether the video is too fast, too slow, or perfectly optimized.
*   **Audio Analysis**: Analyzes audio energy, beat strength, tempo (BPM), and speech clarity to determine the impact of sound on engagement.
*   **Lighting Analysis**: Evaluates video technical quality by measuring brightness, contrast, highlight/shadow clipping, and exposure consistency.
*   **Subject Presentation**: Uses computer vision to detect and evaluate how the primary subject (faces/people) is framed, centered, and lit.
*   **Storytelling Clarity**: A composite metric that evaluates narrative flow by combining pacing, motion flow, and on-screen text support.
*   **Viral Pattern Analysis**: Synthesizes all data points to calculate a "Viral Score" and identify structural patterns like "Hook-driven structure" or "Fast-paced editing."
*   **Optimized OCR (EasyOCR)**: Leverages highly-parallelised threads with **similarity-based frame skipping** to extract text without redundant processing.
*   **Subcategory Classification (ML)**: Uses pre-trained LightGBM embeddings to categorize content into hyper-specific niches.
*   **Competition Analysis**: Cross-references subcategories against a live Redis database (or JSON fallback) to evaluate market saturation.
*   **Real-time Trend Alignment**: Pulls daily viral lists from YouTube to check how well the video aligns with current trending topics.
*   **Platform Recommendation**: Analytically suggests the best platform (Instagram Reels, TikTok, YouTube Shorts) and optimal upload times.

## ⚡ Performance Optimization

This pipeline is built for speed and scalability:
-   **Parallel Execution**: Extraction stages (frames, audio, metadata, scenes) and Analysis modules run in parallel using `concurrent.futures`.
-   **In-Memory Processing**: Frames are stored and passed as memory objects where possible to minimize disk I/O.
-   **OCR Skipping**: Identifies identical or near-identical frames to skip redundant OCR inference, significantly reducing GPU/CPU load.
-   **Temporal Sampling**: Uniformly samples frames across long videos while ensuring every frame of the critical "hook" period is analyzed.

## 📂 Project Structure & Module Breakdown

```text
Social Analytics API/
├── api.py                      # FastAPI application entry point
├── social_analytics_pipeline.py # Main parallelized orchestrator
├── Config.py                   # Global configuration & constants
├── features/                   # Signal extraction (OCR, Vision, Audio)
├── analysis/                   # Rule-based & ML scoring modules
├── pipeline/                   # Extraction & data maintenance scripts
├── models/                     # Pre-trained ML model binaries
└── output/                     # API generated artifacts & fallbacks
```

### Detailed Breakdown

*   **`api.py`**: The main entry point for the FastAPI server. It handles file uploads, request validation, and returns the final JSON analysis results.
*   **`social_analytics_pipeline.py`**: The central orchestrator of the system. It manages the parallel execution of extraction and analysis modules using `ThreadPoolExecutor`.
*   **`Config.py`**: A centralized configuration file containing all thresholds, weights for scoring algorithms, and global paths.
*   **`.env`**: Stores sensitive credentials like `REDIS_URL` and `YOUTUBE_API_KEY`.

#### Core Directories

*   **`analysis/`**: Contains the "brain" of the system. These modules take raw features (like text, motion, or audio) and apply rule-based logic or ML models to generate scores (0-100), issues, and suggestions.
    *   `viral_analysis.py`: Aggregates all metrics into a final viral score.
    *   `lighting_analysis.py` & `subject_presentation.py`: Evaluate technical video quality.
    *   `subcategory_classification.py`: Uses LightGBM to map content to specific niches.
*   **`features/`**: Lower-level extraction scripts that interface with libraries like EasyOCR, CLIP, and Librosa to pull raw signal data from video and audio.
*   **`pipeline/`**: Handles the heavy lifting of data preparation:
    *   `frame_extractor.py` & `audio_extractor.py`: Interface with FFmpeg to pull assets.
    *   `scene_detector.py`: Uses `PySceneDetect` to find cut points.
    *   `trend_builder.py`: A maintenance script that syncs recent YouTube trends to Redis.
*   **`output/`**: Stores temporary frame/audio files during processing and maintains the `fallback/` directory for system operation when Redis is unavailable.
*   **`models/`**: Stores pre-trained ML models (e.g., LightGBM classifiers).

## 🛠️ Prerequisites

*   **Python 3.10+** (Recommend 3.11 for performance)
*   **Redis** (Used for real-time trending and competition data)
*   **CUDA (Optional)**: Highly recommended for faster OCR and CLIP inference.

## 📦 Installation

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/social-analytics-api.git
    cd "Social Analytics API"
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate # Mac/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables (`.env`)**
    ```env
    REDIS_URL="redis://localhost:6379/0"
    YOUTUBE_API_KEY="your_api_key_here"
    ```

## 🚀 Usage

### 1. Database Initialization
Populate the subcategory clusters and trending topics to Redis (and generate fallbacks):
```bash
python pipeline/competition_pipeline.py
python pipeline/trend_builder.py
```

### 2. Start the API Server
```bash
uvicorn api:app --reload
```

### 3. Analyze a Video
Interactive documentation is available at `http://127.0.0.1:8000/docs`. To analyze via `curl`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/analyze' \
  -H 'Content-Type: multipart/form-data' \
  -F 'video=@path/to/your/video.mp4'
```

## ⚙️ Configuration
Fine-tune analysis sensitivity in `Config.py`:
-   `OCR_MAX_FRAMES`: Maximum frames to process via OCR (default: 60).
-   `TARGET_FPS`: Capture rate for extraction (default: 1.0).
-   `HOOK_DURATION`: Duration considered for hook analysis (default: 3.0s).
-   `VIRAL_WEIGHT_*`: Adjust the importance of different metrics in the final viral score.

## 🔄 Maintenance Schedule
-   **`pipeline/trend_builder.py`**: Run daily to keep YouTube trend data fresh.
-   **`pipeline/competition_pipeline.py`**: Run weekly to update market saturation metrics.

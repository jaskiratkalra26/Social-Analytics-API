# Social Analytics API

A powerful FastAPI-based service designed to analyze social media videos (TikTok, Reels, Shorts) and extract actionable insights. This tool processes video content to evaluate engagement potential, technical quality, and platform suitability.

## 🚀 Features

The API processes uploaded videos through a multi-stage pipeline to provide:

*   **Metadata Extraction**: Basic video properties (duration, resolution, fps, bitrate).
*   **Hook Analysis**: Evaluates the first few seconds of the video to determine how well it grabs attention.
*   **Pacing Analysis**: Helper metrics like cut rate (scenes per minute) and motion intensity to measure video energy.
*   **Lighting Analysis**: Analyzes frame brightness and contrast.
*   **Text Overlay Analysis**: Uses OCR to detect and analyze text, captions, and keywords appearing on screen.
*   **Content Classification**: Uses AI (e.g., CLIP) to understand the visual content of the video.
*   **Platform Recommendation**: Suggests the best platform (Instagram, TikTok, YouTube Shorts) based on video characteristics.
*   **Hashtag Generation**: Generates relevant hashtags based on the video content.

## 📂 Project Structure

```
Social Analytics API/
├── api.py                      # FastAPI application entry point
├── social_analytics_pipeline.py # Main analysis orchestrator
├── Config.py                   # Configuration settings (paths, thresholds)
├── requirements.txt            # Python dependencies
├── features/                   # Analysis modules
│   ├── audio_features.py       # Audio analysis
│   ├── clip_analysis.py        # CLIP-based content classification
│   ├── hook_analysis.py        # Introduction engagement analysis
│   ├── pacing_analysis.py      # Cut detection and motion analysis
│   ├── text_overlay_analysis.py # OCR text detection
│   └── ...
├── pipeline/                   # Data extraction modules
│   ├── audio_extractor.py      # Extracts audio track
│   ├── frame_extractor.py      # Extracts frames for analysis
│   ├── scene_detector.py       # Detects scene changes
│   └── video_loader.py         # Loads video files
└── models/                     # Saved ML models (e.g., LightGBM)
```

## 🛠️ Prerequisites

*   **Python 3.8+**

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/social-analytics-api.git
    cd "Social Analytics API"
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  **Start the API server:**
    ```bash
    uvicorn api:app --reload
    ```
    The server will start at `http://127.0.0.1:8000`.

2.  **Access the API Documentation:**
    Open your browser and navigate to `http://127.0.0.1:8000/docs` to see the interactive Swagger UI.

3.  **Analyze a Video:**
    You can use the Swagger UI to upload a file to the `/analyze` endpoint, or use `curl`:

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/analyze' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'video=@/path/to/your/video.mp4'
    ```

## ⚙️ Configuration

You can adjust analysis parameters in `Config.py`:
-   **`SCENE_THRESHOLD`**: Sensitivity for scene detection.
-   **`TARGET_FPS`**: Frame rate for sampling (default: 1 FPS for speed).
-   **`TESSERACT_CMD`**: Path to the Tesseract executable (e.g., `r"C:\Program Files\Tesseract-OCR\tesseract.exe"`).







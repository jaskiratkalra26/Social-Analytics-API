from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import shutil
import os
import uuid
import logging
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from social_analytics_pipeline import analyze_video
from analysis import clip_analysis
import Config

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SocialAnalyticsAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load heavy models
    logger.info("Loading CLIP and ML models into memory...")
    clip_analysis.preload_models()
    logger.info("Models loaded and ready.")
    yield
    # Shutdown: Clean up if needed
    pass

app = FastAPI(title="Social Analytics API", description="API for analyzing social media videos.", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Social Analytics API"}

UPLOAD_DIR = "temp_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/analyze")
async def analyze_social_video(video: UploadFile = File(...)):
    """
    Endpoint to analyze a video file for social analytics.
    """
    # Generate a unique filename to avoid collisions
    file_extension = os.path.splitext(video.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"Video saved to {file_path}. Starting analysis...")

        # Run the analysis pipeline
        try:
            results = analyze_video(file_path)
            
            # Check if the pipeline returned an error
            if "error" in results and isinstance(results["error"], str):
                 logger.error(f"Pipeline error: {results['error']}")
                 # Depending on requirements, we might return 400 or just the error JSON
                 # For now, let's return it as part of the JSON response
                 
            return JSONResponse(content=results)

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    finally:
        # Cleanup: remove the temporary file and generated artifacts
        cleanup_paths = [
            file_path,  # The uploaded video file
            os.path.join(Config.FRAMES_OUTPUT_DIR, os.path.splitext(unique_filename)[0]), # Extracted frames folder
            os.path.join(Config.AUDIO_OUTPUT_DIR, os.path.splitext(unique_filename)[0] + ".wav"), # Extracted audio
            os.path.join(Config.METADATA_OUTPUT_DIR, os.path.splitext(unique_filename)[0] + ".json") # Metadata
        ]

        for path in cleanup_paths:
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        logger.info(f"Cleaned up directory: {path}")
                    else:
                        os.remove(path)
                        logger.info(f"Cleaned up file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {path}: {e}")

@app.get("/")
def read_root():
    return {"message": "Social Analytics API is running. Use /analyze to analyze videos."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

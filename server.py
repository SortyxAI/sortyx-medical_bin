"""
Backend Classification Server
Handles YOLO and Gemini AI classification requests
Runs on port 8001 (locally) or PORT env variable (cloud)
Optimized for Render cloud deployment
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import uvicorn
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import logging
from datetime import datetime
import sys

# Configure logging for cloud deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensures logs appear in Render dashboard
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.80'))  # Allow config via env
MODEL_PATH = os.getenv('MODEL_PATH', 'models/ClassifyBest.pt')
PORT = int(os.getenv('PORT', '8001'))  # Render provides PORT env variable

# Startup logging
logger.info("="*60)
logger.info("🚀 Initializing Medical Waste Classification Server")
logger.info(f"📍 Environment: {'PRODUCTION (Render)' if os.getenv('RENDER') else 'DEVELOPMENT'}")
logger.info(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
logger.info(f"📂 Model Path: {MODEL_PATH}")
logger.info("="*60)

app = FastAPI(
    title="Medical Waste Classification Server",
    description="AI-powered medical waste classification using YOLO and Gemini",
    version="1.0.0"
)

# Request counter for monitoring
request_counter = {"total": 0, "classify": 0, "success": 0, "errors": 0}

# Initialize Gemini API
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✓ Gemini model initialized successfully")
    else:
        gemini_model = None
        logger.warning("⚠ Warning: No valid Gemini API key provided. Gemini fallback disabled.")
except Exception as e:
    logger.error(f"⚠ Warning: Could not initialize Gemini model: {e}")
    gemini_model = None

# Initialize YOLO model
try:
    logger.info(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = list(model.names.values())
    logger.info(f"✓ YOLO model loaded successfully. Classes: {CLASS_NAMES}")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {e}")
    raise

# Waste counts tracking
waste_counts = {name: 0 for name in CLASS_NAMES}


class ClassificationRequest(BaseModel):
    image: str  # Base64 encoded image
    

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    model_used: str  # "YOLO", "GEMINI", "NONE"
    explanation: str = ""  # Gemini explanation
    class_names: list
    counts: dict


def classify_with_gemini(frame):
    """
    Classify waste using Gemini AI
    Returns: (label, confidence, explanation)
    """
    logger.info("🤖 Attempting Gemini AI classification...")
    
    if gemini_model is None:
        logger.warning("Gemini API not available")
        return "GEMINI_FEED_NOT_PRESENT", 0.0, "Gemini API not available"
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        prompt = f"""
        Analyze this image and classify the waste item into one of these categories: {', '.join(CLASS_NAMES)}.
        
        Look for:
        - General waste: regular trash, food waste, paper, plastic containers
        - Sharp: needles, broken glass, scalpels, metal fragments  
        - Chemical waste: containers with chemical labels, hazardous symbols
        - Biohazard: medical waste, blood products, contaminated materials
        
        Respond with:
        1. First line: Only the category name from the list: {CLASS_NAMES}
        2. Second line onwards: Brief explanation of why you classified it this way
        
        If you cannot clearly identify waste in the image, respond with "UNCLEAR" and explain why.
        """
        
        response = gemini_model.generate_content([prompt, pil_image])
        
        if response and response.text:
            response_lines = response.text.strip().split('\n', 1)
            predicted_class = response_lines[0].strip().upper()
            explanation = response_lines[1].strip() if len(response_lines) > 1 else "Classification completed"
            
            # Match against known class names
            for class_name in CLASS_NAMES:
                if class_name.upper() in predicted_class:
                    logger.info(f"✓ Gemini classified as: {class_name}")
                    return class_name, 0.75, explanation
            
            logger.warning(f"Gemini classification unclear: {predicted_class}")
            return "GEMINI_UNCLEAR", 0.0, explanation
        else:
            logger.error("No response from Gemini")
            return "GEMINI_NO_RESPONSE", 0.0, "No response from Gemini"
            
    except Exception as e:
        logger.error(f"Gemini classification error: {e}")
        return "GEMINI_ERROR", 0.0, f"Error: {str(e)}"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests"""
    request_counter["total"] += 1
    start_time = datetime.now()
    
    logger.info(f"📥 Request [{request_counter['total']}]: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"📤 Response [{request_counter['total']}]: Status {response.status_code} | Time: {process_time:.3f}s")
    
    return response


@app.on_event("startup")
async def startup_event():
    """Log when the server starts"""
    logger.info("="*60)
    logger.info("✅ Server startup complete!")
    logger.info(f"📊 Available classes: {CLASS_NAMES}")
    logger.info(f"🔗 Access API docs at: /docs")
    logger.info("="*60)


@app.get("/")
async def root():
    """Root endpoint with server information"""
    logger.info("🏠 Root endpoint accessed")
    return {
        "service": "Medical Waste Classification Server",
        "status": "running",
        "version": "1.0.0",
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "gemini_available": gemini_model is not None,
        "total_requests": request_counter["total"],
        "classify_requests": request_counter["classify"],
        "environment": "production" if os.getenv('RENDER') else "development"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    logger.info("💚 Health check endpoint accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "gemini_available": gemini_model is not None,
        "uptime_requests": request_counter["total"]
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(request: ClassificationRequest):
    """
    Classify a medical waste item from base64 encoded image
    Returns classification label, confidence, and model used
    """
    request_counter["classify"] += 1
    request_id = request_counter["classify"]
    
    logger.info(f"🔍 [Request {request_id}] Classification request received")
    
    try:
        # Decode base64 image
        logger.debug(f"[Request {request_id}] Decoding base64 image...")
        if "," in request.image:
            img_data = request.image.split(",")[1]
        else:
            img_data = request.image
            
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error(f"[Request {request_id}] Invalid image data received")
            request_counter["errors"] += 1
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        logger.info(f"[Request {request_id}] Image decoded successfully: {frame.shape}")
        
        # Resize for YOLO processing
        processed_frame = cv2.resize(frame, (640, 640))
        
        # Run YOLO classification
        logger.info(f"[Request {request_id}] Running YOLO classification...")
        results = model(processed_frame)
        probs = results[0].probs
        
        if probs is None:
            logger.warning(f"[Request {request_id}] No probabilities detected from YOLO")
            return ClassificationResponse(
                label="NO_PROBS_DETECTED",
                confidence=0.0,
                model_used="NONE",
                explanation="",
                class_names=CLASS_NAMES,
                counts=waste_counts
            )
        
        cls_id = int(probs.top1)
        confidence = float(probs.top1conf.item())
        
        logger.info(f"[Request {request_id}] YOLO result: class_id={cls_id}, confidence={confidence:.2%}, threshold={CONFIDENCE_THRESHOLD:.2%}")
        
        # Initialize response variables
        label = None
        model_used = "NONE"
        explanation = ""
        
        # Check YOLO confidence levels
        if confidence >= CONFIDENCE_THRESHOLD and 0 <= cls_id < len(CLASS_NAMES):
            # High confidence classification (≥80%) - use YOLO
            label = CLASS_NAMES[cls_id]
            model_used = "YOLO"
            explanation = f"High confidence YOLO classification"
            logger.info(f"[Request {request_id}] ✓ High confidence YOLO: {label} ({confidence:.2%})")
            
        elif 0 <= cls_id < len(CLASS_NAMES):
            # YOLO confidence below 80% - use Gemini API fallback
            logger.info(f"[Request {request_id}] YOLO confidence {confidence:.2%} below threshold, using Gemini...")
            gemini_label, gemini_confidence, gemini_explanation = classify_with_gemini(frame)
            
            if gemini_label in CLASS_NAMES:
                # Gemini successfully classified
                label = gemini_label
                confidence = gemini_confidence
                model_used = "GEMINI"
                explanation = gemini_explanation
                logger.info(f"[Request {request_id}] ✓ Gemini classification: {label} ({confidence:.2%})")
                
            elif gemini_label == "GEMINI_FEED_NOT_PRESENT":
                label = "GEMINI_FEED_NOT_PRESENT"
                model_used = "GEMINI_NO_KEY"
                explanation = gemini_explanation
                logger.warning(f"[Request {request_id}] Gemini feed not present - no valid API key")
                
            else:
                # Gemini failed to classify
                label = "UNCLEAR"
                confidence = 0.0
                model_used = "NONE"
                explanation = gemini_explanation
                logger.warning(f"[Request {request_id}] Gemini failed to classify")
        else:
            # Invalid class ID from YOLO
            label = "INVALID_CLASS"
            confidence = 0.0
            model_used = "NONE"
            explanation = "Invalid classification result"
            logger.error(f"[Request {request_id}] Invalid class ID from YOLO: {cls_id}")
        
        request_counter["success"] += 1
        logger.info(f"[Request {request_id}] ✅ Classification complete: {label} using {model_used}")
        
        return ClassificationResponse(
            label=label,
            confidence=confidence,
            model_used=model_used,
            explanation=explanation,
            class_names=CLASS_NAMES,
            counts=waste_counts
        )
        
    except HTTPException:
        raise
    except Exception as e:
        request_counter["errors"] += 1
        logger.error(f"[Request {request_id}] ❌ Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/update_count")
async def update_waste_count(label: str):
    """
    Update the waste count for a given label
    """
    logger.info(f"📊 Updating count for label: {label}")
    if label in waste_counts:
        waste_counts[label] += 1
        logger.info(f"✓ Updated waste counts: {waste_counts}")
        return {"success": True, "counts": waste_counts}
    else:
        logger.error(f"❌ Invalid label attempted: {label}")
        raise HTTPException(status_code=400, detail=f"Invalid label: {label}")


@app.get("/counts")
async def get_counts():
    """
    Get current waste counts
    """
    logger.info(f"📈 Waste counts requested: {waste_counts}")
    return {"counts": waste_counts}


@app.post("/reset_counts")
async def reset_counts():
    """
    Reset all waste counts to zero
    """
    global waste_counts
    waste_counts = {name: 0 for name in CLASS_NAMES}
    logger.info("🔄 Waste counts reset to zero")
    return {"success": True, "counts": waste_counts}


@app.get("/stats")
async def get_statistics():
    """
    Get server statistics - useful for monitoring in Render
    """
    logger.info("📊 Statistics requested")
    return {
        "total_requests": request_counter["total"],
        "classify_requests": request_counter["classify"],
        "successful_classifications": request_counter["success"],
        "errors": request_counter["errors"],
        "success_rate": f"{(request_counter['success'] / max(request_counter['classify'], 1)) * 100:.2f}%",
        "waste_counts": waste_counts,
        "model_info": {
            "yolo_loaded": model is not None,
            "gemini_available": gemini_model is not None,
            "classes": CLASS_NAMES
        }
    }


# For local development only - Render won't use this
if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting Medical Waste Classification Server (LOCAL MODE)")
    logger.info("="*60)
    logger.info(f"📍 Server URL: http://0.0.0.0:{PORT}")
    logger.info(f"📊 API Docs: http://0.0.0.0:{PORT}/docs")
    logger.info(f"🎯 Classes: {CLASS_NAMES}")
    logger.info(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"🤖 Gemini Available: {gemini_model is not None}")
    logger.info("="*60 + "\n")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False  # Set to False for production
    )

"""
Backend Classification Server
Handles YOLO and Gemini AI classification requests
Runs on port 8001
"""

from fastapi import FastAPI, HTTPException
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

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CONFIDENCE_THRESHOLD = 0.80  # 80% threshold for YOLO, below this uses Gemini
MODEL_PATH = "models/ClassifyBest.pt"

app = FastAPI(title="Medical Waste Classification Server")

# Initialize Gemini API
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("✓ Gemini model initialized successfully")
    else:
        gemini_model = None
        print("⚠ Warning: No valid Gemini API key provided. Gemini fallback disabled.")
except Exception as e:
    print(f"⚠ Warning: Could not initialize Gemini model: {e}")
    gemini_model = None

# Initialize YOLO model
print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = list(model.names.values())
print(f"✓ YOLO model loaded successfully. Classes: {CLASS_NAMES}")

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
    if gemini_model is None:
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
                    return class_name, 0.75, explanation
            
            return "GEMINI_UNCLEAR", 0.0, explanation
        else:
            return "GEMINI_NO_RESPONSE", 0.0, "No response from Gemini"
            
    except Exception as e:
        print(f"Gemini classification error: {e}")
        return "GEMINI_ERROR", 0.0, f"Error: {str(e)}"


@app.get("/")
async def root():
    return {
        "service": "Medical Waste Classification Server",
        "status": "running",
        "classes": CLASS_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "gemini_available": gemini_model is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gemini_available": gemini_model is not None
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(request: ClassificationRequest):
    """
    Classify a medical waste item from base64 encoded image
    Returns classification label, confidence, and model used
    """
    try:
        # Decode base64 image
        if "," in request.image:
            img_data = request.image.split(",")[1]
        else:
            img_data = request.image
            
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Resize for YOLO processing
        processed_frame = cv2.resize(frame, (640, 640))
        
        # Run YOLO classification
        results = model(processed_frame)
        probs = results[0].probs
        
        if probs is None:
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
        
        print(f"YOLO result: class_id={cls_id}, confidence={confidence:.2f}, threshold={CONFIDENCE_THRESHOLD}")
        
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
            print(f"High confidence YOLO classification: {label} with confidence {confidence:.2f}")
            
        elif 0 <= cls_id < len(CLASS_NAMES):
            # YOLO confidence below 80% - use Gemini API fallback
            print(f"YOLO confidence {confidence:.2f} below threshold, using Gemini...")
            gemini_label, gemini_confidence, gemini_explanation = classify_with_gemini(frame)
            
            if gemini_label in CLASS_NAMES:
                # Gemini successfully classified
                label = gemini_label
                confidence = gemini_confidence
                model_used = "GEMINI"
                explanation = gemini_explanation
                print(f"Gemini classification: {label} with confidence {confidence:.2f}")
                
            elif gemini_label == "GEMINI_FEED_NOT_PRESENT":
                label = "GEMINI_FEED_NOT_PRESENT"
                model_used = "GEMINI_NO_KEY"
                explanation = gemini_explanation
                print("Gemini feed not present - no valid API key")
                
            else:
                # Gemini failed to classify
                label = "UNCLEAR"
                confidence = 0.0
                model_used = "NONE"
                explanation = gemini_explanation
                print(f"Gemini failed to classify")
        else:
            # Invalid class ID from YOLO
            label = "INVALID_CLASS"
            confidence = 0.0
            model_used = "NONE"
            explanation = "Invalid classification result"
            print(f"Invalid class ID from YOLO")
        
        return ClassificationResponse(
            label=label,
            confidence=confidence,
            model_used=model_used,
            explanation=explanation,
            class_names=CLASS_NAMES,
            counts=waste_counts
        )
        
    except Exception as e:
        print(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/update_count")
async def update_waste_count(label: str):
    """
    Update the waste count for a given label
    """
    if label in waste_counts:
        waste_counts[label] += 1
        print(f"Updated waste counts: {waste_counts}")
        return {"success": True, "counts": waste_counts}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid label: {label}")


@app.get("/counts")
async def get_counts():
    """
    Get current waste counts
    """
    return {"counts": waste_counts}


@app.post("/reset_counts")
async def reset_counts():
    """
    Reset all waste counts to zero
    """
    global waste_counts
    waste_counts = {name: 0 for name in CLASS_NAMES}
    print("Waste counts reset")
    return {"success": True, "counts": waste_counts}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Medical Waste Classification Server")
    print("="*60)
    print(f"📍 Server URL: http://0.0.0.0:8001")
    print(f"📊 API Docs: http://0.0.0.0:8001/docs")
    print(f"🎯 Classes: {CLASS_NAMES}")
    print(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"🤖 Gemini Available: {gemini_model is not None}")
    print("="*60 + "\n")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

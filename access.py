from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
import uvicorn
from datetime import datetime
import os
import csv
import asyncio
import google.generativeai as genai
from PIL import Image
import io


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'Enter api key here')
CONFIDENCE_THRESHOLD = 0.80  # 80% threshold for YOLO, below this uses Gemini
MOTION_THRESHOLD = 2000  # Motion threshold to avoid false detections from camera noise
MODEL_PATH = "models/ClassifyBest.pt"

app = FastAPI()

# Gemini API
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != 'Enter api key here':
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model initialized successfully")
    else:
        gemini_model = None
        print("Warning: No valid Gemini API key provided. Gemini fallback disabled.")
except Exception as e:
    print(f"Warning: Could not initialize Gemini model: {e}")
    gemini_model = None

model = YOLO("models/ClassifyBest.pt")

CLASS_NAMES = list(model.names.values()) 
waste_counts = {name: 0 for name in CLASS_NAMES}

def classify_with_gemini(frame):
  
    if gemini_model is None:
        return "GEMINI_FEED_NOT_PRESENT", 0.0
    
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
        
        Respond with only the category name from the list: {CLASS_NAMES}
        If you cannot clearly identify waste in the image, respond with "UNCLEAR".
        """
        
    
        response = gemini_model.generate_content([prompt, pil_image])
        
        
        if response and response.text:
            predicted_class = response.text.strip().upper()
            
            
            for class_name in CLASS_NAMES:
                if class_name.upper() in predicted_class:
                    return class_name, 0.75  # Assign 75% confidence for Gemini predictions
            
           
            return "GEMINI_UNCLEAR", 0.0
        else:
            return "GEMINI_NO_RESPONSE", 0.0
            
    except Exception as e:
        print(f"Gemini classification error: {e}")
        return "GEMINI_ERROR", 0.0

app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/static", StaticFiles(directory="static"), name="static") 

@app.get("/")
async def index():
    return FileResponse("templates/index.html")

@app.websocket("/ws")
async def classify_websocket(websocket: WebSocket):
    global waste_counts
    await websocket.accept()
    
    prev_gray = None
    
   
    timing_state = None  # None, "analyzing", "showing_result", "waiting"
    timing_start = None
    last_classification = None
    last_confidence = 0.0
    last_model_used = "NONE" 
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            image_data = data.get("frame")

            if not image_data:
                continue
            try:
                img_bytes = base64.b64decode(image_data.split(",")[1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None: continue
            except Exception as e:
                print("Decode error:", e)
                continue

            processed_frame = cv2.resize(frame, (640, 640))   # *NEW*    
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            
            current_time = datetime.now()
            if timing_state is not None and timing_start is not None:
                elapsed = (current_time - timing_start).total_seconds()
                
                if timing_state == "analyzing" and elapsed < 3.0:
                    
                    countdown = int(3 - elapsed)
                    await websocket.send_text(json.dumps({
                        "label": "ANALYZING_COUNTDOWN",
                        "confidence": 0.0,
                        "counts": waste_counts,
                        "model_used": "NONE",
                        "countdown": countdown + 1
                    }))
                    continue  # Skip motion detection during analysis
                    
                elif timing_state == "analyzing" and elapsed >= 3.0:
                    
                    waste_counts[last_classification] += 1
                    print(f"Updated waste counts: {waste_counts}")
                    await websocket.send_text(json.dumps({
                        "label": last_classification,
                        "confidence": last_confidence,
                        "counts": waste_counts,
                        "model_used": last_model_used
                    }))
                    timing_state = "waiting"
                    timing_start = current_time
                    continue
                    
                elif timing_state == "waiting" and elapsed < 3.0:
                    
                    countdown = int(3 - elapsed)
                    await websocket.send_text(json.dumps({
                        "label": "WAIT_COUNTDOWN",
                        "confidence": last_confidence,
                        "counts": waste_counts,
                        "model_used": last_model_used,
                        "countdown": countdown + 1,
                        "last_classification": last_classification
                    }))
                    continue 
                    
                elif timing_state == "waiting" and elapsed >= 3.0:
                   
                    timing_state = None
                    timing_start = None
                    last_classification = None

            # Motion Detection Logic 
            if prev_gray is None:
                prev_gray = gray.copy()
                label = "INIT"
                confidence = 0.0
            else:
        
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_level = np.sum(thresh) / 255
                
                label = "NO_MOTION" 
                confidence = 0.0
                
                
                if motion_level > MOTION_THRESHOLD:
                    print(f"MOTION DETECTED: {motion_level:.0f} (threshold: {MOTION_THRESHOLD}) - Running YOLO")
                else:
                    print(f"NO MOTION: {motion_level:.0f} (threshold: {MOTION_THRESHOLD}) - Should show intro image")

        
                if motion_level > MOTION_THRESHOLD:
                    try:
                        #results = model(frame)
                        results = model(processed_frame)
                        probs = results[0].probs
                        
                        if probs is not None:
                            cls_id = int(probs.top1)
                            confidence = float(probs.top1conf.item())
                            
                            print(f"YOLO result: class_id={cls_id}, confidence={confidence:.2f}, threshold={CONFIDENCE_THRESHOLD}")
                            
                            # Initialize variables
                            label = None
                            model_used = "NONE"
                            valid_classification = False
                            
                            # Check YOLO confidence levels
                            if confidence >= CONFIDENCE_THRESHOLD and 0 <= cls_id < len(CLASS_NAMES):
                                # High confidence classification (≥80%) - use YOLO with timing sequence
                                label = CLASS_NAMES[cls_id]
                                model_used = "YOLO"
                                valid_classification = True
                                print(f"High confidence YOLO classification: {label} with confidence {confidence:.2f}")
                                
                            elif 0 <= cls_id < len(CLASS_NAMES):
                                # YOLO confidence below 80% - use Gemini API fallback
                                print(f"YOLO confidence {confidence:.2f} below 80% threshold, using Gemini...")
                                gemini_label, gemini_confidence = classify_with_gemini(frame)
                                
                                if gemini_label in CLASS_NAMES:
                                    # Gemini successfully classified
                                    label = gemini_label
                                    confidence = gemini_confidence
                                    model_used = "GEMINI"
                                    valid_classification = True
                                    print(f"Gemini classification: {label} with confidence {confidence:.2f}")
                                    
                                elif gemini_label == "GEMINI_FEED_NOT_PRESENT":
                                    
                                    label = "GEMINI_FEED_NOT_PRESENT"
                                    model_used = "GEMINI_NO_KEY"
                                    valid_classification = False  # Don't use timing sequence
                                    print("Gemini feed not present - no valid API key")
                                    
                                else:
                                    
                                    label = "NO_MOTION"
                                    confidence = 0.0
                                    valid_classification = False
                                    print(f"Gemini failed to classify, showing NO_MOTION")
                                
                            else:
                               
                                label = "NO_MOTION"
                                confidence = 0.0
                                valid_classification = False
                                print(f"Invalid class ID from YOLO. No classification shown.")
                            
                            
                            if valid_classification and label in CLASS_NAMES and timing_state is None:
                                print(f"Valid object identified: {label} with confidence {confidence:.2f}. Starting timing sequence...")
                                
                                # Start non-blocking timing sequence
                                timing_state = "analyzing"
                                timing_start = datetime.now()
                                last_classification = label
                                last_confidence = confidence
                                last_model_used = model_used
                                
                               
                                await websocket.send_text(json.dumps({
                                    "label": "ANALYZING",
                                    "confidence": 0.0,
                                    "counts": waste_counts,
                                    "model_used": "NONE"
                                }))
                                
                               
                                continue
                            else:
                                
                                if timing_state is None:
                                    print(f"Motion detected but no timing sequence. Label: {label}, Valid: {valid_classification}, Model: {model_used}")
                                else:
                                    print(f"Timing sequence in progress, ignoring new motion detection")
                            
                        else:
                            label = "NO_PROBS_DETECTED"
                            confidence = 0.0

                    except Exception as e:
                        print("Model error:", e)
                        label = "ERROR"
                        confidence = 0.0
                
        
                prev_gray = gray.copy()
                
                
                print(f"Final label after motion processing: '{label}', confidence: {confidence:.2f}")

    
            if timing_state is None:
                if label in ["NO_MOTION", "INIT", "ERROR", "NO_PROBS_DETECTED", "GEMINI_FEED_NOT_PRESENT"] or label not in CLASS_NAMES:
                    # Handle cases that don't need timing sequence
                    if label == "GEMINI_FEED_NOT_PRESENT":
                        model_used = "GEMINI_NO_KEY"
                    else:
                        model_used = "NONE"
                    
                    print(f"Sending message: label='{label}', confidence={confidence:.2f}, model_used='{model_used}'")
                    
                    await websocket.send_text(json.dumps({
                        "label": label,
                        "confidence": confidence,
                        "counts": waste_counts,
                        "model_used": model_used
                    }))

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print("Unexpected error:", e)
    finally:
    
        try:
            if not websocket.client_state.name == "DISCONNECTED":
                await websocket.close()
        except Exception as e:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "access:app",         
        host="0.0.0.0",
        port=8000,
        reload=True             
    )

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


CONFIDENCE_THRESHOLD = 0.70  
LOW_CONFIDENCE_THRESHOLD = 0.30
MOTION_THRESHOLD = 2000  
MODEL_PATH = "models/ClassifyBest.pt"

app = FastAPI()

model = YOLO("models/ClassifyBest.pt")

CLASS_NAMES = list(model.names.values()) 
waste_counts = {name: 0 for name in CLASS_NAMES}


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
    
   
    timing_state = None  # "analyzing", "show_result", "waiting"
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

            processed_frame = cv2.resize(frame, (640, 640))     
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Handle time
            current_time = datetime.now()
            if timing_state is not None and timing_start is not None:
                elapsed = (current_time - timing_start).total_seconds()
                
                if timing_state == "analyzing" and elapsed < 3.0:
                    # 3-second analyzing phase
                    countdown = int(3 - elapsed)
                    await websocket.send_text(json.dumps({
                        "label": "ANALYZING_COUNTDOWN",
                        "confidence": 0.0,
                        "counts": waste_counts,
                        "model_used": "NONE",
                        "countdown": countdown + 1
                    }))
                    continue  # Skip motion during analysis
                    
                elif timing_state == "analyzing" and elapsed >= 3.0:
                    # Classification result
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
                    # 3-second wait
                    countdown = int(3 - elapsed)
                    await websocket.send_text(json.dumps({
                        "label": "WAIT_COUNTDOWN",
                        "confidence": last_confidence,
                        "counts": waste_counts,
                        "model_used": last_model_used,
                        "countdown": countdown + 1,
                        "last_classification": last_classification
                    }))
                    continue  # Skip motion during wait
                    
                elif timing_state == "waiting" and elapsed >= 3.0:
                    # normal detection
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
                
                # Debug motion
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
                            
                          
                            label = None
                            model_used = "NONE"
                            valid_classification = False
                            
                            # YOLO confidence level
                            if confidence >= CONFIDENCE_THRESHOLD and 0 <= cls_id < len(CLASS_NAMES):
                                # High confidence classification (gt 70%) - use timing sequence
                                label = CLASS_NAMES[cls_id]
                                model_used = "YOLO"
                                valid_classification = True
                                print(f"High confidence YOLO classification: {label} with confidence {confidence:.2f}")
                                
                            elif confidence >= LOW_CONFIDENCE_THRESHOLD and 0 <= cls_id < len(CLASS_NAMES):
                                # Low confidence classification (30-70%)
                                label = f"{CLASS_NAMES[cls_id]}_LOW_CONF"
                                model_used = "YOLO_LOW"
                                valid_classification = False
                                print(f"Low confidence YOLO classification: {CLASS_NAMES[cls_id]} with confidence {confidence:.2f}")
                                
                            else:
                                # Very low confidence (<30%)
                                # Set to NO_MOTION to show intro image (default behavior)
                                label = "NO_MOTION"
                                confidence = 0.0
                                valid_classification = False
                                print(f"YOLO confidence {confidence:.2f} below 30% threshold. No classification shown.")
                            
                           
                            if valid_classification and label in CLASS_NAMES and timing_state is None:
                                print(f"Valid object identified: {label} with confidence {confidence:.2f}. Starting timing sequence...")
                                

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
                                    print(f"Motion detected but no timing sequence. Label: {label}, Valid: {valid_classification}")
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
                if label in ["NO_MOTION", "INIT", "ERROR", "NO_PROBS_DETECTED"] or label not in CLASS_NAMES:
                    
                    if label.endswith("_LOW_CONF"):
                        model_used = "YOLO_LOW"
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
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

app = FastAPI()


model = YOLO("models/ClassifyBest.pt")

CLASS_NAMES = list(model.names.values()) 
waste_counts = {name: 0 for name in CLASS_NAMES}

MOTION_THRESHOLD = 2000

app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/static", StaticFiles(directory="static"), name="static") 

@app.get("/")
async def index():
    return FileResponse("templates/index.html")

@app.websocket("/ws")
async def classify_websocket(websocket: WebSocket):
    global waste_counts
    await websocket.accept()

    prev_gray = None   # PrevMotion
    
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
                    try:
                        results = model(processed_frame)
                        probs = results[0].probs
                        
                        if probs is not None and probs.top1conf.item() > 0.5:
                            cls_id = int(probs.top1)

                            if 0 <= cls_id < len(CLASS_NAMES):   
                                label = CLASS_NAMES[cls_id]
                                confidence = float(probs.top1conf.item())

                                if label in waste_counts:        
                                    waste_counts[label] += 1
                            
                        else:
                            label = "MOTION_LOW_CONFIDENCE" 

                    except Exception as e:
                        print("Model error:", e)
                        label = "ERROR"
        
                prev_gray = gray.copy()
        
            await websocket.send_text(json.dumps({
                "label": label,
                "confidence": confidence,
                "counts": waste_counts
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
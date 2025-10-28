"""
Frontend Client Application
Handles camera input, motion detection, image processing, and UI communication
Communicates with backend server for classification
Runs on port 8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import json
import uvicorn
from datetime import datetime
import httpx
import asyncio

# Configuration
MOTION_THRESHOLD = 2000  # Motion threshold to avoid false detections from camera noise
BACKEND_URL = "http://localhost:8001"  # Backend server URL

app = FastAPI(title="Medical Waste Classification Frontend")

# Mount static files and templates
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    """Serve the main HTML page"""
    return FileResponse("templates/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health", timeout=5.0)
            backend_status = response.json()
    except Exception as e:
        backend_status = {"error": str(e)}
    
    return {
        "frontend": "healthy",
        "backend": backend_status
    }


async def classify_with_backend(image_base64: str) -> dict:
    """
    Send image to backend server for classification
    Returns classification result
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/classify",
                json={"image": image_base64},
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Backend classification error: {response.status_code}")
                return {
                    "label": "BACKEND_ERROR",
                    "confidence": 0.0,
                    "model_used": "NONE",
                    "explanation": f"Backend error: {response.status_code}",
                    "class_names": [],
                    "counts": {}
                }
    except httpx.TimeoutException:
        print("Backend classification timeout")
        return {
            "label": "BACKEND_TIMEOUT",
            "confidence": 0.0,
            "model_used": "NONE",
            "explanation": "Backend server timeout",
            "class_names": [],
            "counts": {}
        }
    except Exception as e:
        print(f"Backend communication error: {e}")
        return {
            "label": "BACKEND_UNAVAILABLE",
            "confidence": 0.0,
            "model_used": "NONE",
            "explanation": f"Backend unavailable: {str(e)}",
            "class_names": [],
            "counts": {}
        }


async def update_backend_count(label: str) -> dict:
    """
    Update waste count on backend server
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/update_count",
                params={"label": label},
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        print(f"Error updating backend count: {e}")
        return {"success": False, "error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera feed and classification
    Handles motion detection and timing sequences
    """
    await websocket.accept()
    print("✓ Client connected to frontend")
    
    # Motion detection state
    prev_gray = None
    
    # Timing sequence state
    timing_state = None  # None, "analyzing", "waiting"
    timing_start = None
    last_classification = None
    last_confidence = 0.0
    last_model_used = "NONE"
    last_explanation = ""
    current_counts = {}
    
    try:
        while True:
            # Receive frame from client
            message = await websocket.receive_text()
            data = json.loads(message)
            image_data = data.get("frame")

            if not image_data:
                continue

            # Decode image
            try:
                img_bytes = base64.b64decode(image_data.split(",")[1])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                print(f"Image decode error: {e}")
                continue

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Handle timing sequences
            current_time = datetime.now()
            if timing_state is not None and timing_start is not None:
                elapsed = (current_time - timing_start).total_seconds()
                
                # Analyzing state - 3 second countdown
                if timing_state == "analyzing" and elapsed < 3.0:
                    countdown = int(3 - elapsed) + 1
                    await websocket.send_text(json.dumps({
                        "label": "ANALYZING_COUNTDOWN",
                        "confidence": 0.0,
                        "counts": current_counts,
                        "model_used": "NONE",
                        "countdown": countdown,
                        "explanation": ""
                    }))
                    continue
                
                # Analyzing complete - show result and update count
                elif timing_state == "analyzing" and elapsed >= 3.0:
                    # Update count on backend
                    result = await update_backend_count(last_classification)
                    if result.get("success"):
                        current_counts = result.get("counts", current_counts)
                    
                    print(f"Updated waste counts: {current_counts}")
                    await websocket.send_text(json.dumps({
                        "label": last_classification,
                        "confidence": last_confidence,
                        "counts": current_counts,
                        "model_used": last_model_used,
                        "explanation": last_explanation
                    }))
                    timing_state = "waiting"
                    timing_start = current_time
                    continue
                
                # Waiting state - 3 second countdown before accepting new input
                elif timing_state == "waiting" and elapsed < 3.0:
                    countdown = int(3 - elapsed) + 1
                    await websocket.send_text(json.dumps({
                        "label": "WAIT_COUNTDOWN",
                        "confidence": last_confidence,
                        "counts": current_counts,
                        "model_used": last_model_used,
                        "countdown": countdown,
                        "last_classification": last_classification,
                        "explanation": last_explanation
                    }))
                    continue
                
                # Waiting complete - reset state
                elif timing_state == "waiting" and elapsed >= 3.0:
                    timing_state = None
                    timing_start = None
                    last_classification = None

            # Motion Detection Logic
            if prev_gray is None:
                prev_gray = gray.copy()
                await websocket.send_text(json.dumps({
                    "label": "INIT",
                    "confidence": 0.0,
                    "counts": current_counts,
                    "model_used": "NONE",
                    "explanation": ""
                }))
                continue

            # Calculate motion level
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_level = np.sum(thresh) / 255
            
            if motion_level > MOTION_THRESHOLD:
                print(f"🔍 MOTION DETECTED: {motion_level:.0f} (threshold: {MOTION_THRESHOLD})")
            else:
                print(f"💤 NO MOTION: {motion_level:.0f} (threshold: {MOTION_THRESHOLD})")

            # Process motion detection
            if motion_level > MOTION_THRESHOLD and timing_state is None:
                # Send image to backend for classification
                print("📤 Sending image to backend for classification...")
                classification_result = await classify_with_backend(image_data)
                
                label = classification_result.get("label", "ERROR")
                confidence = classification_result.get("confidence", 0.0)
                model_used = classification_result.get("model_used", "NONE")
                explanation = classification_result.get("explanation", "")
                current_counts = classification_result.get("counts", current_counts)
                class_names = classification_result.get("class_names", [])
                
                print(f"📥 Backend response: {label} ({confidence:.2f}) using {model_used}")
                
                # Check if it's a valid classification that needs timing sequence
                valid_classification = (
                    label in class_names and 
                    label not in ["UNCLEAR", "INVALID_CLASS", "BACKEND_ERROR", "BACKEND_TIMEOUT", "BACKEND_UNAVAILABLE"]
                )
                
                if valid_classification:
                    # Start timing sequence
                    print(f"✓ Valid classification: {label}. Starting timing sequence...")
                    timing_state = "analyzing"
                    timing_start = datetime.now()
                    last_classification = label
                    last_confidence = confidence
                    last_model_used = model_used
                    last_explanation = explanation
                    
                    await websocket.send_text(json.dumps({
                        "label": "ANALYZING",
                        "confidence": 0.0,
                        "counts": current_counts,
                        "model_used": "NONE",
                        "explanation": ""
                    }))
                else:
                    # Send result immediately (no timing sequence for errors/unclear)
                    if label == "GEMINI_FEED_NOT_PRESENT":
                        model_used = "GEMINI_NO_KEY"
                    
                    await websocket.send_text(json.dumps({
                        "label": label,
                        "confidence": confidence,
                        "counts": current_counts,
                        "model_used": model_used,
                        "explanation": explanation
                    }))
            
            elif timing_state is None:
                # No motion detected - send NO_MOTION status
                await websocket.send_text(json.dumps({
                    "label": "NO_MOTION",
                    "confidence": 0.0,
                    "counts": current_counts,
                    "model_used": "NONE",
                    "explanation": ""
                }))
            
            # Update previous frame
            prev_gray = gray.copy()

    except WebSocketDisconnect:
        print("✗ Client disconnected from frontend")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🖥️  Starting Medical Waste Classification Frontend")
    print("="*60)
    print(f"📍 Frontend URL: http://0.0.0.0:8000")
    print(f"🔗 Backend URL: {BACKEND_URL}")
    print(f"🎥 Motion Threshold: {MOTION_THRESHOLD}")
    print(f"💡 Open http://localhost:8000 in your browser")
    print("="*60 + "\n")
    
    uvicorn.run(
        "frontend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

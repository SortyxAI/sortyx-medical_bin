# Sortyx Medical Waste Classifier
Real-time waste classification system using YOLO AI model.


## System Architecture

### Backend (`access.py`)
- **FastAPI WebSocket Server**: Real-time communication with web interface
- **YOLO Classification**: Custom trained waste classification model
- **Motion Detection**: Frame differencing to trigger classification only when needed
- **Timing System**: Non-blocking 3s+3s display sequence for identified objects
- **Confidence Handling**: Different behaviors for high/low confidence predictions

### Frontend (`templates/index.html`)
- **Live Camera Feed**: Browser-based webcam access
- **Real-time Display**: WebSocket-powered live updates
- **Waste Counting**: Dynamic category counter with live updates
- **Status Indicators**: Visual feedback for system state and confidence levels
- **Responsive Design**: Clean, modern interface with Montserrat fonts


## Waste Categories
The system classifies waste into these categories:
- **GENERAL**: Regular trash, food waste, paper, plastic containers
- **SHARP**: Needles, broken glass, scalpels, metal fragments
- **CHEMICAL**: Containers with chemical labels, hazardous symbols
- **BIOHAZARD**: Medical waste, blood products, contaminated materials


## Setup Instructions
### Python version 3.12.4
### Activate Virtual Environment
```bash
# Windows PowerShell
& D:/TechCodeBin/venv/Scripts/Activate.ps1
# Or using pip venv
python -m venv venv
venv\Scripts\activate
```
### Run the Application
```bash
python access.py
```
### Access Web Interface
Open your browser and navigate to: `http://localhost:8000`


## Configuration

### Motion Detection Settings
```python
CONFIDENCE_THRESHOLD = 0.70    # 70% for high confidence (green display)
LOW_CONFIDENCE_THRESHOLD = 0.30 # 30% minimum for low confidence (yellow)
MOTION_THRESHOLD = 2000        # Motion sensitivity (higher = less sensitive)
```
### Classification Behavior
- **≥70% confidence**: Green display → 3s+3s timing sequence → Count increment
- **30-70% confidence**: Yellow Low Conf. display → No timing → No count  
- **<30% confidence**: No display → No count
- **No motion**: Intro image


---

## 🚀 Client-Server Architecture (New)

The system has been refactored into a proper client-server architecture for better scalability, maintainability, and separation of concerns.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (index.html)                     │
│                  User Interface & Camera Feed                │
└──────────────────────────┬──────────────────────────────────┘
                           │ WebSocket
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Frontend Server (frontend.py)                   │
│                       Port 8000                              │
│  • Camera Input Capture                                      │
│  • Motion Detection (threshold: 2000)                        │
│  • Image Preprocessing                                       │
│  • WebSocket Communication with UI                           │
│  • Timing Sequences (3s analyzing + 3s wait)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Backend Server (server.py)                      │
│                       Port 8001                              │
│  • YOLO Model Loading & Inference                            │
│  • Gemini AI Fallback (confidence < 80%)                     │
│  • Classification REST API                                   │
│  • Waste Count Management                                    │
│  • Health Check Endpoints                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
                  ┌────────────────────┐
                  │   YOLO Model       │
                  │   Gemini AI        │
                  └────────────────────┘
```

### File Structure

#### **`server.py`** - Backend Classification Server (Port 8001)
**Responsibilities:**
- 🧠 AI/ML classification engine
- 🎯 YOLO model inference
- 🤖 Gemini AI fallback for low-confidence results
- 📊 Waste count tracking
- 🔌 REST API endpoints

**Key Endpoints:**
- `POST /classify` - Classify waste from base64 image
- `POST /update_count` - Update waste count for a category
- `GET /counts` - Get current waste counts
- `POST /reset_counts` - Reset all counts
- `GET /health` - Health check

**Configuration:**
```python
CONFIDENCE_THRESHOLD = 0.80    # 80% threshold for YOLO
MODEL_PATH = "models/ClassifyBest.pt"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
```

#### **`frontend.py`** - Frontend Client Server (Port 8000)
**Responsibilities:**
- 📹 Camera input management
- 🎥 Motion detection processing
- 🖼️ Image preprocessing
- 💬 WebSocket communication with browser
- ⏱️ Timing sequence management
- 🔗 Backend communication via HTTP

**Configuration:**
```python
MOTION_THRESHOLD = 2000           # Motion sensitivity
BACKEND_URL = "http://localhost:8001"
```

#### **`access.py`** - Original Monolithic Application (Deprecated)
The original all-in-one application. Kept for reference but replaced by the client-server architecture.

### Classification Flow

1. **Browser** captures camera frame and sends to **Frontend** via WebSocket
2. **Frontend** performs motion detection
3. If motion detected → **Frontend** sends image to **Backend** via HTTP POST
4. **Backend** runs YOLO classification:
   - If confidence ≥ 80% → Use YOLO result
   - If confidence < 80% → Fallback to Gemini AI
5. **Backend** returns classification result to **Frontend**
6. **Frontend** manages timing sequence:
   - 3 seconds: "Analyzing..." countdown
   - Update waste count on backend
   - 3 seconds: "Wait" countdown before accepting new input
7. **Frontend** sends UI updates to **Browser** via WebSocket

### Running the System

#### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

#### Start Backend Server (Terminal 1)
```bash
cd /Users/ranjith/development/sortyx-medical_bin
python server.py
```
**Output:**
```
🚀 Starting Medical Waste Classification Server
📍 Server URL: http://0.0.0.0:8001
📊 API Docs: http://0.0.0.0:8001/docs
```

#### Start Frontend Server (Terminal 2)
```bash
cd /Users/ranjith/development/sortyx-medical_bin
python frontend.py
```
**Output:**
```
🖥️  Starting Medical Waste Classification Frontend
📍 Frontend URL: http://0.0.0.0:8000
💡 Open http://localhost:8000 in your browser
```

#### Access Application
Open your browser and navigate to: **http://localhost:8000**

### API Documentation

Once the backend is running, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### Key Improvements

✅ **Separation of Concerns**
- Classification logic isolated in backend
- UI and camera logic in frontend
- Clean API boundaries

✅ **Scalability**
- Backend can serve multiple frontends
- Horizontal scaling possible
- Load balancing ready

✅ **Maintainability**
- Easier to debug individual components
- Independent testing
- Clear responsibility boundaries

✅ **Flexibility**
- Backend API can be used by other clients (mobile apps, other services)
- Frontend can be swapped without touching backend
- Easy to add new features

✅ **Performance**
- Async/await throughout
- Non-blocking operations
- Efficient resource usage

✅ **Monitoring**
- Separate health checks for each component
- Better error tracking
- Independent logging

### Configuration Options

#### Backend (`server.py`)
- `CONFIDENCE_THRESHOLD`: Minimum YOLO confidence (default: 0.80)
- `MODEL_PATH`: Path to YOLO model file
- `GEMINI_API_KEY`: Google Gemini API key (from .env)

#### Frontend (`frontend.py`)
- `MOTION_THRESHOLD`: Motion detection sensitivity (default: 2000)
- `BACKEND_URL`: Backend server URL (default: http://localhost:8001)

### Troubleshooting

**Backend not starting?**
- Check if YOLO model exists at `models/ClassifyBest.pt`
- Verify Python dependencies are installed
- Check port 8001 is not in use

**Frontend can't connect to backend?**
- Ensure backend server is running
- Check `BACKEND_URL` in frontend.py
- Verify network connectivity

**Gemini fallback not working?**
- Check `GEMINI_API_KEY` in .env file
- Verify API key is valid
- Check internet connectivity

**Camera not working?**
- Allow camera permissions in browser
- Check camera is not used by another application
- Try a different browser (Chrome recommended)

### Development Tips

**Testing Backend Independently:**
```bash
# Using curl
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_here"}'

# Check health
curl http://localhost:8001/health
```

**Testing Frontend Independently:**
```bash
# Check health (includes backend status)
curl http://localhost:8000/health
```

### Future Enhancements

- [ ] Add authentication/authorization
- [ ] Database integration for persistent counts
- [ ] Multiple camera support
- [ ] Cloud deployment ready
- [ ] Metrics and analytics dashboard
- [ ] Mobile app integration
- [ ] Containerization (Docker)
- [ ] CI/CD pipeline

---

## Original System (access.py)

The original monolithic `access.py` combined all functionality in a single file. While functional, it has been superseded by the client-server architecture for better maintainability and scalability.


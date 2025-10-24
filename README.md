# Sortyx Three Category Bin 2.0
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


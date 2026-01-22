# People Counter

> **Privacy Notice:** This project includes face detection and recognition for visitor tracking. Before deploying, make sure you get consent from people being recorded and follow privacy laws in your area. This is a learning project - use at your own risk.

A real-time people counting system using YOLOX for person detection with tracking and hybrid Re-ID for accurate unique visitor counting.

## Features

- Person detection using YOLOX (ONNX Runtime)
- Real-time tracking with ByteTrack (stable IDs)
- IN/OUT counting with visual line crossing detection
- **Hybrid Re-ID system:**
  - Primary: Face recognition (ArcFace ONNX models) - works across clothing changes
  - Fallback: Body features (color + texture + shape) - when face not visible
- Capacity monitoring with visual alerts
- Modern UI overlay with live statistics

## Use Case

Designed for **multi-day events/expos** where:
- Same visitors return on different days in different clothes
- Accurate unique visitor count is needed
- Face recognition provides cross-day identification

## Known Limitations

- **Face visibility**: Face recognition requires the person to be facing the camera
- **Fallback accuracy**: When face is not visible, body-based Re-ID is less accurate
- **Privacy notice**: Face recognition requires visitor consent notice at entrance

## Project Structure

```
people-counter-tracker/
├── main.py                 # Application entry point
├── config.py               # All configuration settings
├── components/
│   ├── __init__.py
│   ├── detector.py         # YOLOX detector (ONNX Runtime)
│   ├── counter.py          # Main counting logic
│   ├── face_reid.py        # Hybrid Re-ID (face + body fallback)
│   ├── reid.py             # Body-only Re-ID (fallback)
│   └── ui.py               # UI rendering
├── weights/                # Model weights (auto-downloaded)
│   ├── yolox_s.onnx        # Person detection
│   ├── det_500m.onnx       # Face detection (SCRFD)
│   └── w600k_mbf.onnx      # Face recognition (ArcFace)
├── requirements.txt
└── README.md
```

## Dependencies

| Component | License | Purpose |
|-----------|---------|---------|
| YOLOX | Apache 2.0 | Person detection |
| SCRFD | MIT | Face detection |
| ArcFace | MIT | Face recognition |
| Supervision | MIT | Tracking (ByteTrack) |
| OpenCV | Apache 2.0 | Video processing |
| ONNX Runtime | MIT | Model inference |

> Note: Face models (SCRFD + ArcFace) are from the InsightFace project but loaded as standalone ONNX files - no InsightFace package installation required.

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

On first run, models will be automatically downloaded:
- YOLOX person detection model (~35MB)
- Face detection model - SCRFD (~2.5MB)
- Face recognition model - ArcFace (~13MB)

## Keyboard Controls

| Key | Action |
|-----|--------|
| Q | Quit application |
| F | Toggle fullscreen |
| R | Reset all counts |
| S | Swap IN/OUT direction |
| Left/Right Arrow | Move counting line position |
| Up/Down Arrow | Adjust max capacity (+/- 10) |

## Configuration

Edit `config.py` to customize settings:

```python
# Capacity
MAX_CAPACITY = 50

# Detection
MODEL_SIZE = "s"        # Options: nano, tiny, s, m, l, x
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# Re-ID (higher = less restrictive duplicate detection)
SIMILARITY_THRESHOLD = 0.65

# Display
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
```

## How It Works

1. **Detection**: YOLOX detects people in each frame
2. **Tracking**: ByteTrack assigns persistent IDs to each person
3. **Counting**: When a person crosses the counting line, direction is detected (IN or OUT)
4. **Re-ID** (Hybrid):
   - If face is visible → Extract face embedding → Compare with face database
   - If no face → Extract body features → Compare with body database
   - This allows accurate counting even across days when people change clothes

## GPU vs CPU

| Mode | Detection Speed | Face Recognition | Real-time? |
|------|----------------|------------------|------------|
| GPU (CUDA) | ~20-50ms | ~5-10ms | Yes (15-30 FPS) |
| CPU only | ~200-500ms | ~50-100ms | Limited (2-5 FPS) |

For CPU-only servers, consider processing every 3rd frame.

## Potential Improvements

- [ ] Add support for multiple cameras
- [ ] Add data persistence (save counts to database)
- [ ] Add web dashboard for remote monitoring
- [ ] Optimize for edge devices (Jetson, RPi)

## Requirements

- Python 3.8+
- Webcam or camera device
- Windows, Linux, or macOS

## License

MIT License - feel free to use and modify.

## Contributing

This project needs improvement! Feel free to submit issues or pull requests, especially for:
- Better Re-ID accuracy
- Performance optimizations
- New features

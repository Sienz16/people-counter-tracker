# People Counter

A real-time people counting system using YOLOX for person detection with tracking and line-crossing detection.

> **Note**: This is an experimental project and is **not production-ready**. The Re-ID (duplicate detection) feature needs significant improvement - it currently may flag different people as duplicates when they wear similar clothing. Contributions and improvements are welcome!

## Features

- Person detection using YOLOX (ONNX Runtime)
- Real-time tracking with ByteTrack (stable IDs)
- IN/OUT counting with visual line crossing detection
- Re-ID to prevent duplicate counting (color + texture + body shape matching)
- Capacity monitoring with visual alerts
- Modern UI overlay with live statistics

## Known Limitations

- **Re-ID accuracy**: The duplicate detection uses hand-crafted features (color histograms, LBP texture, aspect ratio) which may not reliably distinguish between different people wearing similar clothes
- **Lighting sensitivity**: Color-based matching can be affected by lighting changes
- **For better accuracy**: Consider implementing a deep learning-based Re-ID model (e.g., OSNet)

## Project Structure

```
people-counter-tracker/
├── main.py                 # Application entry point
├── config.py               # All configuration settings
├── components/
│   ├── __init__.py
│   ├── detector.py         # YOLOX detector (ONNX Runtime)
│   ├── counter.py          # Main counting logic
│   ├── reid.py             # Re-identification (duplicate prevention)
│   └── ui.py               # UI rendering
├── weights/                # Model weights
├── requirements.txt
└── README.md
```

## Dependencies

| Component | License |
|-----------|---------|
| YOLOX | Apache 2.0 |
| Supervision | MIT |
| OpenCV | Apache 2.0 |
| ONNX Runtime | MIT |
| NumPy | BSD |

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

The YOLOX model will be automatically downloaded on first run.

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
4. **Re-ID**: Feature matching (color, texture, body shape) attempts to prevent counting the same person twice

## Potential Improvements

- [ ] Implement deep learning Re-ID model (OSNet, FastReID)
- [ ] Add face detection for better person distinction
- [ ] Improve lighting invariance
- [ ] Add support for multiple cameras
- [ ] Add data persistence (save counts to database)

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

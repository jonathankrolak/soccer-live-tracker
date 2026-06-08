# soccer-live-tracker# Soccer Live Tracker

A Python computer vision project that captures a live soccer broadcast from the screen, detects players using YOLO, and displays their positions on a simplified top-down soccer field.

## Current Features

- Live screen capture with `mss`
- Player detection with YOLO
- OpenCV display window
- Simple top-down field visualization
- Smoothed player dot movement
- Cropped capture area for better FPS
- FPS counter

## Tech Stack

- Python
- OpenCV
- MSS
- NumPy
- Ultralytics YOLO

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
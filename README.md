# AI Air Canvas

AI Air Canvas is a Python computer vision project in development. Its goal is to let users draw in the air by tracking hand gestures through a webcam.

> **Development status:** The capabilities and usage below are planned and may not be available in the current project yet.

## Planned capabilities

- Capture a live webcam feed at 1280 × 720 and display it mirrored.
- Track one hand using the MediaPipe Tasks Hand Landmarker.
- Use raised fingers to switch modes:
  - **Selection Mode:** raise the index and middle fingers.
  - **Drawing Mode:** raise only the index finger.
- Provide a toolbar with Blue, Green, Red, Eraser, and Clear All options.
- Draw smoothly on a separate canvas and blend it with the webcam feed.
- Save a drawing as a PNG when `S` is pressed.
- Quit the application when `Q` is pressed.

## Technology

- Python
- OpenCV Contrib (`opencv-contrib-python==5.1.0.95`)
- MediaPipe Tasks (`mediapipe==1.2.3`)
- NumPy (`numpy==2.5.5`)

## Requirements

- Python and pip installed.
- A working webcam is required for the planned live camera experience.
- MediaPipe Tasks Hand Landmarker requires a compatible model asset. Model sourcing and setup instructions have not yet been finalized; this README does not provide a download URL.

## Installation

1. Clone the repository:

   ```powershell
   git clone https://github.com/waleedtechnologiesugandaltd-ship-it/ai-air-canvas.git
   cd ai-air-canvas
   ```

2. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

3. Install the project dependencies:

   You can run the virtual environment's Python executable directly without activating the environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

## Planned usage

Once the application entry point is implemented, the intended command is:

```powershell
python air_canvas.py
```

Planned keyboard controls:

| Key | Action |
| --- | --- |
| `S` | Save the current drawing as a PNG |
| `Q` | Quit the application |

## Development

The application is under active development. Features, model setup, and runtime behavior will be documented as they are implemented and verified.

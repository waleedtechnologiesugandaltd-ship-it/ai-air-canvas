# AI Air Canvas

AI Air Canvas is a Python computer vision project in development. Its goal is to let users draw in the air by tracking hand gestures through a webcam.

> **Development status:** AI Air Canvas is under development; some planned capabilities may not yet be available.

## Capabilities

- Capture a live webcam feed at 1280 × 720 and display it mirrored.
- Track one hand using the MediaPipe Tasks Hand Landmarker.
- Use raised fingers to switch modes:
  - **Selection Mode:** raise the index and middle fingers.
  - **Drawing Mode:** raise only the index finger.
- Provide a toolbar with Blue, Green, Red, Eraser, and Clear All options.
- Draw smoothly on a separate canvas and blend it with the webcam feed.
- Save the drawing to `drawings/` as a transparent PNG when `S` or `s` is pressed.
- Quit the application when `Q` or `q` is pressed.

## Technology

- Python
- OpenCV Contrib (`opencv-contrib-python==5.1.0.95`)
- MediaPipe Tasks (`mediapipe==1.2.3`)
- NumPy (`numpy==2.5.5`)

## Requirements

- Python and pip installed.
- A working webcam is required to use the application.
- Before launching, download the [official Google MediaPipe Hand Landmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and save it as `models/hand_landmarker.task` in the project root. Create the `models` folder if it does not exist. The application does not download the model automatically. The `/models/` folder is excluded by `.gitignore`; do not commit the model binary.

## Installation

1. Clone the repository:

   ```powershell
   git clone https://github.com/waleedtechnologiesugandaltd-ship-it/ai-air-canvas.git
   cd ai-air-canvas
   ```

2. Create a virtual environment. This bootstrap step uses your installed Python because the virtual environment does not exist yet:

   ```powershell
   python -m venv .venv
   ```

3. Install the project dependencies using the virtual environment's Python:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

## Usage

Run the application with the virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe .\air_canvas.py
```

Keyboard controls:

| Key | Action |
| --- | --- |
| `S` or `s` | Save the current drawing as a transparent PNG in `drawings/` |
| `Q` or `q` | Quit the application |

## Development

The application is under active development. Additional features and runtime behavior will be documented as they are implemented and verified.

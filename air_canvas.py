"""Real-time webcam hand landmark preview using MediaPipe Tasks."""

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, hand_landmarker


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "AI Air Canvas"
MODE_IDLE = "Idle"
MODE_SELECTION = "Selection"
MODE_DRAWING = "Drawing"
MODE_NAMES = {
    MODE_IDLE: "Idle",
    MODE_SELECTION: "Selection",
    MODE_DRAWING: "Drawing",
}
TOOL_BLUE = "Blue"
TOOL_GREEN = "Green"
TOOL_RED = "Red"
TOOL_ERASER = "Eraser"
TOOL_CLEAR_ALL = "Clear All"
TOOL_COLORS = {
    TOOL_BLUE: (255, 0, 0),
    TOOL_GREEN: (0, 255, 0),
    TOOL_RED: (0, 0, 255),
}
DRAWING_TOOLS = frozenset((*TOOL_COLORS, TOOL_ERASER))
TOOLBAR_BUTTONS = {
    TOOL_BLUE: (260, 10, 400, 64),
    TOOL_GREEN: (410, 10, 550, 64),
    TOOL_RED: (560, 10, 700, 64),
    TOOL_ERASER: (710, 10, 850, 64),
    TOOL_CLEAR_ALL: (860, 10, 1020, 64),
}
TOOLBAR_HEIGHT = 74
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 36
FINGER_EXTENSION_ANGLE = 150
HOVER_DWELL_SECONDS = 0.5
FEEDBACK_DURATION_SECONDS = 1.5
UI_FONT = cv2.FONT_HERSHEY_DUPLEX
TOOLBAR_FONT_SCALE = 0.65
TOOLBAR_FONT_THICKNESS = 2
STATUS_FONT_SCALE = 0.8
STATUS_FONT_THICKNESS = 2


def create_hand_landmarker(model_path: Path) -> vision.HandLandmarker:
    """Create a VIDEO-mode landmarker configured for a single hand."""
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
    )
    return vision.HandLandmarker.create_from_options(options)


def is_finger_extended(landmarks, mcp_index: int, pip_index: int, tip_index: int) -> bool:
    """Check finger extension from the 3D angle at its PIP joint."""
    mcp = landmarks[mcp_index]
    pip = landmarks[pip_index]
    tip = landmarks[tip_index]
    toward_mcp = (mcp.x - pip.x, mcp.y - pip.y, mcp.z - pip.z)
    toward_tip = (tip.x - pip.x, tip.y - pip.y, tip.z - pip.z)
    if not all(math.isfinite(value) for value in (*toward_mcp, *toward_tip)):
        return False

    dot_product = sum(a * b for a, b in zip(toward_mcp, toward_tip))
    magnitude_mcp = math.sqrt(sum(value * value for value in toward_mcp))
    magnitude_tip = math.sqrt(sum(value * value for value in toward_tip))
    if magnitude_mcp == 0 or magnitude_tip == 0:
        return False

    cosine = dot_product / (magnitude_mcp * magnitude_tip)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return angle >= FINGER_EXTENSION_ANGLE


def classify_gesture(landmarks) -> str:
    """Classify the requested gestures; thumb position is intentionally ignored."""
    if len(landmarks) < 21:
        return MODE_IDLE

    index_extended = is_finger_extended(landmarks, 5, 6, 8)
    middle_extended = is_finger_extended(landmarks, 9, 10, 12)
    ring_extended = is_finger_extended(landmarks, 13, 14, 16)
    pinky_extended = is_finger_extended(landmarks, 17, 18, 20)
    other_fingers_folded = not ring_extended and not pinky_extended

    if index_extended and middle_extended and other_fingers_folded:
        return MODE_SELECTION
    if index_extended and not middle_extended and other_fingers_folded:
        return MODE_DRAWING
    return MODE_IDLE


def normalized_to_pixel(landmark, width: int, height: int):
    """Convert a normalized point to a clamped pixel coordinate."""
    if width <= 0 or height <= 0:
        return None
    if not math.isfinite(landmark.x) or not math.isfinite(landmark.y):
        return None

    x = round(min(1.0, max(0.0, landmark.x)) * (width - 1))
    y = round(min(1.0, max(0.0, landmark.y)) * (height - 1))
    return x, y


def draw_hand(frame, landmarks) -> None:
    """Draw blue landmarks and white connections on a BGR frame."""
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_utils.DrawingSpec(
            color=TOOL_COLORS[TOOL_BLUE], thickness=2, circle_radius=5
        ),
        connection_drawing_spec=drawing_utils.DrawingSpec(
            color=(255, 255, 255), thickness=2
        ),
    )


class DrawingSurface:
    """Keep stroke colors and their alpha mask separate for drawing and erasing."""

    def __init__(self) -> None:
        self.colors = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        self.previous_point = None

    def reset_stroke(self) -> None:
        self.previous_point = None

    def clear(self) -> None:
        self.colors.fill(0)
        self.mask.fill(0)
        self.reset_stroke()

    def update_stroke(self, mode: str, point, tool: str) -> None:
        if mode != MODE_DRAWING or point is None or tool not in DRAWING_TOOLS:
            self.reset_stroke()
            return

        erasing = tool == TOOL_ERASER
        thickness = ERASER_THICKNESS if erasing else BRUSH_THICKNESS
        mask_color = 0 if erasing else 255
        color = TOOL_COLORS.get(tool)

        if self.previous_point is not None:
            if color is not None:
                cv2.line(
                    self.colors,
                    self.previous_point,
                    point,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            cv2.line(
                self.mask,
                self.previous_point,
                point,
                mask_color,
                thickness,
                cv2.LINE_AA,
            )

        radius = thickness // 2
        if color is not None:
            cv2.circle(self.colors, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(self.mask, point, radius, mask_color, -1, cv2.LINE_AA)
        self.previous_point = point

    def composite(self, frame):
        """Blend active canvas pixels over the live frame using mask coverage."""
        active = self.mask > 0
        if not np.any(active):
            return frame

        alpha = self.mask[active].astype(np.float32)[:, np.newaxis] / 255.0
        frame_pixels = frame[active].astype(np.float32)
        color_pixels = self.colors[active].astype(np.float32)
        blended = frame_pixels * (1.0 - alpha) + color_pixels * alpha
        frame[active] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame


def save_drawing(surface: DrawingSurface, drawings_folder: Path) -> Path | None:
    """Write the color layer and alpha mask as a transparent BGRA PNG."""
    if not np.any(surface.mask):
        return None

    drawings_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"air_canvas_{timestamp}.png"
    output_path = drawings_folder / filename
    suffix = 1
    while output_path.exists():
        output_path = drawings_folder / f"air_canvas_{timestamp}_{suffix:02d}.png"
        suffix += 1

    bgra = np.dstack((surface.colors, surface.mask))
    try:
        written = cv2.imwrite(str(output_path), bgra)
    except cv2.error:
        if output_path.exists():
            output_path.unlink()
        raise
    if not written:
        if output_path.exists():
            output_path.unlink()
        raise OSError(f"Could not write PNG file: {output_path}")
    return output_path


def button_at(point):
    if point is None:
        return None
    for tool, (left, top, right, bottom) in TOOLBAR_BUTTONS.items():
        if left <= point[0] < right and top <= point[1] < bottom:
            return tool
    return None


def draw_ui_text(frame, text: str, origin, color, font_scale: float, thickness: int) -> None:
    """Draw outlined, anti-aliased interface text at the frame's native size."""
    cv2.putText(
        frame,
        text,
        origin,
        UI_FONT,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        UI_FONT,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class ToolbarController:
    """Track selected and hovered toolbar tools with a one-shot dwell trigger."""

    def __init__(self) -> None:
        self.selected_tool = TOOL_BLUE
        self.hovered_tool = None
        self.hover_started_at = None
        self.hover_committed = False
        self.feedback_text = None
        self.feedback_until = 0.0

    def reset_hover(self) -> None:
        self.hovered_tool = None
        self.hover_started_at = None
        self.hover_committed = False

    def show_feedback(
        self,
        message: str,
        now: float,
        duration: float = FEEDBACK_DURATION_SECONDS,
    ) -> None:
        self.feedback_text = message
        self.feedback_until = now + duration

    def update_hover(self, point, now: float):
        tool = button_at(point)
        if tool != self.hovered_tool:
            self.hovered_tool = tool
            self.hover_started_at = now if tool is not None else None
            self.hover_committed = False

        if (
            tool is None
            or self.hover_committed
            or now - self.hover_started_at < HOVER_DWELL_SECONDS
        ):
            return None

        self.hover_committed = True
        if tool == TOOL_CLEAR_ALL:
            self.show_feedback("Canvas cleared", now)
        else:
            self.selected_tool = tool
            self.show_feedback(f"{tool} selected", now)
        return tool


def save_current_drawing(
    surface: DrawingSurface,
    drawings_folder: Path,
    toolbar: ToolbarController,
    now: float,
) -> None:
    try:
        output_path = save_drawing(surface, drawings_folder)
    except (OSError, cv2.error) as error:
        toolbar.show_feedback(f"Save failed: {error}", now, 4.0)
        return

    if output_path is None:
        toolbar.show_feedback("Nothing to save", now)
    else:
        toolbar.show_feedback(f"Saved {output_path.name}", now, 3.0)


def draw_toolbar(frame, controller: ToolbarController, now: float) -> None:
    """Draw the toolbar, selected tool, hover state, and dwell progress."""
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (FRAME_WIDTH - 1, TOOLBAR_HEIGHT - 1),
        (28, 32, 40),
        -1,
    )
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    for tool, (left, top, right, bottom) in TOOLBAR_BUTTONS.items():
        fill = (52, 58, 68)
        if tool == TOOL_CLEAR_ALL:
            fill = (55, 48, 74)
        cv2.rectangle(frame, (left, top), (right - 1, bottom - 1), fill, -1)

        if tool in TOOL_COLORS:
            swatch_color = TOOL_COLORS[tool]
        elif tool == TOOL_ERASER:
            swatch_color = (190, 190, 190)
        else:
            swatch_color = (0, 0, 220)
        cv2.rectangle(
            frame,
            (left + 10, top + 14),
            (left + 34, bottom - 14),
            swatch_color,
            -1,
        )
        if tool == TOOL_ERASER:
            cv2.line(
                frame,
                (left + 13, top + 37),
                (left + 31, top + 19),
                (70, 70, 70),
                2,
                cv2.LINE_AA,
            )
        elif tool == TOOL_CLEAR_ALL:
            cv2.line(
                frame,
                (left + 15, top + 19),
                (left + 29, top + 33),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (left + 29, top + 19),
                (left + 15, top + 33),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        draw_ui_text(
            frame,
            tool,
            (left + 42, top + 34),
            (245, 245, 245),
            TOOLBAR_FONT_SCALE,
            TOOLBAR_FONT_THICKNESS,
        )

        if tool == controller.selected_tool:
            cv2.rectangle(
                frame,
                (left, top),
                (right - 1, bottom - 1),
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        if tool == controller.hovered_tool:
            cv2.rectangle(
                frame,
                (left + 4, top + 4),
                (right - 5, bottom - 5),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if controller.hover_started_at is not None:
                elapsed = now - controller.hover_started_at
                progress = min(1.0, elapsed / HOVER_DWELL_SECONDS)
                progress_right = left + 4 + round((right - left - 8) * progress)
                progress_color = (
                    (0, 220, 255) if controller.hover_committed else (255, 210, 0)
                )
                cv2.rectangle(
                    frame,
                    (left + 3, bottom - 4),
                    (progress_right, bottom - 2),
                    progress_color,
                    -1,
                )


def draw_status(
    frame,
    mode: str,
    selected_tool: str,
    hand_detected: bool,
    controller: ToolbarController,
    now: float,
) -> None:
    """Show mode, selected tool, selection feedback, and tracking status."""
    labels = [
        (f"Mode: {MODE_NAMES[mode]}", (255, 255, 255)),
        (f"Selected tool: {selected_tool}", (255, 255, 255)),
    ]
    if controller.feedback_text and now < controller.feedback_until:
        labels.append((controller.feedback_text, (0, 220, 0)))
    if not hand_detected:
        labels.append(("No hand detected", (0, 0, 255)))

    for line, (text, color) in enumerate(labels):
        origin = (18, 105 + line * 29)
        draw_ui_text(frame, text, origin, color, STATUS_FONT_SCALE, STATUS_FONT_THICKNESS)


def draw_selection_cursor(frame, point) -> None:
    if point is None:
        return
    cv2.circle(frame, point, 12, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, point, 10, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, point, 3, (0, 255, 255), -1, cv2.LINE_AA)


def run() -> None:
    capture = None
    landmarker = None
    try:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Error: could not open the default webcam.", file=sys.stderr)
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        if not model_path.is_file():
            print(f"Error: hand landmark model not found: {model_path}", file=sys.stderr)
            return

        landmarker = create_hand_landmarker(model_path)
        last_timestamp_ms = -1
        drawing_surface = DrawingSurface()
        toolbar = ToolbarController()
        drawings_folder = Path(__file__).resolve().parent / "drawings"

        while True:
            success, frame = capture.read()
            if not success or frame is None:
                print("Error: failed to capture a webcam frame.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = time.monotonic_ns() // 1_000_000
            timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            now = time.monotonic()

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                mode = classify_gesture(landmarks)
                point = normalized_to_pixel(landmarks[8], FRAME_WIDTH, FRAME_HEIGHT)
            else:
                landmarks = None
                mode = MODE_IDLE
                point = None

            if mode == MODE_SELECTION:
                selected_button = toolbar.update_hover(point, now)
                if selected_button == TOOL_CLEAR_ALL:
                    drawing_surface.clear()
            else:
                toolbar.reset_hover()

            drawing_surface.update_stroke(mode, point, toolbar.selected_tool)

            frame = drawing_surface.composite(frame)
            draw_toolbar(frame, toolbar, now)
            draw_status(
                frame,
                mode,
                toolbar.selected_tool,
                landmarks is not None,
                toolbar,
                now,
            )
            if mode == MODE_SELECTION:
                draw_selection_cursor(frame, point)
            if landmarks is not None:
                draw_hand(frame, landmarks)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                save_current_drawing(
                    drawing_surface,
                    drawings_folder,
                    toolbar,
                    time.monotonic(),
                )
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except (RuntimeError, cv2.error) as error:
                print(f"Error closing hand landmarker: {error}", file=sys.stderr)
        if capture is not None:
            try:
                capture.release()
            except cv2.error as error:
                print(f"Error releasing webcam: {error}", file=sys.stderr)
        try:
            cv2.destroyAllWindows()
        except cv2.error as error:
            print(f"Error closing OpenCV windows: {error}", file=sys.stderr)


if __name__ == "__main__":
    run()

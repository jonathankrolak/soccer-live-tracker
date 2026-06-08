import time

import cv2
import mss
import numpy as np
from ultralytics import YOLO
from src.pitch_filter import is_on_pitch
import json
import os
from src.field_line_detector import detect_field_lines, draw_field_line_debug
from src.config import CAPTURE_MONITOR_INDEX, DISPLAY_MONITOR_INDEX
from src.window_utils import move_window_to_monitor
from datetime import datetime

from src.color_classifier import get_jersey_color
from src.config import (
    CONFIDENCE_THRESHOLD,
    CROP_HEIGHT,
    CROP_LEFT,
    CROP_TOP,
    CROP_WIDTH,
    RED,
    YOLO_IMAGE_SIZE,
    YELLOW,
    SHOW_DEBUG_LABELS,
    SHOW_FIELD_LINES_DEBUG,
)
from src.field_view import draw_field
from src.mapper import map_screen_to_field
from src.smoothing import smooth_positions

def load_crop_config():
    config_path = "crop_config.json"

    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return json.load(file)

    return {
        "top": CROP_TOP,
        "left": CROP_LEFT,
        "width": CROP_WIDTH,
        "height": CROP_HEIGHT,
    }

def save_training_frame(frame):
    output_dir = "dataset/raw_frames"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")

    cv2.imwrite(file_path, frame)

    print(f"Saved training frame: {file_path}")

def main():
    model = YOLO("yolov8n.pt")
    previous_player_positions = []

    with mss.mss() as sct:
        capture_monitor = sct.monitors[CAPTURE_MONITOR_INDEX]
        crop = load_crop_config()

        monitor = {
            "left": capture_monitor["left"] + crop["left"],
            "top": capture_monitor["top"] + crop["top"],
            "width": crop["width"],
            "height": crop["height"],
        }

        screen_width = monitor["width"]
        screen_height = monitor["height"]

        prev_time = time.time()

        windows_positioned = False

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if SHOW_FIELD_LINES_DEBUG:
                grass_mask, white_line_mask, detected_lines, intersections = detect_field_lines(frame)
                field_line_debug, grass_preview, white_preview = draw_field_line_debug(
                    frame,
                    grass_mask,
                    white_line_mask,
                    detected_lines,
                    intersections,
                )

                cv2.imshow("Field Line Debug", field_line_debug)
                cv2.imshow("Grass Mask", grass_preview)
                cv2.imshow("White Line Mask", white_preview)

            results = model(frame, imgsz=YOLO_IMAGE_SIZE, verbose=False)

            current_player_positions = []

            light_blue_count = 0
            dark_blue_count = 0
            unknown_count = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if class_name != "person" or confidence <= CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    dot_color, team_label = get_jersey_color(frame, x1, y1, x2, y2)

                    if team_label == "chelsea":
                        light_blue_count += 1
                    elif team_label == "man united":
                        dark_blue_count += 1
                    else:
                        unknown_count += 1

                    if SHOW_DEBUG_LABELS:
                        # Draw player box on broadcast
                        cv2.rectangle(frame, (x1, y1), (x2, y2), dot_color, 2)

                    if SHOW_DEBUG_LABELS:
                        # Label detected team color
                        cv2.putText(
                            frame,
                            team_label,
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            dot_color,
                            2,
                        )
                    

                    # Bottom-center point of player box = estimated feet position
                    player_screen_x = int((x1 + x2) / 2)
                    player_screen_y = int(y2)

                    
                    if SHOW_DEBUG_LABELS:
                        # Ignore detections whose feet are not on the pitch.
                        if not is_on_pitch(frame, player_screen_x, player_screen_y):
                            cv2.circle(frame, (player_screen_x, player_screen_y), 5, (0, 0, 0), -1)
                            cv2.putText(
                                frame,
                                "off pitch",
                                (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 0, 0),
                                2,
                            )
                            continue

                        # Red dot on broadcast shows the point we map to the field
                        cv2.circle(frame, (player_screen_x, player_screen_y), 5, RED, -1)

                    field_x, field_y = map_screen_to_field(
                        player_screen_x,
                        player_screen_y,
                        screen_width,
                        screen_height,
                    )

                    current_player_positions.append((field_x, field_y, dot_color, team_label))

            smoothed_player_positions = smooth_positions(
                previous_player_positions,
                current_player_positions,
            )

            previous_player_positions = smoothed_player_positions

            field_view = draw_field(smoothed_player_positions)

            # FPS counter
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            if SHOW_DEBUG_LABELS:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    YELLOW,
                    2,
                )

            if SHOW_DEBUG_LABELS:
                cv2.putText(
                    frame,
                    f"Chelsea: {light_blue_count}  Man Utd: {dark_blue_count}  Unknown: {unknown_count}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    YELLOW,
                    2,
                )

            cv2.imshow("Broadcast Detection", frame)
            cv2.imshow("Top Down Field", field_view)

            if SHOW_FIELD_LINES_DEBUG:
                cv2.imshow("Field Line Debug", field_line_debug)
                cv2.imshow("Grass Mask", grass_preview)
                cv2.imshow("White Line Mask", white_preview)

            if not windows_positioned:
                cv2.waitKey(1)

                move_window_to_monitor(sct, "Broadcast Detection", DISPLAY_MONITOR_INDEX, 40, 40)
                move_window_to_monitor(sct, "Top Down Field", DISPLAY_MONITOR_INDEX, 40, 500)

                if SHOW_FIELD_LINES_DEBUG:
                    move_window_to_monitor(sct, "Field Line Debug", DISPLAY_MONITOR_INDEX, 700, 40)
                    move_window_to_monitor(sct, "Grass Mask", DISPLAY_MONITOR_INDEX, 700, 500)
                    move_window_to_monitor(sct, "White Line Mask", DISPLAY_MONITOR_INDEX, 700, 750)

                windows_positioned = True

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                save_training_frame(frame)

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
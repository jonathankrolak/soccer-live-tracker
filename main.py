import time

import cv2
import mss
import numpy as np
from ultralytics import YOLO
from src.pitch_filter import is_on_pitch

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
)
from src.field_view import draw_field
from src.mapper import map_screen_to_field
from src.smoothing import smooth_positions


def main():
    model = YOLO("yolov8n.pt")
    previous_player_positions = []

    with mss.mss() as sct:
        monitor = {
            "top": CROP_TOP,
            "left": CROP_LEFT,
            "width": CROP_WIDTH,
            "height": CROP_HEIGHT,
        }

        screen_width = monitor["width"]
        screen_height = monitor["height"]

        prev_time = time.time()

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

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

                    # Draw player box on broadcast
                    cv2.rectangle(frame, (x1, y1), (x2, y2), dot_color, 2)

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

                    current_player_positions.append((field_x, field_y, dot_color))

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

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                YELLOW,
                2,
            )

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

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
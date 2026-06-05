import time

import cv2
import numpy as np
import mss
from ultralytics import YOLO


FIELD_WIDTH = 900
FIELD_HEIGHT = 550

SMOOTHING = 0.75
MATCH_DISTANCE = 80

CROP_TOP = 70
CROP_LEFT = 20
CROP_WIDTH = 1200
CROP_HEIGHT = 760


def draw_field(player_positions):
    field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH, 3), dtype=np.uint8)

    # Green background
    field[:] = (40, 130, 40)

    # Outer boundary
    cv2.rectangle(
        field,
        (20, 20),
        (FIELD_WIDTH - 20, FIELD_HEIGHT - 20),
        (255, 255, 255),
        2
    )

    # Halfway line
    cv2.line(
        field,
        (FIELD_WIDTH // 2, 20),
        (FIELD_WIDTH // 2, FIELD_HEIGHT - 20),
        (255, 255, 255),
        2
    )

    # Center circle
    cv2.circle(
        field,
        (FIELD_WIDTH // 2, FIELD_HEIGHT // 2),
        70,
        (255, 255, 255),
        2
    )

    # Center spot
    cv2.circle(
        field,
        (FIELD_WIDTH // 2, FIELD_HEIGHT // 2),
        4,
        (255, 255, 255),
        -1
    )

    # Left penalty box
    cv2.rectangle(
        field,
        (20, 150),
        (140, FIELD_HEIGHT - 150),
        (255, 255, 255),
        2
    )

    # Right penalty box
    cv2.rectangle(
        field,
        (FIELD_WIDTH - 140, 150),
        (FIELD_WIDTH - 20, FIELD_HEIGHT - 150),
        (255, 255, 255),
        2
    )

    # Draw players
    for x, y in player_positions:
        cv2.circle(field, (int(x), int(y)), 8, (255, 255, 255), -1)

    return field


def map_screen_to_field(x, y, screen_width, screen_height):
    field_x = int((x / screen_width) * (FIELD_WIDTH - 40)) + 20
    field_y = int((y / screen_height) * (FIELD_HEIGHT - 40)) + 20

    return field_x, field_y


def distance(point_a, point_b):
    ax, ay = point_a
    bx, by = point_b

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def smooth_positions(previous_positions, current_positions):
    smoothed_positions = []
    used_previous_indexes = set()

    for current in current_positions:
        best_index = None
        best_distance = float("inf")

        for i, previous in enumerate(previous_positions):
            if i in used_previous_indexes:
                continue

            dist = distance(current, previous)

            if dist < best_distance:
                best_distance = dist
                best_index = i

        if best_index is not None and best_distance < MATCH_DISTANCE:
            previous = previous_positions[best_index]
            used_previous_indexes.add(best_index)

            smooth_x = previous[0] * SMOOTHING + current[0] * (1 - SMOOTHING)
            smooth_y = previous[1] * SMOOTHING + current[1] * (1 - SMOOTHING)

            smoothed_positions.append((smooth_x, smooth_y))
        else:
            smoothed_positions.append(current)

    return smoothed_positions


def main():
    model = YOLO("yolov8n.pt")

    previous_player_positions = []

    with mss.mss() as sct:
        # Crop only the broadcast area.
        # Tweak these numbers if your video is shifted or cut off.
        monitor = {
            "top": CROP_TOP,
            "left": CROP_LEFT,
            "width": CROP_WIDTH,
            "height": CROP_HEIGHT
        }

        screen_width = monitor["width"]
        screen_height = monitor["height"]

        prev_time = time.time()

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLO on the cropped broadcast.
            # Lower imgsz = faster, but less accurate.
            results = model(frame, imgsz=640, verbose=False)

            current_player_positions = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if class_name == "person" and confidence > 0.4:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Draw player box on broadcast
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Bottom-center point of the player box
                        player_screen_x = int((x1 + x2) / 2)
                        player_screen_y = int(y2)

                        # Red dot = player's estimated feet position
                        cv2.circle(frame, (player_screen_x, player_screen_y), 5, (0, 0, 255), -1)

                        field_x, field_y = map_screen_to_field(
                            player_screen_x,
                            player_screen_y,
                            screen_width,
                            screen_height
                        )

                        current_player_positions.append((field_x, field_y))

            smoothed_player_positions = smooth_positions(
                previous_player_positions,
                current_player_positions
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
                (0, 255, 255),
                2
            )

            cv2.imshow("Broadcast Detection", frame)
            cv2.imshow("Top Down Field", field_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
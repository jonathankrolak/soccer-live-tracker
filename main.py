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

CONFIDENCE_THRESHOLD = 0.4

# OpenCV uses BGR, not RGB
LIGHT_BLUE = (255, 220, 120)
DARK_BLUE = (180, 60, 20)
UNKNOWN_COLOR = (200, 200, 200)


def draw_field(player_positions):
    field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH, 3), dtype=np.uint8)

    # Green background
    field[:] = (40, 130, 40)

    # Outer boundary
    cv2.rectangle(field, (20, 20), (FIELD_WIDTH - 20, FIELD_HEIGHT - 20), (255, 255, 255), 2)

    # Halfway line
    cv2.line(field, (FIELD_WIDTH // 2, 20), (FIELD_WIDTH // 2, FIELD_HEIGHT - 20), (255, 255, 255), 2)

    # Center circle and spot
    cv2.circle(field, (FIELD_WIDTH // 2, FIELD_HEIGHT // 2), 70, (255, 255, 255), 2)
    cv2.circle(field, (FIELD_WIDTH // 2, FIELD_HEIGHT // 2), 4, (255, 255, 255), -1)

    # Penalty boxes
    cv2.rectangle(field, (20, 150), (140, FIELD_HEIGHT - 150), (255, 255, 255), 2)
    cv2.rectangle(field, (FIELD_WIDTH - 140, 150), (FIELD_WIDTH - 20, FIELD_HEIGHT - 150), (255, 255, 255), 2)

    # Draw players
    for x, y, color in player_positions:
        cv2.circle(field, (int(x), int(y)), 9, color, -1)
        cv2.circle(field, (int(x), int(y)), 9, (255, 255, 255), 1)

    return field


def map_screen_to_field(x, y, screen_width, screen_height):
    field_x = int((x / screen_width) * (FIELD_WIDTH - 40)) + 20
    field_y = int((y / screen_height) * (FIELD_HEIGHT - 40)) + 20

    return field_x, field_y


def distance(point_a, point_b):
    ax, ay = point_a[0], point_a[1]
    bx, by = point_b[0], point_b[1]

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

            # Keep the newest detected color
            color = current[2]

            smoothed_positions.append((smooth_x, smooth_y, color))
        else:
            smoothed_positions.append(current)

    return smoothed_positions


def get_jersey_color(frame, x1, y1, x2, y2):
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return UNKNOWN_COLOR, "unknown"

    # Sample upper-middle part of the player box.
    # This should roughly be the jersey/chest area.
    jersey_x1 = x1 + int(box_width * 0.25)
    jersey_x2 = x1 + int(box_width * 0.75)
    jersey_y1 = y1 + int(box_height * 0.20)
    jersey_y2 = y1 + int(box_height * 0.55)

    jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

    if jersey_crop.size == 0:
        return UNKNOWN_COLOR, "unknown"

    # Average BGR color in jersey area
    avg_color = jersey_crop.mean(axis=(0, 1))
    blue = avg_color[0]
    green = avg_color[1]
    red = avg_color[2]

    brightness = (blue + green + red) / 3

    # Debug color box on broadcast
    avg_bgr = (int(blue), int(green), int(red))

    # Simple rules for this game:
    # Light blue usually has high blue + decent green + higher brightness.
    # Dark blue usually has high blue compared to red/green but lower brightness.
    if blue > red + 20 and blue > green - 10 and brightness > 95:
        return LIGHT_BLUE, "light blue"

    if blue > red + 15 and brightness <= 95:
        return DARK_BLUE, "dark blue"

    return UNKNOWN_COLOR, "unknown"


def main():
    model = YOLO("yolov8n.pt")

    previous_player_positions = []

    with mss.mss() as sct:
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

            results = model(frame, imgsz=640, verbose=False)

            current_player_positions = []

            light_blue_count = 0
            dark_blue_count = 0
            unknown_count = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if class_name == "person" and confidence > CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        dot_color, team_label = get_jersey_color(frame, x1, y1, x2, y2)

                        if team_label == "light blue":
                            light_blue_count += 1
                        elif team_label == "dark blue":
                            dark_blue_count += 1
                        else:
                            unknown_count += 1

                        # Draw player box on broadcast
                        cv2.rectangle(frame, (x1, y1), (x2, y2), dot_color, 2)

                        # Label the detected team color
                        cv2.putText(
                            frame,
                            team_label,
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            dot_color,
                            2
                        )

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

                        current_player_positions.append((field_x, field_y, dot_color))

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

            cv2.putText(
                frame,
                f"Light: {light_blue_count}  Dark: {dark_blue_count}  Unknown: {unknown_count}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
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
import json
import os

import cv2
import mss
import numpy as np


CROP_CONFIG_PATH = "crop_config.json"
HOMOGRAPHY_CONFIG_PATH = "homography_config.json"

WINDOW_NAME = "Homography Calibration"

FIELD_WIDTH = 900
FIELD_HEIGHT = 550

FIELD_MARGIN = 20

# These match how we currently draw the penalty boxes in field_view.py
RIGHT_PENALTY_BOX_LEFT = FIELD_WIDTH - 140
RIGHT_PENALTY_BOX_RIGHT = FIELD_WIDTH - 20
RIGHT_PENALTY_BOX_TOP = 150
RIGHT_PENALTY_BOX_BOTTOM = FIELD_HEIGHT - 150

LEFT_PENALTY_BOX_LEFT = 20
LEFT_PENALTY_BOX_RIGHT = 140
LEFT_PENALTY_BOX_TOP = 150
LEFT_PENALTY_BOX_BOTTOM = FIELD_HEIGHT - 150

clicked_points = []

# Change this depending on what field area you are calibrating.
# Options:
# "right_penalty_box"
# "left_penalty_box"
# "full_field"
CALIBRATION_MODE = "right_penalty_box"


def load_crop_config():
    if not os.path.exists(CROP_CONFIG_PATH):
        raise FileNotFoundError(
            "crop_config.json not found. Run calibrate_crop.py first."
        )

    with open(CROP_CONFIG_PATH, "r") as file:
        return json.load(file)


def save_homography_config(source_points, destination_points, matrix):
    config = {
        "mode": CALIBRATION_MODE,
        "source_points": source_points.tolist(),
        "destination_points": destination_points.tolist(),
        "matrix": matrix.tolist(),
    }

    with open(HOMOGRAPHY_CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=4)

    print(f"Saved homography to {HOMOGRAPHY_CONFIG_PATH}")
    print(f"Mode: {CALIBRATION_MODE}")


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Point {len(clicked_points)}: {(x, y)}")


def draw_instruction_panel(frame):
    instructions = [
        f"Mode: {CALIBRATION_MODE}",
        "Click 4 corners of the selected field area:",
        "1. top-left corner",
        "2. top-right corner",
        "3. bottom-right corner",
        "4. bottom-left corner",
        "Press R to reset | S to save | Q to quit",
    ]

    y = 35

    for text in instructions:
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        y += 30


def draw_clicked_points(frame):
    for index, point in enumerate(clicked_points):
        x, y = point

        cv2.circle(frame, (x, y), 7, (0, 255, 255), -1)
        cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)

        cv2.putText(
            frame,
            str(index + 1),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    if len(clicked_points) >= 2:
        for i in range(len(clicked_points) - 1):
            cv2.line(
                frame,
                clicked_points[i],
                clicked_points[i + 1],
                (0, 255, 255),
                2,
            )

    if len(clicked_points) == 4:
        cv2.line(
            frame,
            clicked_points[3],
            clicked_points[0],
            (0, 255, 255),
            2,
        )


def get_destination_points():
    if CALIBRATION_MODE == "right_penalty_box":
        return np.array(
            [
                [RIGHT_PENALTY_BOX_LEFT, RIGHT_PENALTY_BOX_TOP],
                [RIGHT_PENALTY_BOX_RIGHT, RIGHT_PENALTY_BOX_TOP],
                [RIGHT_PENALTY_BOX_RIGHT, RIGHT_PENALTY_BOX_BOTTOM],
                [RIGHT_PENALTY_BOX_LEFT, RIGHT_PENALTY_BOX_BOTTOM],
            ],
            dtype=np.float32,
        )

    if CALIBRATION_MODE == "left_penalty_box":
        return np.array(
            [
                [LEFT_PENALTY_BOX_LEFT, LEFT_PENALTY_BOX_TOP],
                [LEFT_PENALTY_BOX_RIGHT, LEFT_PENALTY_BOX_TOP],
                [LEFT_PENALTY_BOX_RIGHT, LEFT_PENALTY_BOX_BOTTOM],
                [LEFT_PENALTY_BOX_LEFT, LEFT_PENALTY_BOX_BOTTOM],
            ],
            dtype=np.float32,
        )

    if CALIBRATION_MODE == "full_field":
        return np.array(
            [
                [FIELD_MARGIN, FIELD_MARGIN],
                [FIELD_WIDTH - FIELD_MARGIN, FIELD_MARGIN],
                [FIELD_WIDTH - FIELD_MARGIN, FIELD_HEIGHT - FIELD_MARGIN],
                [FIELD_MARGIN, FIELD_HEIGHT - FIELD_MARGIN],
            ],
            dtype=np.float32,
        )

    raise ValueError(f"Unknown CALIBRATION_MODE: {CALIBRATION_MODE}")


def main():
    global clicked_points

    crop = load_crop_config()

    with mss.mss() as sct:
        screenshot = sct.grab(crop)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        display_frame = frame.copy()

        draw_instruction_panel(display_frame)
        draw_clicked_points(display_frame)

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            clicked_points = []
            print("Reset clicked points.")

        elif key == ord("s"):
            if len(clicked_points) != 4:
                print("You need exactly 4 points before saving.")
                continue

            source_points = np.array(clicked_points, dtype=np.float32)
            destination_points = get_destination_points()

            matrix = cv2.getPerspectiveTransform(
                source_points,
                destination_points,
            )

            save_homography_config(source_points, destination_points, matrix)

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import json
import os

import cv2
import mss
import numpy as np
from src.config import CAPTURE_MONITOR_INDEX, DISPLAY_MONITOR_INDEX
from src.window_utils import move_window_to_monitor


CONFIG_PATH = "crop_config.json"

WINDOW_NAME = "Crop Calibration"

# Starting crop if no saved config exists yet
DEFAULT_CROP = {
    "left": 20,
    "top": 70,
    "width": 1200,
    "height": 760,
}

HANDLE_RADIUS = 10

dragging_corner = None
crop = DEFAULT_CROP.copy()


def load_crop():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as file:
            return json.load(file)

    return DEFAULT_CROP.copy()


def save_crop(crop_data):
    with open(CONFIG_PATH, "w") as file:
        json.dump(crop_data, file, indent=4)


def get_corners(crop_data):
    left = crop_data["left"]
    top = crop_data["top"]
    right = crop_data["left"] + crop_data["width"]
    bottom = crop_data["top"] + crop_data["height"]

    return {
        "top_left": (left, top),
        "top_right": (right, top),
        "bottom_left": (left, bottom),
        "bottom_right": (right, bottom),
    }


def point_distance(point_a, point_b):
    ax, ay = point_a
    bx, by = point_b

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def find_corner_near_mouse(x, y, corners):
    mouse_point = (x, y)

    for corner_name, corner_point in corners.items():
        if point_distance(mouse_point, corner_point) <= HANDLE_RADIUS * 2:
            return corner_name

    return None


def update_crop_from_corner(crop_data, corner_name, x, y):
    left = crop_data["left"]
    top = crop_data["top"]
    right = crop_data["left"] + crop_data["width"]
    bottom = crop_data["top"] + crop_data["height"]

    if corner_name == "top_left":
        left = x
        top = y
    elif corner_name == "top_right":
        right = x
        top = y
    elif corner_name == "bottom_left":
        left = x
        bottom = y
    elif corner_name == "bottom_right":
        right = x
        bottom = y

    # Prevent inverted / tiny rectangles
    min_width = 100
    min_height = 100

    if right - left < min_width:
        return crop_data

    if bottom - top < min_height:
        return crop_data

    crop_data["left"] = int(left)
    crop_data["top"] = int(top)
    crop_data["width"] = int(right - left)
    crop_data["height"] = int(bottom - top)

    return crop_data


def mouse_callback(event, x, y, flags, param):
    global dragging_corner, crop

    corners = get_corners(crop)

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging_corner = find_corner_near_mouse(x, y, corners)

    elif event == cv2.EVENT_MOUSEMOVE and dragging_corner is not None:
        crop = update_crop_from_corner(crop, dragging_corner, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_corner = None


def draw_crop_overlay(frame, crop_data):
    overlay = frame.copy()

    left = crop_data["left"]
    top = crop_data["top"]
    right = crop_data["left"] + crop_data["width"]
    bottom = crop_data["top"] + crop_data["height"]

    # Darken everything
    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], frame.shape[0]),
        (0, 0, 0),
        -1,
    )

    # Put original crop area back over the dark overlay
    overlay[top:bottom, left:right] = frame[top:bottom, left:right]

    # Blend overlay with frame
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Crop rectangle
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

    # Corner handles
    corners = get_corners(crop_data)
    for corner in corners.values():
        cv2.circle(frame, corner, HANDLE_RADIUS, (0, 255, 255), -1)
        cv2.circle(frame, corner, HANDLE_RADIUS, (0, 0, 0), 2)

    # Instructions
    cv2.putText(
        frame,
        "Drag corners to set broadcast crop",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Press S to save | Q to quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"left={left}, top={top}, width={crop_data['width']}, height={crop_data['height']}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return frame


def main():
    global crop

    crop = load_crop()

    with mss.mss() as sct:
        monitor = sct.monitors[CAPTURE_MONITOR_INDEX]

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        move_window_to_monitor(sct, WINDOW_NAME, DISPLAY_MONITOR_INDEX, 40, 40)

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            display_frame = draw_crop_overlay(frame, crop)

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                save_crop(crop)
                print(f"Saved crop to {CONFIG_PATH}: {crop}")

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
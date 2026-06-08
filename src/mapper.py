import json
import os

import cv2
import numpy as np

from src.config import FIELD_WIDTH, FIELD_HEIGHT


HOMOGRAPHY_CONFIG_PATH = "homography_config.json"

homography_matrix = None


def load_homography_matrix():
    global homography_matrix

    if homography_matrix is not None:
        return homography_matrix

    if not os.path.exists(HOMOGRAPHY_CONFIG_PATH):
        return None

    with open(HOMOGRAPHY_CONFIG_PATH, "r") as file:
        config = json.load(file)

    homography_matrix = np.array(config["matrix"], dtype=np.float32)

    return homography_matrix


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def map_with_homography(x, y):
    matrix = load_homography_matrix()

    if matrix is None:
        return None

    point = np.array([[[x, y]]], dtype=np.float32)

    transformed = cv2.perspectiveTransform(point, matrix)

    field_x = int(transformed[0][0][0])
    field_y = int(transformed[0][0][1])

    field_x = clamp(field_x, 20, FIELD_WIDTH - 20)
    field_y = clamp(field_y, 20, FIELD_HEIGHT - 20)

    return field_x, field_y


def map_simple_scale(x, y, screen_width, screen_height):
    field_x = int((x / screen_width) * (FIELD_WIDTH - 40)) + 20
    field_y = int((y / screen_height) * (FIELD_HEIGHT - 40)) + 20

    return field_x, field_y


def map_screen_to_field(x, y, screen_width, screen_height):
    homography_result = map_with_homography(x, y)

    if homography_result is not None:
        return homography_result

    return map_simple_scale(x, y, screen_width, screen_height)
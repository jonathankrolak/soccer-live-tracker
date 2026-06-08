import cv2
import numpy as np

from src.config import (
    FIELD_WIDTH,
    FIELD_HEIGHT,
    GREEN_FIELD,
    WHITE,
)


def draw_field(player_positions):
    field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH, 3), dtype=np.uint8)

    # Green background
    field[:] = GREEN_FIELD

    # Outer boundary
    cv2.rectangle(
        field,
        (20, 20),
        (FIELD_WIDTH - 20, FIELD_HEIGHT - 20),
        WHITE,
        2
    )

    # Halfway line
    cv2.line(
        field,
        (FIELD_WIDTH // 2, 20),
        (FIELD_WIDTH // 2, FIELD_HEIGHT - 20),
        WHITE,
        2
    )

    # Center circle and center spot
    cv2.circle(
        field,
        (FIELD_WIDTH // 2, FIELD_HEIGHT // 2),
        70,
        WHITE,
        2
    )

    cv2.circle(
        field,
        (FIELD_WIDTH // 2, FIELD_HEIGHT // 2),
        4,
        WHITE,
        -1
    )

    # Left penalty box
    cv2.rectangle(
        field,
        (20, 150),
        (140, FIELD_HEIGHT - 150),
        WHITE,
        2
    )

    # Right penalty box
    cv2.rectangle(
        field,
        (FIELD_WIDTH - 140, 150),
        (FIELD_WIDTH - 20, FIELD_HEIGHT - 150),
        WHITE,
        2
    )

    # Players
    for x, y, color in player_positions:
        cv2.circle(field, (int(x), int(y)), 9, color, -1)
        cv2.circle(field, (int(x), int(y)), 9, WHITE, 1)

    return field
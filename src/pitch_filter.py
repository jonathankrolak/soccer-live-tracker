from src.config import (
    PITCH_GREEN_THRESHOLD,
    PITCH_MAX_Y_RATIO,
    PITCH_MIN_Y_RATIO,
    PITCH_SAMPLE_RADIUS,
)


def patch_looks_like_grass(frame, center_x, center_y):
    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, center_x - PITCH_SAMPLE_RADIUS)
    x2 = min(frame_width, center_x + PITCH_SAMPLE_RADIUS)
    y1 = max(0, center_y - PITCH_SAMPLE_RADIUS)
    y2 = min(frame_height, center_y + PITCH_SAMPLE_RADIUS)

    sample = frame[y1:y2, x1:x2]

    if sample.size == 0:
        return False

    avg_color = sample.mean(axis=(0, 1))

    # OpenCV uses BGR
    blue = avg_color[0]
    green = avg_color[1]
    red = avg_color[2]

    brightness = (blue + green + red) / 3

    green_is_strong = green > red - 8 and green > blue - 8
    bright_enough = brightness > 40
    green_enough = green > PITCH_GREEN_THRESHOLD

    return green_is_strong and bright_enough and green_enough


def is_on_pitch(frame, foot_x, foot_y):
    frame_height, frame_width = frame.shape[:2]

    min_y = int(frame_height * PITCH_MIN_Y_RATIO)
    max_y = int(frame_height * PITCH_MAX_Y_RATIO)

    if foot_y < min_y or foot_y > max_y:
        return False

    if foot_x < 0 or foot_x >= frame_width:
        return False

    # Check multiple nearby points instead of only one exact foot point.
    # This helps when players overlap or one player blocks the grass under another.
    sample_points = [
        (foot_x, foot_y + 12),       # below feet
        (foot_x - 18, foot_y + 12),  # below-left
        (foot_x + 18, foot_y + 12),  # below-right
        (foot_x - 25, foot_y),       # left of feet
        (foot_x + 25, foot_y),       # right of feet
    ]

    grass_hits = 0

    for sample_x, sample_y in sample_points:
        sample_x = max(0, min(frame_width - 1, sample_x))
        sample_y = max(0, min(frame_height - 1, sample_y))

        if patch_looks_like_grass(frame, sample_x, sample_y):
            grass_hits += 1

    # If at least one nearby area looks like grass, assume the player is on the pitch.
    return grass_hits >= 1
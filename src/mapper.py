from src.config import (
    FIELD_WIDTH,
    FIELD_HEIGHT,
    MAP_BOTTOM,
    MAP_LEFT,
    MAP_RIGHT,
    MAP_TOP,
)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def map_screen_to_field(x, y, screen_width, screen_height):
    # Clamp the point inside the usable pitch area.
    x = clamp(x, MAP_LEFT, MAP_RIGHT)
    y = clamp(y, MAP_TOP, MAP_BOTTOM)

    pitch_width = MAP_RIGHT - MAP_LEFT
    pitch_height = MAP_BOTTOM - MAP_TOP

    normalized_x = (x - MAP_LEFT) / pitch_width
    normalized_y = (y - MAP_TOP) / pitch_height

    field_x = int(normalized_x * (FIELD_WIDTH - 40)) + 20
    field_y = int(normalized_y * (FIELD_HEIGHT - 40)) + 20

    return field_x, field_y
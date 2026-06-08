from src.config import FIELD_WIDTH, FIELD_HEIGHT


def map_screen_to_field(x, y, screen_width, screen_height):
    field_x = int((x / screen_width) * (FIELD_WIDTH - 40)) + 20
    field_y = int((y / screen_height) * (FIELD_HEIGHT - 40)) + 20

    return field_x, field_y
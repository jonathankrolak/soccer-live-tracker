from src.config import CHELSEA_BLUE, MAN_UNITED_RED, UNKNOWN_COLOR


def get_jersey_color(frame, x1, y1, x2, y2):
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return UNKNOWN_COLOR, "unknown"

    # Sample the upper-middle of the box.
    # This is usually the chest/jersey area.
    jersey_x1 = x1 + int(box_width * 0.25)
    jersey_x2 = x1 + int(box_width * 0.75)
    jersey_y1 = y1 + int(box_height * 0.20)
    jersey_y2 = y1 + int(box_height * 0.55)

    jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

    if jersey_crop.size == 0:
        return UNKNOWN_COLOR, "unknown"

    # OpenCV uses BGR order, not RGB
    avg_color = jersey_crop.mean(axis=(0, 1))
    blue = avg_color[0]
    green = avg_color[1]
    red = avg_color[2]

    brightness = (blue + green + red) / 3

   # Very dark kits / refs / shadows
    if brightness < 45:
        return UNKNOWN_COLOR, "unknown"

    # Man United red:
    # Keep this first because red is working well.
    if red > blue + 25 and red > green + 20:
        return MAN_UNITED_RED, "man united"

    # Chelsea blue:
    # Looser rule because Chelsea kits can look dark or shadowed.
    if blue > red + 5 and blue > green - 15 and brightness > 45:
        return CHELSEA_BLUE, "chelsea"

    return UNKNOWN_COLOR, "unknown"
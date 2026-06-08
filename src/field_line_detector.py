import cv2
import numpy as np
import math

from src.config import (
    GRASS_MAX_HUE,
    GRASS_MIN_HUE,
    GRASS_MIN_SATURATION,
    GRASS_MIN_VALUE,
    MAX_LINE_GAP,
    MIN_LINE_LENGTH,
    WHITE_LINE_MAX_SATURATION,
    WHITE_LINE_MIN_VALUE,
)


def create_grass_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array(
        [GRASS_MIN_HUE, GRASS_MIN_SATURATION, GRASS_MIN_VALUE],
        dtype=np.uint8,
    )

    upper_green = np.array(
        [GRASS_MAX_HUE, 255, 255],
        dtype=np.uint8,
    )

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def create_white_line_mask(frame, grass_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array(
        [0, 0, WHITE_LINE_MIN_VALUE],
        dtype=np.uint8,
    )

    upper_white = np.array(
        [179, WHITE_LINE_MAX_SATURATION, 255],
        dtype=np.uint8,
    )

    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Expand grass area so white field lines near grass are included.
    grass_kernel = np.ones((15, 15), np.uint8)
    expanded_grass_mask = cv2.dilate(grass_mask, grass_kernel, iterations=1)

    # Keep white pixels that are near the pitch.
    white_near_grass = cv2.bitwise_and(white_mask, expanded_grass_mask)

    # Clean and slightly thicken the line mask.
    close_kernel = np.ones((5, 5), np.uint8)
    white_near_grass = cv2.morphologyEx(white_near_grass, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = np.ones((3, 3), np.uint8)
    white_near_grass = cv2.dilate(white_near_grass, dilate_kernel, iterations=1)

    return white_near_grass

def line_has_grass_nearby(grass_mask, x1, y1, x2, y2):
    frame_height, frame_width = grass_mask.shape[:2]

    # Sample a few points along the detected line
    sample_count = 8
    grass_hits = 0

    for i in range(sample_count):
        t = i / max(sample_count - 1, 1)

        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)

        # Look in a small area around this point
        radius = 10

        px1 = max(0, x - radius)
        px2 = min(frame_width, x + radius)
        py1 = max(0, y - radius)
        py2 = min(frame_height, y + radius)

        patch = grass_mask[py1:py2, px1:px2]

        if patch.size == 0:
            continue

        grass_ratio = patch.mean() / 255

        if grass_ratio > 0.25:
            grass_hits += 1

    # Most sampled points should be near grass
    return grass_hits >= 5

def get_line_length(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def get_line_angle(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    # Normalize angle to 0-180
    if angle < 0:
        angle += 180

    return angle


def get_line_midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def lines_are_similar(line_a, line_b):
    ax1, ay1, ax2, ay2 = line_a
    bx1, by1, bx2, by2 = line_b

    angle_a = get_line_angle(ax1, ay1, ax2, ay2)
    angle_b = get_line_angle(bx1, by1, bx2, by2)

    angle_diff = abs(angle_a - angle_b)
    angle_diff = min(angle_diff, 180 - angle_diff)

    if angle_diff > 8:
        return False

    a_mid_x, a_mid_y = get_line_midpoint(ax1, ay1, ax2, ay2)
    b_mid_x, b_mid_y = get_line_midpoint(bx1, by1, bx2, by2)

    midpoint_distance = ((a_mid_x - b_mid_x) ** 2 + (a_mid_y - b_mid_y) ** 2) ** 0.5

    return midpoint_distance < 90


def merge_line_group(lines):
    points = []

    for x1, y1, x2, y2 in lines:
        points.append((x1, y1))
        points.append((x2, y2))

    # Use the two farthest points as the merged line endpoints
    best_pair = None
    best_distance = -1

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]

            dist = get_line_length(x1, y1, x2, y2)

            if dist > best_distance:
                best_distance = dist
                best_pair = (x1, y1, x2, y2)

    return best_pair


def merge_similar_lines(lines):
    groups = []

    for line in lines:
        added_to_group = False

        for group in groups:
            if any(lines_are_similar(line, existing_line) for existing_line in group):
                group.append(line)
                added_to_group = True
                break

        if not added_to_group:
            groups.append([line])

    merged_lines = []

    for group in groups:
        merged_line = merge_line_group(group)

        if merged_line is not None:
            merged_lines.append(merged_line)

    return merged_lines

def dedupe_intersections(intersections, min_distance=25):
    deduped = []

    for point in intersections:
        x, y = point

        too_close = False

        for existing_x, existing_y in deduped:
            distance = ((x - existing_x) ** 2 + (y - existing_y) ** 2) ** 0.5

            if distance < min_distance:
                too_close = True
                break

        if not too_close:
            deduped.append(point)

    return deduped

def find_line_intersections(lines, frame_width, frame_height):
    intersections = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            angle_a = get_line_angle(*lines[i])
            angle_b = get_line_angle(*lines[j])

            angle_diff = abs(angle_a - angle_b)
            angle_diff = min(angle_diff, 180 - angle_diff)

            # If the lines are too close in direction, their intersection
            # is usually not a useful field corner.
            if angle_diff < 30:
                continue

            intersection = find_line_intersection(lines[i], lines[j])

            if intersection is None:
                continue

            x, y = intersection

            if 0 <= x < frame_width and 0 <= y < frame_height:
                intersections.append((x, y))

    return dedupe_intersections(intersections)


def filter_strong_lines_by_angle(lines, max_lines_per_group=3):
    angle_groups = []

    for line in lines:
        x1, y1, x2, y2 = line
        angle = get_line_angle(x1, y1, x2, y2)
        length = get_line_length(x1, y1, x2, y2)

        added = False

        for group in angle_groups:
            group_angle = group["angle"]

            angle_diff = abs(angle - group_angle)
            angle_diff = min(angle_diff, 180 - angle_diff)

            if angle_diff < 12:
                group["lines"].append((line, length))
                added = True
                break

        if not added:
            angle_groups.append(
                {
                    "angle": angle,
                    "lines": [(line, length)],
                }
            )

    filtered_lines = []

    for group in angle_groups:
        group["lines"].sort(key=lambda item: item[1], reverse=True)

        for line, length in group["lines"][:max_lines_per_group]:
            filtered_lines.append(line)

    return filtered_lines

def find_line_intersection(line_a, line_b):
    x1, y1, x2, y2 = line_a
    x3, y3, x4, y4 = line_b

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Lines are parallel or basically parallel
    if abs(denominator) < 1e-6:
        return None

    px = (
        ((x1 * y2 - y1 * x2) * (x3 - x4)
         - (x1 - x2) * (x3 * y4 - y3 * x4))
        / denominator
    )

    py = (
        ((x1 * y2 - y1 * x2) * (y3 - y4)
         - (y1 - y2) * (x3 * y4 - y3 * x4))
        / denominator
    )

    return int(px), int(py)

def detect_field_lines(frame):
    grass_mask = create_grass_mask(frame)
    white_line_mask = create_white_line_mask(frame, grass_mask)

    edges = cv2.Canny(white_line_mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    detected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if (
                line_has_grass_nearby(grass_mask, x1, y1, x2, y2)
                and not line_is_too_close_to_grass_boundary(grass_mask, x1, y1, x2, y2)
            ):
                detected_lines.append((x1, y1, x2, y2))

    merged_lines = merge_similar_lines(detected_lines)
    strong_lines = filter_strong_lines_by_angle(merged_lines)

    frame_height, frame_width = frame.shape[:2]
    intersections = find_line_intersections(
        strong_lines,
        frame_width,
        frame_height,
    )

    return grass_mask, white_line_mask, strong_lines, intersections

def line_is_too_close_to_grass_boundary(grass_mask, x1, y1, x2, y2):
    frame_height, frame_width = grass_mask.shape[:2]

    sample_count = 8
    near_boundary_hits = 0

    for i in range(sample_count):
        t = i / max(sample_count - 1, 1)

        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)

        x = max(0, min(frame_width - 1, x))
        y = max(0, min(frame_height - 1, y))

        # Look above the line. If there is mostly non-grass above,
        # this is probably the top boundary of the pitch/ad board area.
        look_above = 25
        y_above = max(0, y - look_above)

        patch_above = grass_mask[y_above:y, max(0, x - 8):min(frame_width, x + 8)]

        if patch_above.size == 0:
            continue

        grass_ratio_above = patch_above.mean() / 255

        if grass_ratio_above < 0.15:
            near_boundary_hits += 1

    return near_boundary_hits >= 5


def draw_field_line_debug(frame, grass_mask, white_line_mask, detected_lines, intersections):
    debug_frame = frame.copy()

    # Draw detected line segments
    for x1, y1, x2, y2 in detected_lines:
        cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for x, y in intersections:
        cv2.circle(debug_frame, (x, y), 7, (0, 0, 255), -1)
        cv2.circle(debug_frame, (x, y), 10, (255, 255, 255), 2)

    cv2.putText(
        debug_frame,
        f"Lines: {len(detected_lines)}  Intersections: {len(intersections)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    # Make masks easier to view
    grass_preview = cv2.cvtColor(grass_mask, cv2.COLOR_GRAY2BGR)
    white_preview = cv2.cvtColor(white_line_mask, cv2.COLOR_GRAY2BGR)

    return debug_frame, grass_preview, white_preview
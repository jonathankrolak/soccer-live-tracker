from src.config import SMOOTHING, MATCH_DISTANCE


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

            # Keep newest detected color
            color = current[2]

            smoothed_positions.append((smooth_x, smooth_y, color))
        else:
            smoothed_positions.append(current)

    return smoothed_positions
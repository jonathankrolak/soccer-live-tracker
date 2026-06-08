import cv2


def get_monitor_position(sct, monitor_index):
    monitor = sct.monitors[monitor_index]

    return monitor["left"], monitor["top"], monitor["width"], monitor["height"]


def move_window_to_monitor(sct, window_name, monitor_index, offset_x=40, offset_y=40):
    left, top, width, height = get_monitor_position(sct, monitor_index)

    cv2.moveWindow(
        window_name,
        left + offset_x,
        top + offset_y,
    )
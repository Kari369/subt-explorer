"""Render an occupancy grid as an image (numpy BGR, for OpenCV windows and PNG files).

Unknown space is dark, free space is pale, walls are near-black with a warm edge.
The robot's trail and current pose are drawn on top, plus a small status bar.
"""

import math

import cv2
import numpy as np

from core.mapping import FREE, OCCUPIED

COL_UNKNOWN = (58, 50, 44)      # BGR
COL_FREE = (196, 214, 226)
COL_OCC = (30, 30, 34)
COL_TRAIL = (230, 190, 40)
COL_ROBOT = (0, 140, 255)
COL_TEXT = (235, 235, 235)


def render(grid, pose=None, trail=(), target_px=760, status_lines=(), markers=(), path=(), goal=None,
           title="OCCUPANCY MAP (robot's own lidar)", info=None):
    """Return a BGR image of the explored part of the map.

    markers: iterable of (x, y, bgr_colour, label) drawn as filled circles.
    """
    r0, r1, c0, c1 = grid.known_bounds(margin_cells=15)
    cls = classify_crop(grid, r0, r1, c0, c1)

    img = np.empty(cls.shape + (3,), dtype=np.uint8)
    img[:] = COL_UNKNOWN
    img[cls == FREE] = COL_FREE
    img[cls == OCCUPIED] = COL_OCC
    img = img[::-1]  # grid row 0 is the south edge; images draw row 0 at the top

    h, w = img.shape[:2]
    scale = target_px / max(h, w)
    img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_NEAREST)

    def to_px(x, y):
        col = (x - grid.origin[0]) / grid.res - c0
        row = (y - grid.origin[1]) / grid.res - r0
        return int(col * scale), int((h - row) * scale)

    if len(trail) > 1:
        pts = np.array([to_px(x, y) for x, y in trail], dtype=np.int32)
        cv2.polylines(img, [pts], False, COL_TRAIL, 2, cv2.LINE_AA)

    if len(path) > 1:   # planned route: magenta
        pts = np.array([to_px(x, y) for x, y in path], dtype=np.int32)
        cv2.polylines(img, [pts], False, (200, 60, 220), 2, cv2.LINE_AA)
    if goal is not None:
        g = to_px(*goal)
        cv2.drawMarker(img, g, (200, 60, 220), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    rad = max(4, int(3 * scale))
    for x, y, colour, label in markers:
        p = to_px(x, y)
        cv2.circle(img, p, rad, colour, -1, cv2.LINE_AA)
        cv2.circle(img, p, rad, (0, 0, 0), 1, cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (p[0] + rad + 3, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COL_TEXT, 1, cv2.LINE_AA)

    if pose is not None:
        x, y, yaw = pose
        L = 0.6  # metres, arrow length
        tip = to_px(x + L * math.cos(yaw), y + L * math.sin(yaw))
        left = to_px(x + 0.35 * math.cos(yaw + 2.5), y + 0.35 * math.sin(yaw + 2.5))
        right = to_px(x + 0.35 * math.cos(yaw - 2.5), y + 0.35 * math.sin(yaw - 2.5))
        cv2.fillPoly(img, [np.array([tip, left, right], dtype=np.int32)], COL_ROBOT, cv2.LINE_AA)

    bar_h = 22 + 18 * len(status_lines)
    out = np.full((img.shape[0] + bar_h, img.shape[1], 3), 20, dtype=np.uint8)
    out[bar_h:] = img
    if info is not None:   # lets a caller turn a click on the image back into world coordinates
        info.update({"scale": scale, "r0": r0, "c0": c0, "h": h, "bar_h": bar_h})
    cv2.putText(out, title or "LAST MAP RECEIVED FROM ROBOT", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 165, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(status_lines):
        cv2.putText(out, line, (10, 34 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)
    return out


def classify_crop(grid, r0, r1, c0, c1):
    from core.mapping import P_FREE, P_OCC, UNKNOWN
    lo = grid.logodds[r0:r1, c0:c1]
    p = 1.0 / (1.0 + np.exp(-lo))
    out = np.full(lo.shape, UNKNOWN, dtype=np.uint8)
    out[p < P_FREE] = FREE
    out[p > P_OCC] = OCCUPIED
    return out

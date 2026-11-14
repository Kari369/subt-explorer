"""Turn a multi-layer lidar scan into one 2D obstacle scan ("terrain analysis", simplified).

A single horizontal lidar plane at 0.78 m misses anything shorter than the sensor:
crates, equipment cases, a person lying on the ground. So the lidar has several
layers; the downward ones hit low obstacles. For every horizontal direction we take
each layer's hit, work out how high above the floor it is, and keep the nearest hit
that is tall enough to be an obstacle. Floor hits are ignored.

Pure numpy, no simulator imports.
"""

import numpy as np

MIN_OBSTACLE_HEIGHT = 0.12   # m above the floor; rails and glow sticks are lower than this
MAX_OBSTACLE_HEIGHT = 2.0    # ignore ceiling fittings


def layer_angles(n_layers, vertical_fov, top_first=True):
    """Elevation angle of each layer (rad, + = up)."""
    if n_layers == 1:
        return np.zeros(1)
    a = np.linspace(vertical_fov / 2, -vertical_fov / 2, n_layers)
    return a if top_first else a[::-1]


def flatten(ranges, elevations, sensor_height, max_range):
    """ranges: (layers, beams) array. Returns (beams,) horizontal distances; inf = nothing.

    Only layers at or below horizontal are used (upward layers just see the roof).
    """
    ranges = np.asarray(ranges, dtype=float)
    out = np.full(ranges.shape[1], np.inf)
    for layer, elev in zip(ranges, elevations):
        if elev > 1e-6:
            continue
        valid = np.isfinite(layer) & (layer < max_range)
        horiz = layer * np.cos(elev)
        height = sensor_height + layer * np.sin(elev)          # elev <= 0, so this goes down
        obstacle = valid & (height > MIN_OBSTACLE_HEIGHT) & (height < MAX_OBSTACLE_HEIGHT)
        out = np.where(obstacle & (horiz < out), horiz, out)
    return out

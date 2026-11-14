"""Offline test for core/mapping.py, no Webots needed.

A fake lidar ray-casts against the true mine layout from a series of poses along
the west and north tunnels. The resulting map must:
  * mark the tunnel the robot can see as free,
  * mark the walls as occupied,
  * never mark solid rock (more than a cell inside a wall) as free.

Run:  python tests/test_mapping.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.mapping import FREE, OCCUPIED, OccupancyGrid  # noqa: E402
from core.truth import load_ground_truth, true_open_mask  # noqa: E402

MAX_RANGE = 15.0


def fake_scan(open_mask, grid, pose, n_beams=360):
    """Ray-cast through the true layout; returns (angles, ranges) like the Webots adapter."""
    x, y, yaw = pose
    angles = math.pi - np.arange(n_beams) * (2 * math.pi / n_beams)
    ranges = np.full(n_beams, np.inf)
    d = np.arange(0.0, MAX_RANGE, grid.res * 0.25)
    for i, a in enumerate(angles):
        px = x + np.cos(yaw + a) * d
        py = y + np.sin(yaw + a) * d
        c, r = grid.world_to_cell(px, py)
        blocked = ~open_mask[r, c]
        if blocked.any():
            ranges[i] = d[np.argmax(blocked)]
    return angles, ranges


def run():
    gt = load_ground_truth()
    start = (gt["start"]["x"], gt["start"]["y"])
    grid = OccupancyGrid(start, size_m=100.0, resolution=0.1)
    open_mask = true_open_mask(grid, gt)

    # drive south down the west tunnel, then back and east along the north tunnel
    poses = [(4.5, -4.5 - k, -math.pi / 2) for k in range(0, 12)]
    poses += [(4.5 + k, -4.5, 0.0) for k in range(0, 18)]
    for pose in poses:
        angles, ranges = fake_scan(open_mask, grid, pose)
        grid.update(pose, angles, ranges, MAX_RANGE)

    cls = grid.classified()

    # 1) visible tunnel is free: the start cell and the corridor it opens onto
    c, r = grid.world_to_cell(np.array([4.5, 4.5, 12.0]), np.array([-4.5, -12.0, -4.5]))
    assert (cls[r, c] == FREE).all(), "corridor cells should be free"

    # 2) the north wall of the start tunnel (y = -3.0) is occupied just inside the rock
    wc, wr = grid.world_to_cell(np.array([6.0, 9.0, 15.0]), np.array([-2.95, -2.95, -2.95]))
    assert (cls[wr, wc] == OCCUPIED).all(), "north wall should be occupied"

    # 3) no free cells deep inside rock (eroded by 2 cells to allow for wall-edge rounding)
    rock_core = ~open_mask
    for _ in range(2):
        rock_core = rock_core & np.roll(rock_core, 1, 0) & np.roll(rock_core, -1, 0) \
                              & np.roll(rock_core, 1, 1) & np.roll(rock_core, -1, 1)
    leaked = np.count_nonzero((cls == FREE) & rock_core)
    assert leaked == 0, f"{leaked} cells inside solid rock were marked free"

    free_seen = np.count_nonzero((cls == FREE) & open_mask) * grid.res ** 2
    total = open_mask.sum() * grid.res ** 2
    print(f"OK: {len(poses)} scans, {free_seen:.0f} of {total:.0f} m2 of tunnel mapped "
          f"({100 * free_seen / total:.0f}%), no free space leaked into rock")


if __name__ == "__main__":
    run()

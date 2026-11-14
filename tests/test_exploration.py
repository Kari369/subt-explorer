"""Offline test of the whole autonomy loop, no Webots needed.

A simple simulated robot (a point that drives and turns, with a fake 2D lidar that
ray-casts against the true layout) runs map -> plan -> follow until the explorer
says it is done. The test passes if the robot
  * maps at least 95% of the tunnel area,
  * never drives into rock,
  * ends back at the start.

Run:  python tests/test_exploration.py          (add --png to save the final map)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.exploration import DONE, Explorer  # noqa: E402
from core.follower import follow  # noqa: E402
from core.mapping import FREE, OccupancyGrid  # noqa: E402
from core.truth import load_ground_truth, true_open_mask  # noqa: E402

MAX_RANGE = 15.0
N_BEAMS = 360
ANGLES = math.pi - np.arange(N_BEAMS) * (2 * math.pi / N_BEAMS)
DT = 0.1
PLAN_EVERY = 10          # steps -> replan once per simulated second
MAP_EVERY = 2


def fast_scan(open_mask, grid, pose):
    x, y, yaw = pose
    d = np.arange(0.0, MAX_RANGE, grid.res * 0.5)
    px = x + np.cos(yaw + ANGLES)[:, None] * d[None, :]
    py = y + np.sin(yaw + ANGLES)[:, None] * d[None, :]
    c, r = grid.world_to_cell(px, py)
    blocked = ~open_mask[np.clip(r, 0, grid.n - 1), np.clip(c, 0, grid.n - 1)]
    first = np.argmax(blocked, axis=1)
    return np.where(blocked.any(axis=1), d[first], np.inf)


def run(save_png=False, with_cart=False):
    gt = load_ground_truth()
    start = (gt["start"]["x"], gt["start"]["y"])
    grid = OccupancyGrid(start, size_m=100.0, resolution=0.1)
    open_mask = true_open_mask(grid, gt)
    if with_cart:
        # A parked mine cart in the top tunnel and a collapse behind it: the pocket between
        # them can't be driven into, only looked into past the cart.
        xs = grid.origin[0] + (np.arange(grid.n) + 0.5) * grid.res
        ys = grid.origin[1] + (np.arange(grid.n) + 0.5) * grid.res
        X, Y = np.meshgrid(xs, ys)
        open_mask &= ~((np.abs(X - 19.1) < 0.6) & (np.abs(Y + 4.5) < 0.43))      # the cart
        open_mask &= ~((X > 24) & (X < 27) & (Y < -6) & (Y > -9))                # the collapse
    explorer = Explorer(start)
    pose = [start[0], start[1], 0.0]
    trail, t0 = [], time.perf_counter()

    step = 0
    while explorer.state != DONE and step < 60000:
        if step % MAP_EVERY == 0:
            grid.update(pose, ANGLES, fast_scan(open_mask, grid, pose), MAX_RANGE)
        if step % PLAN_EVERY == 0:
            explorer.plan(grid, pose)
        vx, wz, _ = follow(pose, explorer.path)
        pose[2] += wz * DT
        pose[0] += vx * math.cos(pose[2]) * DT
        pose[1] += vx * math.sin(pose[2]) * DT
        c, r = grid.world_to_cell(pose[0], pose[1])
        assert open_mask[r, c], f"robot drove into rock at {pose} (step {step})"
        if step % 10 == 0:
            trail.append((pose[0], pose[1]))
        step += 1

    cls = grid.classified()
    covered = np.count_nonzero((cls == FREE) & open_mask) / open_mask.sum()
    home_err = math.hypot(pose[0] - start[0], pose[1] - start[1])
    dist = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(trail, trail[1:]))
    print(f"state={explorer.state}  sim time={step * DT / 60:.1f} min  distance={dist:.0f} m  "
          f"coverage={100 * covered:.1f}%  home error={home_err:.2f} m  "
          f"(wall clock {time.perf_counter() - t0:.0f} s)")
    if save_png:
        import cv2
        from core.map_view import render
        out = Path(__file__).resolve().parents[1] / "docs" / "offline_exploration.png"
        cv2.imwrite(str(out), render(grid, pose, trail, status_lines=["offline autonomy test"]))
        print(f"saved {out}")
    assert explorer.state == DONE, "exploration did not finish"
    assert covered > 0.95, f"only {100 * covered:.1f}% of the tunnels were mapped"
    assert home_err < 1.0, "robot did not return to the start"
    print("OK")


if __name__ == "__main__":
    run(save_png="--png" in sys.argv)
    run(with_cart=True)

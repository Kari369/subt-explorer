"""Offline test for the base station and operator console.

Run:  python tests/test_console.py        (add --png to save docs/console_example.png)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.base_station import HOME, WAYPOINT, BaseStation  # noqa: E402
from core.console_view import MAP_PX, render_console  # noqa: E402
from core.mapping import OccupancyGrid  # noqa: E402
from core.truth import load_ground_truth, true_open_mask  # noqa: E402
from tests.test_exploration import ANGLES, MAX_RANGE, fast_scan  # noqa: E402

COLOURS = {"survivor": (0, 122, 255), "gas_hazard": (76, 201, 242)}


def run(save_png=False):
    gt = load_ground_truth()
    grid = OccupancyGrid((4.5, -4.5))
    mask = true_open_mask(grid, gt)
    trail = []
    for k in range(14):
        pose = (4.5 + k, -4.5, 0.0)
        grid.update(pose, ANGLES, fast_scan(mask, grid, pose), MAX_RANGE)
        trail.append(pose[:2])

    base = BaseStation()
    base.downlink(20.0, grid, pose, trail, {"mode": "AUTO: EXPLORING", "battery": 97, "mapped": 150})

    # the base station keeps its own copy: later changes on the robot must not leak across a dead link
    before = base.grid.logodds.copy()
    grid.update((4.5, -10.0, 0.0), ANGLES, fast_scan(mask, grid, (4.5, -10.0, 0.0)), MAX_RANGE)
    assert np.array_equal(base.grid.logodds, before), "base station map changed without a downlink"
    assert base.age(65.0) == 45.0

    # commands wait while the link is down and go through when it returns
    base.command(30.0, WAYPOINT, x=10.0, y=-4.5)
    base.command(31.0, HOME)
    assert base.uplink(40.0, link_up=False) == [] and len(base.pending) == 2
    sent = base.uplink(50.0, link_up=True)
    assert [c["kind"] for c in sent] == [WAYPOINT, HOME] and not base.pending and sent[0]["sent_s"] == 50.0

    inbox = [
        {"kind": "ARTIFACT", "text": "survivor at (12.0, -4.0)", "created_s": 8.0, "delivered_s": 8.0,
         "payload": {"id": 1, "class": "survivor", "x": 12.0, "y": -4.0}},
        {"kind": "ALERT", "text": "LINK_LOST: lost contact at (22.4, -14.5); continuing autonomously",
         "created_s": 20.0, "delivered_s": 64.0, "payload": {"code": "LINK_LOST"}},
        {"kind": "ARTIFACT", "text": "gas_hazard at (15.0, -5.2)", "created_s": 30.0, "delivered_s": 64.0,
         "payload": {"id": 2, "class": "gas_hazard", "x": 15.0, "y": -5.2}},
    ]
    base.command(60.0, WAYPOINT, x=9.0, y=-4.5)
    img, to_world = render_console(base, inbox, 65.0, False, COLOURS, waypoint=(9.0, -4.5))
    assert img.shape[1] > MAP_PX and to_world is not None

    # a click on the map converts back to the right world position (check with a brute-force search)
    best, target = None, (9.0, -4.5)
    for px in range(0, MAP_PX, 3):
        for py in range(0, img.shape[0], 3):
            w = to_world(px, py)
            if w and (best is None or math.dist(w, target) < math.dist(best[1], target)):
                best = ((px, py), w)
    assert math.dist(best[1], target) < 0.15, f"click mapping is off: nearest {best}"
    assert tuple(img[best[0][1], best[0][0]]) != (24, 24, 24), "that pixel should be on the drawn map"
    assert to_world(MAP_PX + 50, 100) is None, "clicks on the side panel are not waypoints"

    if save_png:
        import cv2
        out = Path(__file__).resolve().parents[1] / "docs" / "console_example.png"
        cv2.imwrite(str(out), img)
        print(f"saved {out}")
    print("OK: base station copy is isolated, commands queue across a dead link, clicks map to world coordinates")


if __name__ == "__main__":
    run(save_png="--png" in sys.argv)

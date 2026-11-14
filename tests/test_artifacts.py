"""Offline test for core/artifacts.py.

Run:  python tests/test_artifacts.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.artifacts import CONFIRM_SIGHTINGS, ArtifactTracker, score  # noqa: E402


def rel(pose, target, cam=(0.34, 0.0)):
    """Where `target` appears in the camera frame for a robot at `pose` (perfect sensor)."""
    x, y, yaw = pose
    cx, cy = x + math.cos(yaw) * cam[0], y + math.sin(yaw) * cam[0]
    dx, dy = target[0] - cx, target[1] - cy
    return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy, 0.0)


def run():
    truth = [{"name": "survivor_1", "class": "survivor", "x": 10.0, "y": 0.0},
             {"name": "survivor_2", "class": "survivor", "x": 30.0, "y": 0.0},
             {"name": "gas_hazard_1", "class": "gas_hazard", "x": 10.5, "y": 1.0}]
    tracker = ArtifactTracker()
    new = []
    # drive along y = -1 past the first survivor and the barrel, seeing them from many angles
    for k in range(40):
        pose = (2.0 + 0.15 * k, -1.0, 0.1)
        dets = [{"class": a["class"], "rel": rel(pose, (a["x"], a["y"]))} for a in truth[:1] + truth[2:]]
        new += tracker.observe(pose, dets, t=k * 0.2)
    # one glimpse of something far away must not be reported
    tracker.observe((20.0, 0.0, 0.0), [{"class": "survivor", "rel": rel((20.0, 0.0, 0.0), (29.0, 0.0))}], t=10.0)

    reports = tracker.reports()
    assert len(new) == 2 and len(reports) == 2, f"expected 2 confirmed reports, got {len(reports)}"
    assert len(tracker.tracks) == 3, "the single far glimpse should be an unconfirmed third track"
    assert all(r["sightings"] >= CONFIRM_SIGHTINGS for r in reports)

    s = score(reports, truth)
    assert s["points"] == 2 and s["missed"] == ["survivor_2"] and not s["false_reports"], s
    assert s["max_error_m"] < 0.6, f"averaged position should be accurate, got {s['max_error_m']} m"

    # class matters and each true artifact scores once
    dup = reports + [dict(reports[0], id=99)]
    s2 = score(dup, truth)
    assert s2["points"] == 2 and s2["false_reports"] == [99], s2
    wrong = [dict(reports[0], **{"class": "equipment_case"})]
    assert score(wrong, truth)["points"] == 0

    print(f"OK: 2 of 3 artifacts reported, mean error {s['mean_error_m']} m, "
          f"duplicate and wrong-class reports score nothing")


if __name__ == "__main__":
    run()

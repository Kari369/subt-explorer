"""Per-run logging: trajectory CSV, map snapshot (npz + png) and a summary.

Each run gets its own folder under logs/, e.g. logs/run_2026-11-09_14-03-22/.
"""

import csv
import json
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


class RunLog:
    def __init__(self, mode="teleop"):
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = ROOT / "logs" / f"run_{stamp}_{mode}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._traj = open(self.dir / "trajectory.csv", "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._traj)
        self._w.writerow(["t", "x", "y", "yaw", "mapped_free_m2"])

    def pose(self, t, x, y, yaw, free_m2):
        self._w.writerow([f"{t:.2f}", f"{x:.3f}", f"{y:.3f}", f"{yaw:.4f}", f"{free_m2:.1f}"])

    def snapshot(self, grid, image, t, extra=None):
        grid.save(self.dir / "map.npz")
        cv2.imwrite(str(self.dir / "map.png"), image)
        summary = {"mode": self.mode, "sim_time_s": round(t, 1),
                   "mapped_free_m2": round(grid.free_area_m2(), 1)}
        summary.update(extra or {})
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._traj.flush()

    def artifacts(self, reports):
        (self.dir / "artifacts.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    def messages(self, messages):
        (self.dir / "messages.json").write_text(json.dumps(messages, indent=2), encoding="utf-8")

    def close(self):
        self._traj.close()

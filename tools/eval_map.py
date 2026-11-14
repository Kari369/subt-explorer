"""Score a saved run's map against the true mine layout.

Coverage   = tunnel area the robot mapped as free / total tunnel area.
             (Boulders, timber and props sit inside the tunnels, so 100% is not reachable.)
Leakage    = area mapped as free that is actually solid rock (more than 0.2 m inside a wall);
             should be ~0. Large values mean the pose or lidar handling is wrong.
Also plots coverage over time from trajectory.csv (the SubT-style exploration curve).

Usage (from the project folder):
  python tools/eval_map.py                 # newest run in logs/
  python tools/eval_map.py logs/run_...    # a specific run
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.artifacts import score  # noqa: E402
from core.mapping import FREE, OCCUPIED, OccupancyGrid  # noqa: E402
from core.truth import load_ground_truth, true_open_mask  # noqa: E402


def erode(mask, n):
    for _ in range(n):
        mask = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    return mask


def main():
    runs = sorted((ROOT / "logs").glob("run_*"))
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else (runs[-1] if runs else None)
    if run is None or not (run / "map.npz").exists():
        raise SystemExit("No saved run with a map.npz found in logs/")

    gt = load_ground_truth()
    grid = OccupancyGrid.load(run / "map.npz")
    cls = grid.classified()
    open_mask = true_open_mask(grid, gt)
    cell_area = grid.res ** 2

    tunnel = open_mask.sum() * cell_area
    mapped = np.count_nonzero((cls == FREE) & open_mask) * cell_area
    leaked = np.count_nonzero((cls == FREE) & erode(~open_mask, 2)) * cell_area
    walls = np.count_nonzero(cls == OCCUPIED) * cell_area

    result = {
        "run": run.name,
        "tunnel_area_m2": round(tunnel, 1),
        "mapped_free_m2": round(mapped, 1),
        "coverage_pct": round(100 * mapped / tunnel, 1),
        "free_leaked_into_rock_m2": round(leaked, 2),
        "occupied_m2": round(walls, 1),
    }
    art_file = run / "artifacts.json"
    art = None
    if art_file.exists():
        reports = json.loads(art_file.read_text(encoding="utf-8"))
        art = score(reports, gt["artifacts"], gt.get("scoring_radius_m", 5.0))
        result.update({"artifact_points": f"{art['points']} / {art['possible']}",
                       "artifact_mean_error_m": art["mean_error_m"],
                       "artifact_max_error_m": art["max_error_m"],
                       "artifacts_missed": ", ".join(art["missed"]) or "none",
                       "false_reports": len(art["false_reports"])})
    summary_file = run / "summary.json"
    if summary_file.exists():
        comms = json.loads(summary_file.read_text(encoding="utf-8")).get("comms")
        if comms:
            result.update({"comms_policy": comms["policy"],
                           "reports_delivered": f"{comms['artifact_reports_delivered']} / {comms['artifact_reports_created']}",
                           "reports_delayed_over_5s": comms["reports_delayed_over_5s"],
                           "mean_report_delay_s": comms["mean_report_delay_s"],
                           "max_report_delay_s": comms["max_report_delay_s"],
                           "returns_to_comms": comms["returns_to_comms"],
                           "alerts": ", ".join(comms["alerts"]) or "none"})
    (run / "eval.json").write_text(json.dumps({**result, "artifact_detail": art}, indent=2), encoding="utf-8")
    for k, v in result.items():
        print(f"{k:>26}: {v}")
    if art:
        for m in art["matches"]:
            print(f"{'':>26}  {m['artifact']:<20} error {m['error_m']:.2f} m")

    traj = run / "trajectory.csv"
    if traj.exists():
        with open(traj, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            t = [float(r["t"]) / 60 for r in rows]
            a = [float(r["mapped_free_m2"]) for r in rows]
            fig, ax = plt.subplots(figsize=(7, 3.5), dpi=130)
            ax.plot(t, a, color="#e67e22", lw=2)
            ax.axhline(tunnel, color="#555", ls="--", lw=1)
            ax.text(t[0], tunnel * 0.97, f"total tunnel area {tunnel:.0f} m²", va="top", fontsize=8, color="#555")
            ax.set_xlabel("simulation time (min)")
            ax.set_ylabel("mapped free space (m²)")
            ax.set_title(f"Exploration over time: {run.name}", fontsize=10)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(run / "coverage.png")
            print(f"{'plot':>26}: {run / 'coverage.png'}")


if __name__ == "__main__":
    main()

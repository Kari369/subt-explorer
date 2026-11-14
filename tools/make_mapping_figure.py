"""Figure for the article's Mapping section: the true mine layout next to the map the
robot built from its own lidar during an autonomous Webots run.

Usage (from the project folder):  python tools/make_mapping_figure.py [logs/run_...]
Writes docs/figures/fig_mapping.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.mapping import FREE, OCCUPIED, OccupancyGrid  # noqa: E402
from core.truth import load_ground_truth, true_open_mask  # noqa: E402

UNKNOWN_C, FREE_C, OCC_C, ROCK_C = (0.17, 0.20, 0.23), (0.89, 0.84, 0.77), (0.10, 0.10, 0.12), (0.30, 0.27, 0.24)


def main():
    runs = sorted((ROOT / "logs").glob("run_*_auto_report"))
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else runs[-1]
    gt = load_ground_truth()
    grid = OccupancyGrid.load(run / "map.npz")
    cls = grid.classified()
    truth = true_open_mask(grid, gt)

    # crop both to the mine, with a small margin
    rows, cols = np.nonzero(truth)
    r0, r1, c0, c1 = rows.min() - 15, rows.max() + 16, cols.min() - 15, cols.max() + 16
    cls, truth = cls[r0:r1, c0:c1][::-1], truth[r0:r1, c0:c1][::-1]

    left = np.empty(truth.shape + (3,))
    left[:] = ROCK_C
    left[truth] = FREE_C
    right = np.empty(cls.shape + (3,))
    right[:] = UNKNOWN_C
    right[cls == FREE] = FREE_C
    right[cls == OCCUPIED] = OCC_C

    covered = 100 * np.count_nonzero((cls == FREE) & truth) / truth.sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    for ax, img, title in ((axes[0], left, "The real mine (never shown to the robot)"),
                           (axes[1], right, f"The robot's own map, built from lidar ({covered:.0f}% of the tunnels)")):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(handles=[Patch(color=FREE_C, label="tunnel"), Patch(color=ROCK_C, label="rock")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=10)
    axes[1].legend(handles=[Patch(color=FREE_C, label="free"), Patch(color=OCC_C, label="wall or obstacle"),
                            Patch(color=UNKNOWN_C, label="unknown")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    # 5 m scale bar on the right panel
    px = 5.0 / grid.res
    h = right.shape[0]
    axes[1].plot([20, 20 + px], [h - 12, h - 12], color="white", lw=3)
    axes[1].text(20 + px / 2, h - 18, "5 m", color="white", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "fig_mapping.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out.relative_to(ROOT)} from {run.name}")


if __name__ == "__main__":
    main()

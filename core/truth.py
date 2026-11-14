"""Ground truth helpers used ONLY for testing and scoring, never by the robot.

Rasterises the layout from config/ground_truth.json onto the same grid the robot
maps into, so maps can be compared cell by cell.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_ground_truth(path=None):
    path = Path(path) if path else ROOT / "config" / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


def true_open_mask(grid, gt):
    """Boolean array, same shape as grid.logodds: True where the layout is tunnel.

    Boulders, timber and props are not in the layout, so this is the tunnel
    envelope, an upper bound on what can ever be mapped as free.
    """
    cell, layout = gt["cell_size"], gt["layout"]
    n = grid.n
    cols = np.arange(n)
    xs = grid.origin[0] + (cols + 0.5) * grid.res          # world x of each column
    ys = grid.origin[1] + (cols + 0.5) * grid.res          # world y of each row
    lc = np.floor(xs / cell).astype(int)                   # layout column of each grid column
    lr = np.floor(-ys / cell).astype(int)                  # layout row of each grid row (y is negative going south)
    rows_ok = (lr >= 0) & (lr < len(layout))
    cols_ok = (lc >= 0) & (lc < len(layout[0]))
    table = np.array([[ch != "#" for ch in row] for row in layout], dtype=bool)
    mask = np.zeros((n, n), dtype=bool)
    r_idx = np.where(rows_ok)[0]
    c_idx = np.where(cols_ok)[0]
    mask[np.ix_(r_idx, c_idx)] = table[np.ix_(lr[r_idx], lc[c_idx])]
    return mask

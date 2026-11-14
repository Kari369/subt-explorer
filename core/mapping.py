"""Occupancy grid mapping from a 2D lidar.

The world is divided into square cells (default 10 cm). Each cell stores the
log-odds that it is occupied:  l = log(p / (1 - p)).  0 means "unknown" (p = 0.5).

For every lidar beam:
  * cells the beam passed through get evidence of being FREE   (l += L_FREE)
  * the cell where the beam stopped gets evidence of being OCCUPIED (l += L_OCC)
  * beams that hit nothing (inf) only mark free space, up to the sensor's max range

Adding log-odds is the Bayes update for independent observations, which is why
repeated passes make the map more confident rather than overwritten. Values are
clamped so a cell can still change its mind if the world changes.

Pure numpy, no simulator imports.
"""

import math

import numpy as np

L_OCC = 0.85
L_FREE = -0.4
L_MIN, L_MAX = -4.0, 4.0
P_OCC = 0.65      # probability above which a cell counts as occupied
P_FREE = 0.35     # probability below which a cell counts as free

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class OccupancyGrid:
    def __init__(self, center_xy, size_m=100.0, resolution=0.1):
        self.res = resolution
        self.n = int(round(size_m / resolution))
        # world coordinate of the grid's lower-left corner
        self.origin = np.array(center_xy, dtype=float) - size_m / 2
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)  # [row = y, col = x]
        self.updates = 0

    # ---- coordinate helpers -------------------------------------------------
    def world_to_cell(self, x, y):
        """World metres -> (col, row) integer indices (numpy arrays or scalars)."""
        col = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int)
        row = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(int)
        return col, row

    def cell_to_world(self, col, row):
        """Cell indices -> world coordinates of the cell centre."""
        return (self.origin[0] + (np.asarray(col) + 0.5) * self.res,
                self.origin[1] + (np.asarray(row) + 0.5) * self.res)

    def in_bounds(self, col, row):
        return (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)

    # ---- the update ----------------------------------------------------------
    def update(self, pose, angles, ranges, max_range):
        """Integrate one scan.

        pose:   (x, y, yaw) of the lidar in the world
        angles: beam angles in the sensor frame (rad, 0 = forward, + = left)
        ranges: measured distances (m); inf / nan = no return
        """
        x, y, yaw = pose
        angles = np.asarray(angles, dtype=float)
        ranges = np.asarray(ranges, dtype=float)

        hit = np.isfinite(ranges) & (ranges < max_range)
        free_len = np.where(hit, ranges, max_range)
        free_len = np.where(np.isfinite(free_len), free_len, max_range)

        world_angles = yaw + angles
        cos_a, sin_a = np.cos(world_angles), np.sin(world_angles)

        # Free space: sample every half cell along each beam, stopping one cell short of the hit.
        step = self.res * 0.5
        d = np.arange(0.0, max_range, step)                       # (S,)
        mask = d[None, :] < (free_len[:, None] - self.res)        # (B, S)
        px = x + cos_a[:, None] * d[None, :]
        py = y + sin_a[:, None] * d[None, :]
        fc, fr = self.world_to_cell(px[mask], py[mask])
        ok = self.in_bounds(fc, fr)
        free_idx = np.unique(fr[ok] * self.n + fc[ok])

        # Occupied: the endpoint of every beam that actually hit something.
        hx = x + cos_a[hit] * ranges[hit]
        hy = y + sin_a[hit] * ranges[hit]
        oc, orow = self.world_to_cell(hx, hy)
        ok = self.in_bounds(oc, orow)
        occ_idx = np.unique(orow[ok] * self.n + oc[ok])

        # A cell that is both crossed and hit in the same scan counts as hit.
        free_idx = np.setdiff1d(free_idx, occ_idx, assume_unique=True)

        flat = self.logodds.reshape(-1)
        flat[free_idx] += L_FREE
        flat[occ_idx] += L_OCC
        np.clip(flat, L_MIN, L_MAX, out=flat)
        self.updates += 1

    # ---- reading the map ----------------------------------------------------
    def probabilities(self):
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def classified(self):
        """UNKNOWN / FREE / OCCUPIED per cell."""
        p = self.probabilities()
        out = np.full(p.shape, UNKNOWN, dtype=np.uint8)
        out[p < P_FREE] = FREE
        out[p > P_OCC] = OCCUPIED
        return out

    def known_bounds(self, margin_cells=20):
        """(row0, row1, col0, col1) bounding box of everything observed so far."""
        rows, cols = np.nonzero(self.logodds != 0)
        if rows.size == 0:
            c = self.n // 2
            return c - margin_cells, c + margin_cells, c - margin_cells, c + margin_cells
        return (max(rows.min() - margin_cells, 0), min(rows.max() + margin_cells, self.n),
                max(cols.min() - margin_cells, 0), min(cols.max() + margin_cells, self.n))

    def free_area_m2(self):
        return float(np.count_nonzero(self.classified() == FREE)) * self.res ** 2

    def save(self, path):
        np.savez_compressed(path, logodds=self.logodds, origin=self.origin, res=self.res)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        grid = cls.__new__(cls)
        grid.logodds = data["logodds"]
        grid.origin = data["origin"]
        grid.res = float(data["res"])
        grid.n = grid.logodds.shape[0]
        grid.updates = 0
        return grid


def lidar_pose_from_gps(gps_xy, yaw, gps_offset):
    """The GPS antenna is not at the robot's centre; shift it back into the body frame.

    gps_offset: (dx, dy) of the antenna in the robot frame (m).
    """
    dx, dy = gps_offset
    c, s = math.cos(yaw), math.sin(yaw)
    return gps_xy[0] - (c * dx - s * dy), gps_xy[1] - (s * dx + c * dy)

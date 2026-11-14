"""Navigation grid + shortest paths.

From the occupancy grid we build a coarser planning grid (0.2 m cells) where:
  * walls are "inflated" by the robot's radius, so a path that is one cell wide
    on the planning grid is actually wide enough for the whole robot;
  * cells close to walls cost more, so paths prefer the middle of a tunnel;
  * unknown space is never driven through.

One Dijkstra search from the robot then gives the cost and the route to EVERY
reachable cell, which the exploration planner uses to compare frontiers.

Pure numpy / OpenCV, no simulator imports.
"""

import heapq
import math

import cv2
import numpy as np

from core.mapping import P_FREE, P_OCC

DOWN = 2                    # planning cell = DOWN x DOWN map cells
INFLATE_M = 0.42            # robot half-width (0.29) + safety margin
PREFERRED_CLEARANCE = 1.0   # m; closer to walls than this costs extra
WALL_PENALTY = 3.0


class NavGrid:
    def __init__(self, grid):
        self.res = grid.res * DOWN
        r0, r1, c0, c1 = grid.known_bounds(margin_cells=6)
        r1 -= (r1 - r0) % DOWN
        c1 -= (c1 - c0) % DOWN
        self.origin = (grid.origin[0] + c0 * grid.res, grid.origin[1] + r0 * grid.res)

        lo = grid.logodds[r0:r1, c0:c1]
        p = 1.0 / (1.0 + np.exp(-lo))
        occ = p > P_OCC
        free = (p < P_FREE).astype(np.uint8)
        # Far from the robot, lidar beams are further apart than a cell and leave thin
        # unknown slivers inside open space. Closing fills them so they aren't mistaken for frontiers.
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & ~occ

        h, w = occ.shape
        blocks = lambda a: a.reshape(h // DOWN, DOWN, w // DOWN, DOWN)  # noqa: E731
        self.occ = blocks(occ).any(axis=(1, 3))
        self.free = blocks(free).all(axis=(1, 3)) & ~self.occ
        self.unknown = ~self.free & ~self.occ
        self.shape = self.occ.shape

        k = int(math.ceil(INFLATE_M / self.res))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        self.inflated = cv2.dilate(self.occ.astype(np.uint8), kernel).astype(bool)
        self.traversable = self.free & ~self.inflated

        clearance = cv2.distanceTransform((~self.occ).astype(np.uint8), cv2.DIST_L2, 3) * self.res
        self.cell_cost = 1.0 + WALL_PENALTY * np.clip(PREFERRED_CLEARANCE - clearance, 0, None)

        self.dist = None
        self.parent = None

    # ---- coordinates ---------------------------------------------------------
    def to_cell(self, x, y):
        return int((y - self.origin[1]) / self.res), int((x - self.origin[0]) / self.res)

    def to_world(self, row, col):
        return self.origin[0] + (col + 0.5) * self.res, self.origin[1] + (row + 0.5) * self.res

    def inside(self, row, col):
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]

    def nearest_traversable(self, x, y, max_m=1.5):
        """The robot may sit inside the inflated zone (e.g. next to a wall); start from the closest OK cell."""
        r, c = self.to_cell(x, y)
        if self.inside(r, c) and self.traversable[r, c]:
            return r, c
        k = int(max_m / self.res)
        r0, r1 = max(r - k, 0), min(r + k + 1, self.shape[0])
        c0, c1 = max(c - k, 0), min(c + k + 1, self.shape[1])
        rows, cols = np.nonzero(self.traversable[r0:r1, c0:c1])
        if rows.size == 0:
            return None
        i = np.argmin((rows + r0 - r) ** 2 + (cols + c0 - c) ** 2)
        return int(rows[i] + r0), int(cols[i] + c0)

    # ---- search --------------------------------------------------------------
    def search_from(self, x, y):
        """Dijkstra over traversable cells from (x, y). Fills self.dist (metres-ish cost) and self.parent."""
        h, w = self.shape
        self.dist = np.full(h * w, np.inf)
        self.parent = np.full(h * w, -1, dtype=np.int64)
        start = self.nearest_traversable(x, y)
        if start is None:
            return False
        trav = self.traversable.reshape(-1).tolist()
        cost = self.cell_cost.reshape(-1).tolist()
        dist = self.dist.tolist()
        parent = self.parent.tolist()
        s = start[0] * w + start[1]
        dist[s] = 0.0
        heap = [(0.0, s)]
        d1, d2 = self.res, self.res * math.sqrt(2)
        moves = ((-w, d1), (w, d1), (-1, d1), (1, d1), (-w - 1, d2), (-w + 1, d2), (w - 1, d2), (w + 1, d2))
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist[i]:
                continue
            col = i % w
            for off, step in moves:
                j = i + off
                if j < 0 or j >= h * w or not trav[j]:
                    continue
                jc = j % w
                if abs(jc - col) > 1:      # wrapped round the edge of the grid
                    continue
                nd = d + step * cost[j]
                if nd < dist[j]:
                    dist[j] = nd
                    parent[j] = i
                    heapq.heappush(heap, (nd, j))
        self.dist = np.array(dist).reshape(h, w)
        self.parent = np.array(parent, dtype=np.int64)
        return True

    def path_to(self, row, col):
        """World-coordinate waypoints from the search origin to (row, col), or None if unreachable."""
        if self.dist is None or not np.isfinite(self.dist[row, col]):
            return None
        w = self.shape[1]
        i, cells = row * w + col, []
        while i != -1:
            cells.append(divmod(i, w))
            i = int(self.parent[i])
        cells.reverse()
        return [self.to_world(r, c) for r, c in cells]

    def path_to_xy(self, x, y):
        cell = self.nearest_traversable(x, y)
        return self.path_to(*cell) if cell else None

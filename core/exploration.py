"""Frontier-based exploration (Yamauchi, 1997), the idea SubT planners such as TARE build on.

A frontier cell is a free cell next to unknown space: the edge of what the robot has
seen. Frontier cells are grouped into clusters; each cluster is a candidate place
to go. The planner picks the cluster with the best trade-off between
  * how far it is to drive there (path cost from one Dijkstra search), and
  * how big the frontier is (a rough measure of how much new space it will reveal),
with a bonus for the cluster it was already heading to, so it doesn't dither.

When no frontier is left, the mine is explored and the robot returns to base.

Pure numpy / OpenCV, no simulator imports.
"""

import math

import cv2
import numpy as np

from core.planner import NavGrid

MIN_CLUSTER_CELLS = 4       # ignore frontiers shorter than ~0.8 m
GAIN_PER_M = 1.0            # each metre of frontier is worth this many metres of driving
MAX_GAIN_M = 6.0
KEEP_GOAL_BONUS = 3.0       # metres of advantage given to the current goal
GOAL_REACHED_M = 0.5
BLACKLIST_RADIUS_M = 1.2
REMOTE_MIN_CELLS = 4        # look-only frontiers must be at least ~0.8 m long
REMOTE_VIEW_M = 2.5         # how close a reachable viewpoint must be to such a frontier
AT_GOAL_PLANS = 3           # plans spent sitting on a goal that doesn't clear -> blacklist it

EXPLORING, RETURNING, DONE = "EXPLORING", "RETURNING", "DONE"


class Explorer:
    def __init__(self, home_xy):
        self.home = tuple(home_xy)
        self.state = EXPLORING
        self.goal = None            # (x, y)
        self.path = []              # [(x, y), ...] from the robot to the goal
        self.frontiers = []         # [(x, y, size_m)] for display
        self.blacklist = []         # goals that turned out to be unreachable in practice
        self.nav = None
        self.detour = None          # (x, y) set by the supervisor to regain radio contact
        self._at_goal = 0

    def give_up_on_goal(self):
        """Called when the robot is stuck: don't pick this spot again."""
        if self.goal is not None:
            self.blacklist.append(self.goal)
        self.goal, self.path = None, []

    def goto(self, grid, pose, target):
        """Plan a route to a specific point (operator waypoint, or a spot with radio signal).

        Returns True if a route exists. Does not change the exploration state.
        """
        nav = self.nav = NavGrid(grid)
        self.frontiers = []
        self.goal = tuple(target)
        self.path = (nav.path_to_xy(*target) or []) if nav.search_from(pose[0], pose[1]) else []
        return bool(self.path)

    def plan(self, grid, pose):
        """Recompute the goal and path from the latest map. Returns the current state."""
        x, y = pose[0], pose[1]
        nav = self.nav = NavGrid(grid)
        if not nav.search_from(x, y):
            self.path = []
            return self.state

        if self.state == EXPLORING and self.detour is not None:
            # The supervisor wants contact with base restored before exploring further.
            self.goal = self.detour
            self.path = nav.path_to_xy(*self.detour) or []
            return self.state

        if self.state == EXPLORING:
            # Sitting on the goal and it still counts as a frontier? Then going there doesn't help.
            if self.goal and math.hypot(x - self.goal[0], y - self.goal[1]) < GOAL_REACHED_M:
                self._at_goal += 1
                if self._at_goal >= AT_GOAL_PLANS:
                    self.give_up_on_goal()
                    self._at_goal = 0
            else:
                self._at_goal = 0
            clusters = self._clusters(nav)
            self.frontiers = [(cx, cy, size) for cx, cy, size, _ in clusters]
            best, best_score = None, math.inf
            for cx, cy, size, (gr, gc) in clusters:
                gx, gy = nav.to_world(gr, gc)
                if any(math.hypot(gx - bx, gy - by) < BLACKLIST_RADIUS_M for bx, by in self.blacklist):
                    continue
                score = nav.dist[gr, gc] - GAIN_PER_M * min(size, MAX_GAIN_M)
                if self.goal and math.hypot(gx - self.goal[0], gy - self.goal[1]) < 1.5:
                    score -= KEEP_GOAL_BONUS
                if score < best_score:
                    best, best_score = (gr, gc), score
            if best is None:
                self.state = RETURNING
            else:
                self.goal = nav.to_world(*best)
                self.path = nav.path_to(*best) or []
                return self.state

        if self.state == RETURNING:
            self.frontiers = []
            self.goal = self.home
            if math.hypot(x - self.home[0], y - self.home[1]) < GOAL_REACHED_M:
                self.state, self.path = DONE, []
            else:
                self.path = nav.path_to_xy(*self.home) or []
        return self.state

    def _clusters(self, nav):
        """[(centroid_x, centroid_y, size_m, cheapest reachable cell)] for each frontier cluster.

        First choice: frontiers the robot can drive right up to. If none are left, fall back
        to frontiers it can only look at from nearby (e.g. open space behind a parked mine
        cart): the goal is then the closest reachable spot, and if standing there doesn't
        clear the frontier, plan() blacklists it.
        """
        near_unknown = cv2.dilate(nav.unknown.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        reachable = nav.traversable & np.isfinite(nav.dist)
        out = self._cluster_list(nav, reachable & near_unknown, MIN_CLUSTER_CELLS, reachable, 0)
        if out:
            return out
        # Cells hugging an obstacle are excluded: gaps between lidar hits along a wall look like
        # tiny frontiers but lead nowhere. A real opening (e.g. beside a parked cart) is wider.
        hugging = cv2.dilate(nav.occ.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        remote = nav.free & near_unknown & ~hugging & ~reachable
        return self._cluster_list(nav, remote, REMOTE_MIN_CELLS, reachable, int(REMOTE_VIEW_M / nav.res))

    @staticmethod
    def _cluster_list(nav, frontier, min_cells, reachable, reach_cells):
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, n):
            cells = stats[i, cv2.CC_STAT_AREA]
            if cells < min_cells:
                continue
            member = labels == i
            if reach_cells:   # goal = reachable cell near the cluster
                k = 2 * reach_cells + 1
                member = cv2.dilate(member.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & reachable
            rows, cols = np.nonzero(member)
            if rows.size == 0:
                continue
            j = np.argmin(nav.dist[rows, cols])
            cx, cy = nav.to_world(centroids[i][1], centroids[i][0])
            out.append((cx, cy, cells * nav.res, (int(rows[j]), int(cols[j]))))
        return out

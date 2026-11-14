"""The base station: everything the human operator can know, and everything they can ask for.

The operator never sees the robot's live state. They see the last copy of it that made
it across the radio link (map, position, mode, battery), which goes stale the moment
the link drops. Commands go the other way and only reach the robot while the link is
up; otherwise they wait in a queue.

Pure Python / numpy, no simulator imports.
"""

import copy

WAYPOINT, HOME, RESUME, AUTO_ON, AUTO_OFF = "WAYPOINT", "HOME", "RESUME", "AUTO_ON", "AUTO_OFF"


class BaseStation:
    def __init__(self):
        self.grid = None          # last map copy received
        self.pose = None          # last robot pose received
        self.trail = []
        self.status = {}          # mode, battery ... as last received
        self.last_update_s = None
        self.pending = []         # commands issued but not yet received by the robot
        self.history = []         # every command, with issue and send times

    # ---- robot -> base -------------------------------------------------------
    def downlink(self, t, grid, pose, trail, status):
        """Called only while the link is up: store a snapshot of the robot's state."""
        snap = copy.copy(grid)
        snap.logodds = grid.logodds.copy()
        self.grid, self.pose, self.trail = snap, tuple(pose), list(trail)
        self.status = dict(status)
        self.last_update_s = t

    def age(self, t):
        """Seconds since the operator's picture was last refreshed."""
        return None if self.last_update_s is None else t - self.last_update_s

    # ---- base -> robot -------------------------------------------------------
    def command(self, t, kind, **payload):
        cmd = {"kind": kind, "payload": payload, "issued_s": round(t, 1), "sent_s": None}
        self.pending.append(cmd)
        self.history.append(cmd)
        return cmd

    def uplink(self, t, link_up):
        """Commands that get through to the robot right now (empty while the link is down)."""
        if not link_up or not self.pending:
            return []
        sent, self.pending = self.pending, []
        for cmd in sent:
            cmd["sent_s"] = round(t, 1)
        return sent

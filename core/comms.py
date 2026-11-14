"""Radio link between the robot and the base station (a simulation of the world, not robot logic).

Underground, radio dies quickly: open tunnel carries a signal a fair distance, rock
absorbs it. The model here is deliberately simple:

    effective distance = straight-line distance + ROCK_PENALTY x (metres of rock on that line)

The link is up while the effective distance is within the radio's budget. A little
hysteresis stops it flickering at the edge. Rock is read from the true mine layout,
because this is physics, not something the robot gets to decide.

SubT teams fought exactly this by dropping relay nodes; here there are none, so the
robot simply loses contact when it goes deep and must cope.
"""

import math

ROCK_PENALTY = 2.0      # each metre of rock costs as much signal as this many extra metres of open tunnel
LINK_BUDGET_M = 40.0    # link is up while effective distance <= this (covers roughly the west half of the mine)
HYSTERESIS_M = 3.0      # once down, it must improve by this much to come back
SAMPLE_M = 0.25


class CommsModel:
    def __init__(self, base_xy, layout, cell_size):
        self.base = tuple(base_xy)
        self.layout = layout
        self.cell = cell_size
        self.up = True

    def _is_rock(self, x, y):
        col, row = int(math.floor(x / self.cell)), int(math.floor(-y / self.cell))
        if 0 <= row < len(self.layout) and 0 <= col < len(self.layout[0]):
            return self.layout[row][col] == "#"
        return True

    def effective_distance(self, x, y):
        dx, dy = x - self.base[0], y - self.base[1]
        d = math.hypot(dx, dy)
        n = max(1, int(d / SAMPLE_M))
        rock = sum(self._is_rock(self.base[0] + dx * i / n, self.base[1] + dy * i / n) for i in range(1, n))
        return d + ROCK_PENALTY * rock * (d / n)

    def update(self, x, y):
        """Recompute the link state for the robot at (x, y). Returns True if the link is up."""
        eff = self.effective_distance(x, y)
        limit = LINK_BUDGET_M if self.up else LINK_BUDGET_M - HYSTERESIS_M
        self.up = eff <= limit
        return self.up

    def signal(self, x, y):
        """0..1 signal quality for display (1 = at the base, 0 = at or beyond the budget)."""
        return max(0.0, 1.0 - self.effective_distance(x, y) / LINK_BUDGET_M)

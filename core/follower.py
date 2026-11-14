"""Pure-pursuit path follower.

Looks a fixed distance ahead along the path and steers toward that point. If the
point is far off to the side or behind, the robot turns on the spot first.
Output is a body velocity command (forward m/s, turn rad/s).
"""

import math

LOOKAHEAD = 0.9      # m
CRUISE = 0.7         # m/s
TURN_GAIN = 1.8
MAX_TURN = 1.1       # rad/s
TURN_IN_PLACE = 0.9  # rad; heading error above this -> rotate without driving
GOAL_TOL = 0.35      # m


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def follow(pose, path):
    """Returns (vx, wz, reached). path is [(x, y), ...] in world coordinates."""
    if not path:
        return 0.0, 0.0, True
    x, y, yaw = pose
    gx, gy = path[-1]
    if math.hypot(gx - x, gy - y) < GOAL_TOL:
        return 0.0, 0.0, True

    # closest point on the path, then walk forward until LOOKAHEAD away
    i0 = min(range(len(path)), key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2)
    target = path[-1]
    for px, py in path[i0:]:
        if math.hypot(px - x, py - y) >= LOOKAHEAD:
            target = (px, py)
            break

    err = _wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
    wz = max(-MAX_TURN, min(MAX_TURN, TURN_GAIN * err))
    if abs(err) > TURN_IN_PLACE:
        return 0.0, wz, False
    dist_goal = math.hypot(gx - x, gy - y)
    vx = CRUISE * math.cos(err) * min(1.0, 0.3 + dist_goal / 1.5)
    return vx, wz, False

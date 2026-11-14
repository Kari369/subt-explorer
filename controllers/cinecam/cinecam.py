"""Cinematographer: a camera-only supervisor robot used to film the demo video.

It has no body (nothing for the explorer's lidar to see) and does not affect the
mission. Modes:
  orbit   circle high above the mine, looking at its centre (shows the layout)
  chase   follow the explorer from above and behind, smoothly

Frames are saved as JPEGs named by simulation time, so they can be matched to the
operator-console frames the explorer saves at the same moments.

Controller arguments (key=value): mode=orbit|chase  out=<folder>  every=<sim s between frames>
                                   end=<sim s to stop>  spin=<orbit seconds per turn>
"""

import math
import sys
from pathlib import Path

from controller import Supervisor

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODE = args.get("mode", "chase")
OUT = Path(args.get("out", "frames_cine"))
EVERY = float(args.get("every", "0.1"))
END = float(args.get("end", "1e9"))
SPIN = float(args.get("spin", "16"))

CENTER = (22.5, -16.5)
ORBIT_R, ORBIT_H = float(args.get("r", "30")), float(args.get("h", "30"))
CHASE_BACK, CHASE_UP = 4.2, 6.5


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def look_rotation(src, dst):
    """Axis-angle rotation that points a Webots camera (looks along +x, z up) from src at dst."""
    dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))        # positive = looking down
    qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    qy = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    w, x, y, z = quat_mul(qz, qy)
    angle = 2 * math.acos(max(-1.0, min(1.0, w)))
    s = math.sqrt(max(1e-12, 1 - w * w))
    return [x / s, y / s, z / s, angle]


def main():
    sup = Supervisor()
    dt = int(sup.getBasicTimeStep())
    cam = sup.getDevice("cine")
    cam.enable(dt)
    me = sup.getSelf()
    tf, rf = me.getField("translation"), me.getField("rotation")
    explorer = sup.getFromDef("EXPLORER")
    OUT.mkdir(parents=True, exist_ok=True)

    cam_pos, cam_yaw = None, None
    next_shot = 0.0
    while sup.step(dt) != -1:
        t = sup.getTime()
        if t > END:
            break
        if MODE == "orbit":
            a = 2 * math.pi * t / SPIN - math.pi / 2
            pos = (CENTER[0] + ORBIT_R * math.cos(a), CENTER[1] + ORBIT_R * math.sin(a), ORBIT_H)
            look = (CENTER[0], CENTER[1], 0.0)
        else:
            p = explorer.getPosition()
            R = explorer.getOrientation()
            yaw = math.atan2(R[3], R[0])
            if cam_yaw is None:
                cam_yaw = yaw
            # smooth the heading so the camera doesn't whip round when the robot turns on the spot
            cam_yaw += 0.04 * math.atan2(math.sin(yaw - cam_yaw), math.cos(yaw - cam_yaw))
            want = (p[0] - CHASE_BACK * math.cos(cam_yaw), p[1] - CHASE_BACK * math.sin(cam_yaw), CHASE_UP)
            cam_pos = want if cam_pos is None else tuple(c + 0.12 * (w - c) for c, w in zip(cam_pos, want))
            pos = cam_pos
            look = (p[0] + 1.2 * math.cos(cam_yaw), p[1] + 1.2 * math.sin(cam_yaw), 0.3)
        tf.setSFVec3f(list(pos))
        rf.setSFRotation(look_rotation(pos, look))
        if t >= next_shot:
            cam.saveImage(str(OUT / f"{int(round(t * 1000)):08d}.jpg"), 90)
            next_shot += EVERY


if __name__ == "__main__":
    main()

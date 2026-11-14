"""Webots controller for the explorer robot (Robotnik Summit-XL Steel).

This file is the *adapter*: it is the only place that talks to Webots. It reads
sensors, sends wheel commands and hands plain Python data (pose, lidar scan,
detections) to the simulator-independent code in ../../core.

Mapping, autonomous exploration, artifact reports, a radio link that drops out, and an
operator console. Everything the operator does travels over the radio link: with the
link down, commands queue and the drive keys do nothing.

Keys (click the 3D view or a map window first). The robot moves only WHILE a drive key is held:
  W / S      forward / backward
  A / D      turn left / right
  Q / E      strafe left / right (mecanum wheels)
  T          autonomy on/off: the robot explores by itself, then returns to base.
             Holding a drive key while autonomous overrides it until you let go.
  click      on the Operator console map: send the robot to that point
  X          cancel the waypoint          H   return to base now
  G          collision guard on/off (on = the robot refuses to drive into walls)
  C          show / hide the Operator console (what the base station knows)
  V          show / hide the Live map (the robot's own view, for debugging and video)
  P          print a full sensor report now
  M          save a map snapshot now (also saved every 15 s and when the run ends)
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from controller import Display, Keyboard, Robot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.artifacts import ArtifactTracker  # noqa: E402
from core.base_station import AUTO_OFF, AUTO_ON, HOME, RESUME, WAYPOINT, BaseStation  # noqa: E402
from core.comms import CommsModel  # noqa: E402
from core.console_view import render_console  # noqa: E402
from core.exploration import DONE, EXPLORING, RETURNING, Explorer  # noqa: E402
from core.supervisor import Supervisor  # noqa: E402
from core.truth import load_ground_truth  # noqa: E402
from core.follower import follow  # noqa: E402
from core.map_view import render  # noqa: E402
from core.mapping import OccupancyGrid, lidar_pose_from_gps  # noqa: E402
from core.run_log import RunLog  # noqa: E402
from core.scan import flatten, layer_angles  # noqa: E402

GPS_OFFSET = (-0.365, -0.167)   # GPS antenna position in the robot frame (SummitXlSteel.proto)
LIDAR_HEIGHT = 0.78             # lidar height above the floor (m)
PLAN_EVERY = 31                 # replan about once a second
STUCK_WINDOW_S = 8.0            # no progress for this long in autonomous mode -> give up on the goal
BACKUP_S = 1.5
CAMERA_OFFSET = (0.34, 0.0)     # camera position in the robot frame (m)
ARTIFACT_COLOURS = {            # BGR, for the map
    "survivor": (0, 122, 255),
    "fire_extinguisher": (53, 57, 229),
    "equipment_case": (237, 128, 47),
    "gas_hazard": (76, 201, 242),
}
MAP_EVERY = 6                   # integrate a scan every 6 steps (~5 Hz at 32 ms)
VIEW_EVERY = 8                  # redraw the map window every 8 steps (~4 Hz)
SNAPSHOT_EVERY_S = 15.0
MAP_WINDOW = "Live map"
CONSOLE_WINDOW = "Operator console"

# Summit-XL Steel geometry (from Cyberbotics' keyboard_control.c)
WHEEL_RADIUS = 0.123
LX, LY = 0.2045, 0.2225

MAX_V = 0.9      # m/s
MAX_W = 1.1      # rad/s
ACCEL_V = 1.8    # m/s^2, how quickly speed ramps up / down
ACCEL_W = 4.0    # rad/s^2

# Collision guard: distances are measured from the lidar at the robot's centre.
# The body is 0.72 m long and 0.58 m wide, so these leave ~0.2 m of clearance.
STOP_FRONT = 0.58
STOP_SIDE = 0.50
SLOW_ZONE = 0.6  # start slowing this far before the stop distance


class ExplorerRobot:
    """Thin wrapper over the Webots devices. Everything it returns is plain Python."""

    def __init__(self):
        self.robot = Robot()
        self.dt_ms = int(self.robot.getBasicTimeStep())

        self.wheels = [self.robot.getDevice(n) for n in (
            "front_left_wheel_joint", "front_right_wheel_joint",
            "back_left_wheel_joint", "back_right_wheel_joint")]
        for w in self.wheels:
            w.setPosition(float("inf"))
            w.setVelocity(0.0)

        self.gps = self.robot.getDevice("gps")
        self.compass = self.robot.getDevice("compass")
        self.lidar = self.robot.getDevice("lidar")
        self.camera = self.robot.getDevice("artifact_camera")
        for dev in (self.gps, self.compass, self.lidar, self.camera):
            dev.enable(self.dt_ms)
        self.camera.recognitionEnable(self.dt_ms)

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.dt_ms)

        self._elev = None
        self.args = [a.lower() for a in sys.argv[1:]]
        self.display = self.robot.getDevice("map_display")  # None if the world has no Display
        n = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()
        # Webots orders a 360-degree lidar from +fov/2 (left-rear) round to -fov/2.
        self.angles = fov / 2 - np.arange(n) * fov / n

    def held_keys(self):
        """Set of keys held down right now: upper-case letters, plus Keyboard.* codes for arrows."""
        held = set()
        key = self.keyboard.getKey()
        while key != -1:
            k = key & 0xFFFF
            held.add(chr(k).upper() if 32 <= k < 127 else k)
            key = self.keyboard.getKey()
        return held

    def ranges(self):
        """One 2D obstacle scan (numpy array aligned with self.angles).

        The lidar has several layers; the downward ones catch obstacles shorter than
        the sensor. core.scan.flatten keeps, per direction, the nearest real obstacle.
        """
        layers = self.lidar.getNumberOfLayers()
        raw = np.array(self.lidar.getRangeImage(), dtype=float).reshape(layers, -1)
        if layers == 1:
            return raw[0]
        if self._elev is None:
            # Work out which end of the layer list points down: that layer hits the floor
            # close by, so its typical range is the shortest.
            med = [np.median(l[np.isfinite(l)]) if np.isfinite(l).any() else np.inf for l in (raw[0], raw[-1])]
            self._elev = layer_angles(layers, self.lidar.getVerticalFov(), top_first=med[0] > med[1])
        return flatten(raw, self._elev, LIDAR_HEIGHT, self.lidar.getMaxRange())

    def show_map(self, image_bgr):
        """Paint an image onto the robot's Display (shown as an overlay in the Webots 3D view)."""
        if self.display is None:
            return
        w, h = self.display.getWidth(), self.display.getHeight()
        ih, iw = image_bgr.shape[:2]
        s = min(w / iw, h / ih)
        small = cv2.resize(image_bgr, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((h, w, 4), (20, 20, 20, 255), dtype=np.uint8)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1], :3] = small
        ref = self.display.imageNew(canvas.tobytes(), Display.BGRA, w, h)
        self.display.imagePaste(ref, 0, 0, False)
        self.display.imageDelete(ref)

    def step(self):
        return self.robot.step(self.dt_ms) != -1

    def time(self):
        return self.robot.getTime()

    def pose(self):
        """(x, y, yaw) of the robot centre in the world frame.

        x east, y north, yaw counter-clockwise from east. The GPS antenna sits off-centre,
        so its reading is shifted back to the centre (where the lidar is mounted).
        """
        gx, gy, _ = self.gps.getValues()
        nx, ny, _ = self.compass.getValues()
        yaw = math.atan2(nx, ny)
        x, y = lidar_pose_from_gps((gx, gy), yaw, GPS_OFFSET)
        return x, y, yaw

    def max_range(self):
        return self.lidar.getMaxRange()

    def scan(self):
        """List of (angle, range) pairs in the robot frame. Angle 0 = straight ahead, + = left.

        Webots orders a 360-degree lidar from +fov/2 down to -fov/2.
        """
        return list(zip(self.angles.tolist(), self.ranges().tolist()))

    def detections(self):
        """Objects the camera's Recognition sees: class, position relative to the camera (m)."""
        out = []
        for obj in self.camera.getRecognitionObjects():
            px, py, pz = obj.getPosition()
            out.append({"id": obj.getId(), "class": obj.getModel(), "rel": (px, py, pz)})
        return out

    def drive(self, vx, vy, wz):
        """Body velocities (m/s, m/s, rad/s) -> mecanum wheel speeds."""
        k = LX + LY
        speeds = (
            (vx - vy - k * wz),
            (vx + vy + k * wz),
            (vx + vy - k * wz),
            (vx - vy + k * wz),
        )
        for w, s in zip(self.wheels, speeds):
            w.setVelocity(s / WHEEL_RADIUS)


def sector_min(scan, centre, half_width):
    vals = [r for a, r in scan if abs(math.atan2(math.sin(a - centre), math.cos(a - centre))) <= half_width
            and math.isfinite(r)]
    return min(vals) if vals else float("inf")


def report(bot, scan):
    x, y, yaw = bot.pose()
    front = sector_min(scan, 0.0, math.radians(15))
    left = sector_min(scan, math.pi / 2, math.radians(15))
    right = sector_min(scan, -math.pi / 2, math.radians(15))
    back = sector_min(scan, math.pi, math.radians(15))
    print(f"[t={bot.time():6.1f}s] pose x={x:6.2f} y={y:6.2f} yaw={math.degrees(yaw):6.1f} deg | "
          f"free space front {front:4.1f} m, left {left:4.1f}, right {right:4.1f}, back {back:4.1f}")
    dets = bot.detections()
    for d in dets:
        rx, ry, _ = d["rel"]
        print(f"    sees {d['class']:<18} {math.hypot(rx, ry):4.1f} m away, "
              f"bearing {math.degrees(math.atan2(ry, rx)):5.1f} deg")


DRIVE_KEYS = {"W", "A", "S", "D", "Q", "E", Keyboard.UP, Keyboard.DOWN, Keyboard.LEFT, Keyboard.RIGHT}


def _ramp(current, target, max_step):
    return current + max(-max_step, min(max_step, target - current))


class Teleop:
    """Hold-to-drive: velocity follows the keys held right now and ramps to zero on release."""

    def __init__(self):
        self.vx = self.vy = self.wz = 0.0

    def update(self, held, dt):
        def axis(pos, neg):
            return (1.0 if any(k in held for k in pos) else 0.0) - (1.0 if any(k in held for k in neg) else 0.0)

        self.vx = _ramp(self.vx, MAX_V * axis(("W", Keyboard.UP), ("S", Keyboard.DOWN)), ACCEL_V * dt)
        self.vy = _ramp(self.vy, 0.6 * MAX_V * axis(("Q",), ("E",)), ACCEL_V * dt)
        self.wz = _ramp(self.wz, MAX_W * axis(("A", Keyboard.LEFT), ("D", Keyboard.RIGHT)), ACCEL_W * dt)
        return self.vx, self.vy, self.wz


def _sector_min(angles, ranges, centre, half_width):
    diff = np.arctan2(np.sin(angles - centre), np.cos(angles - centre))
    vals = ranges[(np.abs(diff) <= half_width) & np.isfinite(ranges)]
    return float(vals.min()) if vals.size else float("inf")


def collision_guard(vx, vy, angles, ranges):
    """Scale a commanded velocity down so the robot stops short of obstacles.

    This is the simplest form of shared control: the human chooses the direction,
    the robot vetoes only the part of the command that would cause a collision.
    Returns (vx, vy, limited) where limited is True if the command was reduced.
    """
    def limit(v, clearance, stop):
        if v <= 0:
            return v
        return min(v, MAX_V * max(0.0, (clearance - stop) / SLOW_ZONE))

    w = math.radians(28)
    new_vx = limit(vx, _sector_min(angles, ranges, 0.0, w), STOP_FRONT) if vx > 0 else \
        -limit(-vx, _sector_min(angles, ranges, math.pi, w), STOP_FRONT)
    new_vy = limit(vy, _sector_min(angles, ranges, math.pi / 2, w), STOP_SIDE) if vy > 0 else \
        -limit(-vy, _sector_min(angles, ranges, -math.pi / 2, w), STOP_SIDE)
    return new_vx, new_vy, (abs(new_vx - vx) > 1e-3 or abs(new_vy - vy) > 1e-3)


def main():
    bot = ExplorerRobot()
    teleop = Teleop()
    print(__doc__)

    x0, y0, _ = bot.pose() if bot.step() else (0.0, 0.0, 0.0)
    grid = OccupancyGrid((x0, y0), size_m=100.0, resolution=0.1)
    auto = "auto" in bot.args
    opts = dict(a.split("=", 1) for a in bot.args if "=" in a)      # e.g. label=manual limit=600
    time_limit = float(opts.get("limit", "inf"))                    # mission time limit (simulated seconds)
    finished = False
    manual_only = opts.get("label") == "manual"     # manual trials: autonomy and waypoints are locked out
    log = RunLog(mode=opts.get("label") or ("auto" if auto else "teleop"))
    explorer = Explorer((x0, y0))
    tracker = ArtifactTracker(camera_offset=CAMERA_OFFSET, seed=int(opts.get("seed", "1")) + 1)
    # The radio link is world physics, so it reads the true layout; the robot only learns up / down.
    gt = load_ground_truth()
    base_xy = (gt["base_station"]["x"], gt["base_station"]["y"])
    sup = Supervisor(CommsModel(base_xy, gt["layout"], gt["cell_size"]),
                     policy="explore_first" if "explore_first" in bot.args else "report_first")
    used_auto = auto
    window_keys = set()
    flash = ("", -1.0)          # (text, show until) for brief on-map confirmations
    interventions = 0
    overriding = False
    backup_until = -1.0
    progress_mark = (0.0, x0, y0, 0.0)      # (time, x, y, yaw) of the last noticeable movement
    trail = []
    image = None
    step = 0
    last_report = last_snapshot = last_log = -100.0
    window_ok = True
    window_on = "livemap" in bot.args    # the robot's own live view starts hidden; V shows it
    print(f"Logging this run to {log.dir}")

    pose = (x0, y0, 0.0)
    sim_t = 0.0

    guard_on = True
    guard_active = False
    prev_held = set()
    dt = bot.dt_ms / 1000.0

    base = BaseStation()
    waypoint = None             # operator waypoint the robot is driving to
    click = None                # world position of the operator's latest click on the console
    last_task = None
    to_world = None
    console_on = "nowindow" not in bot.args
    mouse_hooked = False
    demo_state = {}
    rec_dir = Path(opts["recout"]) if "recout" in opts else None   # demo video: save console frames here
    rec_every = float(opts.get("rec", "0.1"))
    demo_manual = "demomanual" in bot.args     # demo video: a script stands in for a human driver
    if rec_dir:
        rec_dir.mkdir(parents=True, exist_ok=True)
    selftest = [(15.0, WAYPOINT, {"x": 16.5, "y": -4.5}), (70.0, AUTO_ON, {}), (110.0, HOME, {})] \
        if "selftest" in bot.args else []

    def on_mouse(event, px, py, flags, param):
        nonlocal click
        if event == cv2.EVENT_LBUTTONDOWN and to_world is not None:
            click = to_world(px, py)

    def mode_text():
        if finished:
            return "TRIAL COMPLETE (saved)"
        if last_task == "WAYPOINT":
            return "GOING TO WAYPOINT"
        if last_task == "FAILSAFE":
            return "FAILSAFE: RETURNING TO SIGNAL"
        if not auto:
            return "MANUAL"
        if overriding:
            return "AUTO (operator override)"
        if sup.returning_to_comms and explorer.state == EXPLORING:
            return "AUTO: RETURNING TO COMMS"
        return f"AUTO: {explorer.state}"

    def status():
        guard = "OFF" if not guard_on else ("BLOCKING" if guard_active else "on")
        mode = mode_text()
        link = "UP" if sup.link_up else "DOWN"
        lines = [f"t = {sim_t:5.0f} s   mapped = {grid.free_area_m2():5.0f} m2   guard: {guard}   "
                 f"battery: {100 * sup.battery:3.0f}%",
                 f"mode: {mode}   interventions: {interventions}",
                 f"link: {link}   artifacts found: {len(tracker.confirmed())}   "
                 f"delivered to base: {len(sup.delivered_artifacts())}   waiting: {len(sup.outbox)}"]
        if sim_t < flash[1]:
            lines.append(flash[0])
        return lines

    def draw():
        show_plan = last_task is not None
        markers = [(fx, fy, (80, 200, 80), "") for fx, fy, _ in explorer.frontiers] if show_plan else []
        delivered = {m["payload"]["id"] for m in sup.delivered_artifacts()}
        markers += [(tr.x, tr.y, ARTIFACT_COLOURS[tr.cls],
                     tr.cls.replace("_", " ") + ("" if tr.id in delivered else " (not sent)"))
                    for tr in tracker.confirmed()]
        markers.append((base_xy[0], base_xy[1], (90, 220, 90) if sup.link_up else (60, 60, 230), "BASE"))
        return render(grid, pose, trail, status_lines=status(), markers=markers,
                      path=explorer.path if show_plan else (), goal=explorer.goal if show_plan else None)

    def say(text):
        print(f"[t={sim_t:6.1f}s] {text}")

    def snapshot():
        nonlocal image
        image = draw()
        log.snapshot(grid, image, sim_t, extra={"mode": "auto" if used_auto else "teleop",
                                                "label": opts.get("label"),
                                                "time_limit_s": time_limit if time_limit != float("inf") else None,
                                                "interventions": interventions,
                                                "explorer_state": explorer.state if auto else "MANUAL",
                                                "artifacts_reported": len(tracker.confirmed()),
                                                "operator_commands": [c["kind"] for c in base.history],
                                                "comms": sup.stats()})
        delivered = {m["payload"]["id"]: m["delivered_s"] for m in sup.delivered_artifacts()}
        log.artifacts([dict(r, delivered_s=delivered.get(r["id"])) for r in tracker.reports()])
        log.messages(sup.messages)
        save_console("console.png")

    def save_console(name):
        """Write the operator console, exactly as the operator would see it, into the run folder."""
        img, _ = render_console(base, sup.inbox, sim_t, sup.link_up, ARTIFACT_COLOURS, waypoint)
        cv2.imwrite(str(log.dir / name), img)

    try:
        while bot.step():
            step += 1
            sim_t = bot.time()
            held = bot.held_keys()
            pressed = (held - prev_held) | window_keys   # one-shot actions, from Webots or the map window
            window_keys = set()
            prev_held = held

            force_report = "P" in pressed
            if "M" in pressed:
                flash = (f"MAP SAVED to logs/{log.dir.name}", sim_t + 4.0)
                snapshot()
                print(f"Map saved to {log.dir}")
            if "G" in pressed:
                guard_on = not guard_on
                print(f"Collision guard {'ON' if guard_on else 'OFF'}")

            if "selftest_failsafe" in bot.args and auto and not sup.link_up:
                auto = False                             # test only: as if the operator had been driving by hand
            if not finished and (sim_t >= time_limit or (used_auto and explorer.state == DONE and "autoquit" in bot.args)):
                finished = True                          # tells the experiment runner this mission is over
                snapshot()
                (log.dir / "finished.flag").write_text(f"{sim_t:.1f}", encoding="utf-8")
                say(f"mission clock stopped at {sim_t:.0f} s: TRIAL COMPLETE AND SAVED. "
                    f"Reload the world (Ctrl+Shift+R) for the next one.")
            if "consoledemo" in bot.args and step % VIEW_EVERY == 0:
                # Figure capture: when the link has been down a while, an operator queues a waypoint
                # (it shows as PENDING), and the console is saved at moments worth illustrating.
                age = base.age(sim_t)
                if not sup.link_up and age is not None and age > 25 and not demo_state.get("queued"):
                    base.command(sim_t, WAYPOINT, x=13.5, y=-10.5)
                    demo_state["queued"] = True
                if not sup.link_up and age is not None and age > 40 and demo_state.get("queued") \
                        and not demo_state.get("stale"):
                    save_console("console_link_down.png")
                    demo_state["stale"] = True
                if sup.link_up and demo_state.get("stale") and not demo_state.get("restored") \
                        and len(sup.inbox) >= 6:
                    save_console("console_link_up.png")
                    demo_state["restored"] = True
            if selftest and sim_t > selftest[0][0]:      # scripted operator, used only by automated tests
                _, kind, payload = selftest.pop(0)
                base.command(sim_t, kind, **payload)
            # ---- demo-video helpers (never used in experiment trials) ----
            if "democlick" in bot.args and not demo_state.get("clicked") and sim_t > 150 and pose[0] > 26:
                base.command(sim_t, WAYPOINT, x=31.5, y=-10.5)   # the pocket autonomy keeps missing
                demo_state["clicked"] = True
            if rec_dir and sim_t >= demo_state.get("next_rec", 0.0):
                img, _ = render_console(base, sup.inbox, sim_t, sup.link_up, ARTIFACT_COLOURS, waypoint)
                cv2.imwrite(str(rec_dir / f"{int(round(sim_t * 1000)):08d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                (rec_dir / "events.txt").open("a", encoding="utf-8").write(
                    f"{sim_t:.1f}\t{mode_text()}\t{'UP' if sup.link_up else 'DOWN'}\t"
                    f"{len(tracker.confirmed())}\t{len(sup.delivered_artifacts())}\t{interventions}\n")
                demo_state["next_rec"] = demo_state.get("next_rec", 0.0) + rec_every

            # ---- operator commands: issued at the base station, they travel over the link ----
            if "T" in pressed:
                base.command(sim_t, AUTO_OFF if auto else AUTO_ON)
            if "H" in pressed:
                base.command(sim_t, HOME)
            if "X" in pressed:
                base.command(sim_t, RESUME)
            if click is not None:
                base.command(sim_t, WAYPOINT, x=click[0], y=click[1])
                click = None
            if ("T" in pressed or "H" in pressed or "X" in pressed) and not sup.link_up:
                say("command queued: no link to the robot")
            for cmd in base.uplink(sim_t, sup.link_up):
                kind = cmd["kind"]
                if manual_only and kind != RESUME:
                    say(f"ignored {kind}: this is a manual trial, drive with W/A/S/D only")
                    continue
                interventions += 1
                if kind in (AUTO_ON, AUTO_OFF):
                    auto = kind == AUTO_ON
                    explorer.path = []
                    if auto and explorer.state == DONE:
                        explorer.state = EXPLORING
                    used_auto = used_auto or auto
                    interventions -= 1          # choosing the mode is not an intervention
                    say(f"operator: autonomy {'ON' if auto else 'OFF'}")
                elif kind == WAYPOINT:
                    waypoint = (cmd["payload"]["x"], cmd["payload"]["y"])
                    explorer.path = []
                    say(f"operator: go to ({waypoint[0]:.1f}, {waypoint[1]:.1f})")
                elif kind == HOME:
                    waypoint = None
                    auto, used_auto = True, True
                    explorer.state = RETURNING
                    say("operator: return to base")
                elif kind == RESUME:
                    waypoint = None
                    explorer.path = []
                    say("operator: waypoint cancelled")
                progress_mark = (sim_t, pose[0], pose[1], pose[2])
            if "V" in pressed:
                window_on = not window_on
                if not window_on:
                    try:
                        cv2.destroyWindow(MAP_WINDOW)
                    except cv2.error:
                        pass

            ranges = bot.ranges()
            pose = bot.pose()
            if step % MAP_EVERY == 0:
                grid.update(pose, bot.angles, ranges, bot.max_range())
                if not trail or math.hypot(pose[0] - trail[-1][0], pose[1] - trail[-1][1]) > 0.2:
                    trail.append((pose[0], pose[1]))
                dets = [d for d in bot.detections() if d["class"] in ARTIFACT_COLOURS]
                for tr in tracker.observe(pose, dets, sim_t):
                    say(f"ARTIFACT FOUND #{tr.id}: {tr.cls} at ({tr.x:.1f}, {tr.y:.1f})")
                    sup.report_artifact(tr.as_report(), sim_t)
                moving = abs(teleop.vx) + abs(teleop.wz) > 0.05 or (auto and explorer.state != DONE)
                for event in sup.update(sim_t, pose, moving):
                    say(event)
                explorer.detour = sup.detour_target()
                if sup.must_return_home() and explorer.state == EXPLORING:
                    explorer.state = RETURNING

            # ---- who is driving? ------------------------------------------------
            # Joystick commands also travel over the radio: with the link down the keys do nothing.
            human_driving = (bool(held & DRIVE_KEYS) or demo_manual) and sup.link_up
            if human_driving and not overriding and (auto or waypoint is not None):
                interventions += 1              # the operator just took over from the robot
                waypoint = None
            overriding = human_driving and auto

            # What the robot should do on its own right now, if anything:
            #   a waypoint from the operator > lost-link failsafe (manual mode) > exploring (auto mode)
            failsafe = not auto and not sup.link_up and sup.last_link_pose is not None
            if human_driving:
                task = None
            elif waypoint is not None:
                task = "WAYPOINT"
            elif failsafe:
                task = "FAILSAFE"
            elif auto:
                task = "EXPLORE"
            else:
                task = None
            if task != last_task:
                explorer.path = []
                progress_mark = (sim_t, pose[0], pose[1], pose[2])
                if task == "FAILSAFE":
                    say("link lost in manual mode: failsafe driving back to the last place with signal")
                    sup.alert("FAILSAFE", sim_t, f"link lost under manual control at ({pose[0]:.1f}, {pose[1]:.1f}); "
                                                 f"driving back to the last place with signal")
                last_task = task

            if task is None and demo_manual and sup.link_up:
                # Scripted "operator" for the demo video: heads for unexplored areas like a person would,
                # but only while its commands can reach the robot.
                if step % PLAN_EVERY == 0 or not explorer.path:
                    explorer.plan(grid, pose)
                vx, wz, _ = follow(pose, explorer.path)
                vy = 0.0
            elif task is None:
                vx, vy, wz = teleop.update(held if sup.link_up else set(), dt)
            else:
                teleop.vx = teleop.vy = teleop.wz = 0.0
                replan = step % PLAN_EVERY == 0 or not explorer.path
                if task == "EXPLORE":
                    if replan and explorer.state != DONE:
                        before = explorer.state
                        explorer.plan(grid, pose)
                        if explorer.state != before:
                            say(f"explorer: {before} -> {explorer.state}")
                            if explorer.state == RETURNING:
                                sup.alert("EXPLORATION_COMPLETE", sim_t, "no unexplored areas left; returning to base")
                            elif explorer.state == DONE:
                                sup.alert("RETURNED_TO_BASE", sim_t, "mission finished")
                elif replan:
                    target = waypoint if task == "WAYPOINT" else sup.last_link_pose
                    if not explorer.goto(grid, pose, target) and task == "WAYPOINT":
                        sup.alert("WAYPOINT_UNREACHABLE", sim_t,
                                  f"no known route to ({target[0]:.1f}, {target[1]:.1f}); waypoint dropped")
                        say("waypoint unreachable on the current map: dropped")
                        waypoint = None
                vx, wz, reached = follow(pose, explorer.path)
                vy = 0.0
                if task == "WAYPOINT" and waypoint is not None and reached and explorer.path:
                    sup.alert("WAYPOINT_REACHED", sim_t, f"arrived at ({waypoint[0]:.1f}, {waypoint[1]:.1f})")
                    say("waypoint reached" + ("; resuming exploration" if auto else "; holding position"))
                    waypoint = None
                if sim_t < backup_until:
                    vx, wz = -0.3, 0.0
                # stuck check: has the robot moved or turned lately?
                t0, px, py, pyaw = progress_mark
                turned = abs(math.atan2(math.sin(pose[2] - pyaw), math.cos(pose[2] - pyaw)))
                idle_ok = (task == "EXPLORE" and explorer.state == DONE) or (task == "FAILSAFE" and reached)
                if math.hypot(pose[0] - px, pose[1] - py) > 0.25 or turned > 0.5 or idle_ok:
                    progress_mark = (sim_t, pose[0], pose[1], pose[2])
                elif sim_t - t0 > STUCK_WINDOW_S:
                    say(f"stuck near ({pose[0]:.1f}, {pose[1]:.1f}): backing up and trying another route")
                    sup.alert("STUCK", sim_t, f"no progress near ({pose[0]:.1f}, {pose[1]:.1f}); trying another route")
                    if task == "EXPLORE":
                        explorer.give_up_on_goal()
                    elif task == "WAYPOINT":
                        waypoint = None
                    backup_until = sim_t + BACKUP_S
                    progress_mark = (sim_t, pose[0], pose[1], pose[2])

            guard_active = False
            if guard_on:
                vx, vy, guard_active = collision_guard(vx, vy, bot.angles, ranges)
                if task is None:
                    teleop.vx, teleop.vy = vx, vy   # so the ramp restarts from the limited speed
            if finished and time_limit != float("inf"):
                vx = vy = wz = 0.0                  # trial over: the robot stops so it is obvious
            bot.drive(vx, vy, wz)

            # ---- downlink: the base station's picture refreshes only while the link is up ----
            if step % PLAN_EVERY == 0 and sup.link_up:
                base.downlink(sim_t, grid, pose, trail, {"mode": mode_text(), "battery": round(100 * sup.battery),
                                                         "mapped": round(grid.free_area_m2())})

            if step % VIEW_EVERY == 0:
                image = draw()
                bot.show_map(image)             # overlay inside the Webots 3D view
                if window_ok:
                    try:
                        if console_on:          # what the operator at the base station sees
                            console, to_world = render_console(base, sup.inbox, sim_t, sup.link_up,
                                                               ARTIFACT_COLOURS, waypoint)
                            cv2.imshow(CONSOLE_WINDOW, console)
                            cv2.setWindowProperty(CONSOLE_WINDOW, cv2.WND_PROP_TOPMOST, 1)
                            if not mouse_hooked:
                                cv2.setMouseCallback(CONSOLE_WINDOW, on_mouse)
                                mouse_hooked = True
                        if window_on:           # the robot's own live view (debugging / video)
                            cv2.imshow(MAP_WINDOW, image)
                        k = cv2.waitKey(1) & 0xFF
                        if chr(k).upper() in "MTVGPHXC":   # these keys also work when a map window has focus
                            window_keys.add(chr(k).upper())
                    except cv2.error as e:
                        window_ok = False
                        print(f"Console windows unavailable ({e}); the in-view overlay still works")
            if "C" in pressed:
                console_on = not console_on
                if not console_on:
                    try:
                        cv2.destroyWindow(CONSOLE_WINDOW)
                    except cv2.error:
                        pass
                    mouse_hooked = False

            t = bot.time()
            if t - last_log >= 1.0:
                log.pose(t, *pose, grid.free_area_m2())
                last_log = t
            if t - last_snapshot >= SNAPSHOT_EVERY_S:
                snapshot()
                last_snapshot = t
            if force_report:
                report(bot, bot.scan())
    finally:
        try:
            snapshot()
        except Exception as e:  # the simulation may already be shutting down
            print(f"Final snapshot failed: {e}")
        log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        (PROJECT_ROOT / "logs" / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

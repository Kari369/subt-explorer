"""Operator console image: the base station's (possibly stale) map plus a message feed.

Left: the last map the base station received, the robot's last known position, the
artifacts that have actually been reported, and any waypoint the operator has set.
Right: link status, how old the picture is, robot status, pending commands, and the
feed of alerts and reports in the order they arrived.

Returns the image and a function that converts a click on it into world coordinates.
"""

import cv2
import numpy as np

from core.map_view import render

PANEL_W = 380
MAP_PX = 700
HEIGHT = 800
FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN, RED, AMBER, WHITE, GREY = (90, 220, 90), (70, 70, 235), (0, 170, 255), (235, 235, 235), (150, 150, 150)
ALERT_COLOURS = {"LINK_LOST": RED, "LINK_RESTORED": GREEN, "STUCK": AMBER, "LOW_BATTERY": RED}


def _text(img, s, x, y, colour=WHITE, scale=0.45, bold=False):
    cv2.putText(img, s, (x, y), FONT, scale, colour, 2 if bold else 1, cv2.LINE_AA)


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur] if cur else lines


def render_console(base, inbox, t, link_up, artifact_colours, waypoint=None):
    canvas = np.full((HEIGHT, MAP_PX + PANEL_W, 3), 24, dtype=np.uint8)
    to_world = None

    if base.grid is None:
        _text(canvas, "Waiting for first data from the robot...", 30, 60, GREY, 0.6)
    else:
        markers = [(m["payload"]["x"], m["payload"]["y"], artifact_colours[m["payload"]["class"]],
                    m["payload"]["class"].replace("_", " ")) for m in inbox if m["kind"] == "ARTIFACT"]
        info = {}
        img = render(base.grid, base.pose, base.trail, target_px=MAP_PX, markers=markers,
                     goal=waypoint, title="", info=info)
        if not link_up:   # desaturate: this picture is old
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.addWeighted(img, 0.45, cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), 0.55, 0)
        h, w = min(img.shape[0], HEIGHT), min(img.shape[1], MAP_PX)
        canvas[:h, :w] = img[:h, :w]

        def to_world(px, py, _i=info, _g=base.grid, _w=w, _h=h):
            if not (0 <= px < _w and _i["bar_h"] <= py < _h):
                return None
            col = px / _i["scale"] + _i["c0"]
            row = _i["h"] - (py - _i["bar_h"]) / _i["scale"] + _i["r0"]
            return _g.origin[0] + col * _g.res, _g.origin[1] + row * _g.res

    # ---- side panel ------------------------------------------------------------
    x0 = MAP_PX + 14
    cv2.rectangle(canvas, (MAP_PX, 0), (MAP_PX + PANEL_W, HEIGHT), (34, 34, 34), -1)
    _text(canvas, "OPERATOR CONSOLE", x0, 28, AMBER, 0.6, True)
    _text(canvas, "base station view: only what the link delivered", x0, 46, GREY, 0.38)

    cv2.circle(canvas, (x0 + 8, 72), 8, GREEN if link_up else RED, -1, cv2.LINE_AA)
    _text(canvas, "LINK UP" if link_up else "LINK DOWN", x0 + 24, 78, GREEN if link_up else RED, 0.6, True)
    age = base.age(t)
    if age is not None and not link_up:
        _text(canvas, f"picture is {age:.0f} s old", x0 + 170, 78, RED, 0.45)

    st = base.status
    y = 104
    for line in (f"robot mode: {st.get('mode', '-')}",
                 f"battery: {st.get('battery', '-')}%    mapped: {st.get('mapped', '-')} m2",
                 f"mission time: {t:.0f} s"):
        _text(canvas, line, x0, y, WHITE if link_up else GREY)
        y += 20

    y += 8
    _text(canvas, "COMMANDS", x0, y, AMBER, 0.45, True)
    y += 18
    for line in ("click map: send robot to that point", "T: autonomy on/off     H: return to base",
                 "X: cancel waypoint     W A S D: drive (needs link)"):
        _text(canvas, line, x0, y, GREY, 0.38)
        y += 16
    for cmd in base.pending[-3:]:
        _text(canvas, f"PENDING (no link): {cmd['kind']}", x0, y + 4, RED, 0.42)
        y += 18

    y += 14
    _text(canvas, "MESSAGES FROM ROBOT", x0, y, AMBER, 0.45, True)
    y += 6
    rows = []
    for m in inbox:
        delay = m["delivered_s"] - m["created_s"]
        late = f" [+{delay:.0f}s late]" if delay > 5 else ""
        if m["kind"] == "ARTIFACT":
            colour, text = artifact_colours[m["payload"]["class"]], f"FOUND {m['text']}{late}"
        else:
            colour, text = ALERT_COLOURS.get(m["payload"].get("code"), WHITE), f"{m['text']}{late}"
        lines = _wrap(f"{m['delivered_s']:.0f}s  {text}", 50)
        rows.append((colour, lines))
    budget = (HEIGHT - y - 12) // 15
    shown = []
    for colour, lines in reversed(rows):      # newest at the bottom, drop the oldest if full
        if len(lines) > budget:
            break
        budget -= len(lines)
        shown.append((colour, lines))
    for colour, lines in reversed(shown):
        for i, line in enumerate(lines):
            y += 15
            _text(canvas, line if i == 0 else "      " + line, x0, y, colour, 0.38)
    return canvas, to_world

"""Compose the demo video and the article GIF from the footage recorded by record_demo.py.

Output (media/):
  demo_video.mp4   ~1.5 min, 1920x1080, for the Loom slideshow
  mine_demo.gif    ~10 s, for the article

Layout of mission shots: the 3D chase camera (left), the operator console exactly as
the base station saw it (top right), a status panel (bottom right) and a caption bar.
Captions are chosen from what the robot was actually doing at that moment.
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 25

BG = (24, 20, 16)
TAN, AMBER, WHITE, GREY = (236, 214, 168), (255, 170, 0), (240, 240, 240), (160, 155, 145)
GREEN, RED = (90, 220, 90), (235, 80, 70)
_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(f"C:/Windows/Fonts/{'segoeuib' if bold else 'segoeui'}.ttf", size)
    return _font_cache[key]


def frames(folder):
    return sorted((int(p.stem), p) for p in folder.glob("*.jpg"))


def nearest(items, t_ms):
    """items sorted by time; the last one at or before t_ms."""
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if items[mid][0] <= t_ms:
            lo = mid
        else:
            hi = mid - 1
    return items[lo]


def load_events(folder):
    ev = []
    for line in (folder / "events.txt").read_text(encoding="utf-8").splitlines():
        t, mode, link, found, delivered, interv = line.split("\t")
        ev.append((int(float(t) * 1000), {"mode": mode, "link": link, "found": int(found),
                                          "delivered": int(delivered), "interv": int(interv)}))
    return ev


def card(title, subtitle="", lines=()):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 90), title, font=font(64, True), fill=TAN, anchor="mm")
    if subtitle:
        d.text((W // 2, H // 2), subtitle, font=font(34), fill=AMBER, anchor="mm")
    for i, line in enumerate(lines):
        d.text((W // 2, H // 2 + 80 + 52 * i), line, font=font(32), fill=WHITE, anchor="mm")
    return img


def full_frame(path, title, sub):
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, H - 170, W, H], fill=(10, 8, 6, 200))
    d.text((60, H - 130), title, font=font(46, True), fill=TAN)
    d.text((60, H - 65), sub, font=font(30), fill=WHITE)
    return img


def mission_frame(cine_path, console_path, state, caption, note, segment, speed):
    img = Image.new("RGB", (W, H), BG)
    cine = Image.open(cine_path).convert("RGB").resize((1440, 810), Image.LANCZOS)
    img.paste(cine, (0, 0))
    # The console's own text is unreadable at this size, so show just its map (what the operator
    # can see of the mine right now) and redraw the status in large type below it.
    con = Image.open(console_path).convert("RGB").crop((0, 0, 700, 560))
    con = con.resize((480, 384), Image.LANCZOS)
    if state["link"] != "UP":
        con = Image.blend(con, Image.new("RGB", con.size, (40, 10, 10)), 0.25)
    img.paste(con, (1440, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # segment label + speed on the 3D view
    d.rounded_rectangle([20, 20, 20 + 24 + d.textlength(segment, font=font(30, True)), 70], 10, fill=(10, 8, 6, 190))
    d.text((32, 45), segment, font=font(30, True), fill=TAN, anchor="lm")
    sp = f"{speed}x speed"
    d.rounded_rectangle([1420 - 24 - d.textlength(sp, font=font(24)), 20, 1420, 62], 10, fill=(10, 8, 6, 190))
    d.text((1408, 41), sp, font=font(24), fill=GREY, anchor="rm")
    d.text((1450, con.height + 4), "operator console: last map received" if state["link"] == "UP"
           else "operator console: map FROZEN (no link)", font=font(20), fill=GREY if state["link"] == "UP" else RED)

    # status panel
    x0, y0 = 1460, con.height + 44
    up = state["link"] == "UP"
    d.ellipse([x0, y0 + 6, x0 + 26, y0 + 32], fill=GREEN if up else RED)
    d.text((x0 + 40, y0 + 19), "RADIO LINK UP" if up else "RADIO LINK DOWN", font=font(30, True),
           fill=GREEN if up else RED, anchor="lm")
    rows = [("Robot mode", state["mode"].replace("AUTO: ", "AUTO · ")),
            ("Objects found", str(state["found"])),
            ("Delivered to base", str(state["delivered"])),
            ("Operator interventions", str(state["interv"]))]
    for i, (k, v) in enumerate(rows):
        y = y0 + 70 + i * 62
        d.text((x0, y), k, font=font(22), fill=GREY)
        d.text((x0, y + 26), v, font=font(27, True), fill=WHITE)

    # caption bar
    d.rectangle([0, 810, 1440, H], fill=(14, 11, 8))
    d.text((40, 850), caption[0], font=font(42, True), fill=TAN)
    d.text((40, 915), caption[1], font=font(30), fill=WHITE)
    if note:
        d.text((40, 1030), note, font=font(22), fill=GREY)
    return img


def caption_for(seg, state, prev, t, memory):
    """Pick a caption from the robot's state, holding short-lived ones for a few seconds."""
    mode, link = state["mode"], state["link"]
    if prev and state["delivered"] > prev["delivered"] and prev["link"] == "DOWN" and link == "UP":
        memory["late"] = t + 40_000
    if prev and state["found"] > prev["found"]:
        memory["found"] = t + 30_000
    if seg == "manual":
        if "FAILSAFE" in mode:
            return ("Link lost: the controls go dead", "A failsafe drives the robot back to the last place it had signal.")
        if link == "DOWN":
            return ("Out of radio range", "Every driving command travels over the radio, so with no link, nothing gets through.")
        if t > 60_000:
            return ("Stuck at the edge of coverage", "Manual control never gets much past the radio's reach: about 48% of the mine.")
        return ("Manual teleop", "A human drives. The robot maps with its lidar as it goes.")
    if "WAYPOINT" in mode:
        return ("Operator click: \"check that pocket\"", "Autonomy alone missed this barrel in all six trials. One click sends the robot to look.")
    if "RETURNING TO COMMS" in mode:
        return ("Returning to comms", "The robot is holding reports the base hasn't received, so it drives back into range to deliver them.")
    if t < memory.get("late", -1) and link == "UP":
        return ("Link restored: late reports arrive", "Reports found out of contact reach the base station now, marked as late.")
    if link == "DOWN":
        return ("Link down: the operator's picture freezes", "The robot keeps exploring on its own and holds its reports until it has signal.")
    if "RETURNING" in mode:
        return ("Exploration complete", "No unexplored areas left: the robot heads back to base.")
    if "DONE" in mode or "COMPLETE" in mode:
        return ("Mission complete", f"{state['delivered']} of 8 objects delivered to the base station.")
    if t < memory.get("found", -1):
        return ("Object found", "Its camera spots it; repeated sightings are averaged, then a report is sent to base.")
    return ("Supervised autonomy", "The robot picks the edges of unexplored space (frontiers) and maps as it goes.")


class Writer:
    def __init__(self, path, fps=FPS):
        self.p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", str(path)],
                                  stdin=subprocess.PIPE)
        self.n = 0

    def add(self, img, repeat=1):
        data = np.asarray(img, dtype=np.uint8).tobytes()
        for _ in range(repeat):
            self.p.stdin.write(data)
            self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def fade(a, b, n):
    return [Image.blend(a, b, (i + 1) / (n + 1)) for i in range(n)]


def mission(seg, t_from, t_to, speed, note=""):
    cine = frames(MEDIA / seg / "cine")
    con = frames(MEDIA / seg / "console")
    events = load_events(MEDIA / seg / "console")
    out, prev, memory = [], None, {}
    step_ms = 1000 * speed / FPS
    t = t_from * 1000
    label = {"manual": "MANUAL TELEOP", "supervised": "SUPERVISED AUTONOMY"}[seg]
    while t <= min(t_to * 1000, cine[-1][0]):
        _, cp = nearest(cine, t)
        _, kp = nearest(con, t)
        _, state = nearest(events, t)
        cap = caption_for(seg, state, prev, t, memory)
        out.append(mission_frame(cp, kp, state, cap, note, label, speed))
        prev = state
        t += step_ms
    return out


def main():
    orbit = frames(MEDIA / "orbit" / "cine")
    title = card("Recreating DARPA SubT exploration", "a ground robot, an unknown mine, a radio link that drops out",
                 ["Webots simulation · explore · map · detect · report · supervise"])
    orbit_imgs = [full_frame(p, "An unknown mine", "666 m² of tunnel, 8 hidden objects: survivors, fire extinguishers, "
                                                   "equipment cases, gas barrels") for _, p in orbit[::2]]
    manual = mission("manual", 0, 104, 4, note="Driving here is scripted to imitate an operator, for the demo.")
    sup = mission("supervised", 0, 445, 10, note="Autonomous, with one scripted operator click, for the demo.")
    end = card("Results from 12 trials", "10-minute mission clock, SubT scoring (reports within 5 m)",
               ["Manual teleop: 48% mapped · 2 of 8 objects",
                "Autonomy alone: 92% mapped · 7 of 8",
                "Supervised autonomy: 91% mapped · 7.7 of 8 (all 8 in two of three runs)"])

    w = Writer(MEDIA / "demo_video.mp4")
    w.add(title, FPS * 3)
    for img in fade(title, orbit_imgs[0], 12) + orbit_imgs:
        w.add(img)
    for seq in (manual, sup):
        for img in fade(orbit_imgs[-1] if seq is manual else manual[-1], seq[0], 10):
            w.add(img)
        for img in seq:
            w.add(img)
        w.add(seq[-1], FPS)
    for img in fade(sup[-1], end, 12):
        w.add(img)
    w.add(end, FPS * 6)
    w.close()
    print(f"wrote media/demo_video.mp4 ({w.n / FPS:.0f} s)")

    # ---- the article GIF: aerial view, then the mission view around a found object ----
    gif_dir = MEDIA / "gif_frames"
    gif_dir.mkdir(exist_ok=True)
    for f in gif_dir.glob("*.png"):
        f.unlink()
    events = load_events(MEDIA / "supervised" / "console")
    first_down = next(t for t, s in events if s["link"] == "DOWN")
    sup_gif = mission("supervised", max(0, first_down / 1000 - 30), first_down / 1000 + 30, 10)
    seq = orbit_imgs[::2][:40] + sup_gif[::2][:60]           # ~4 s aerial + ~6 s mission, at 10 fps
    for i, img in enumerate(seq):
        img.resize((800, 450), Image.LANCZOS).save(gif_dir / f"{i:04d}.png")
    pattern = str(gif_dir / "%04d.png")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "10", "-i", pattern, "-vf",
                    "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-loop", "0", str(MEDIA / "mine_demo.gif")], check=True)
    size = (MEDIA / "mine_demo.gif").stat().st_size / 1e6
    print(f"wrote media/mine_demo.gif ({len(seq) / 10:.0f} s, {size:.1f} MB)")


if __name__ == "__main__":
    main()

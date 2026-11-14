"""Record the raw footage for the demo video (background Webots runs, roof removed for filming).

Clips, each saved under media/<clip>/:
  orbit       aerial orbit of the whole mine                          (cine frames only)
  manual      a script drives like an operator; the link cuts it off   (cine + console frames)
  supervised  autonomy, plus one operator click into the missed pocket (cine + console frames)

Usage (from the project folder):  python tools/record_demo.py [orbit] [manual] [supervised]
Then:                             python tools/make_demo_video.py
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
WEBOTS = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Webots/msys64/mingw64/bin/webots.exe"

CLIPS = {
    # name: (explorer controller args, cinecam args, sim seconds to record)
    "orbit": ("label=demo", "mode=orbit every=0.05 spin=16 r=34 h=34", 16.5),
    "manual": ("label=demo demomanual rec=0.2", "mode=chase every=0.2", 170),
    "supervised": ("auto label=demo democlick autoquit rec=0.4", "mode=chase every=0.4", 520),
}


def pids():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq webots-bin.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return {int(line.split('","')[1]) for line in out.splitlines() if "webots-bin" in line}


def record(name):
    ex_args, cam_args, end = CLIPS[name]
    out = MEDIA / name
    shutil.rmtree(out, ignore_errors=True)
    cine, con = (out / "cine").as_posix(), (out / "console").as_posix()
    if "rec=" in ex_args:
        ex_args += f" recout={con}"
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_world.py"), "--open", "--nowindow",
                    "--args", ex_args, "--cinecam", f"{cam_args} out={cine} end={end}", "--out", "demo.wbt"],
                   check=True, stdout=subprocess.DEVNULL)
    before = pids()
    subprocess.Popen([str(WEBOTS), "--batch", "--minimize", "--mode=fast", str(ROOT / "worlds" / "demo.wbt")])
    every = float(cam_args.split("every=")[1].split()[0])
    want, t0 = int(end / every) - 2, time.time()
    print(f"recording {name} ({end:.0f} simulated s)...", flush=True)
    while time.time() - t0 < 1200:
        time.sleep(5)
        n = len(list((out / "cine").glob("*.jpg"))) if (out / "cine").exists() else 0
        if n >= want:
            break
    time.sleep(3)
    for p in pids() - before:
        subprocess.run(["taskkill", "/F", "/PID", str(p)], capture_output=True)
    time.sleep(2)
    print(f"  {name}: {n} camera frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    for clip in sys.argv[1:] or list(CLIPS):
        record(clip)
    for f in ("demo.wbt", ".demo.wbproj"):
        (ROOT / "worlds" / f).unlink(missing_ok=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_world.py")], stdout=subprocess.DEVNULL)

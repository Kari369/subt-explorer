"""Run and report the experiments for the article.

Every trial has the same mission clock (default 600 simulated seconds). SubT-style
scoring: only artifact reports that have REACHED THE BASE STATION before the clock
stops count, and they must be within 5 m of the real object.

Conditions
  auto_report   autonomous; returns to radio range to deliver reports (unattended)
  auto_explore  autonomous; never turns back, delivers when it gets home (unattended)
  manual        you drive with W/A/S/D only (needs you)
  supervised    autonomy on, you watch the console and step in (needs you)

Usage (from the project folder)
  python tools/experiment.py auto [--trials 3]     run both unattended conditions in the background
  python tools/experiment.py setup manual          prepare mine.wbt for a manual trial, then reload Webots
  python tools/experiment.py setup supervised      same, for a supervised trial
  python tools/experiment.py setup normal          put mine.wbt back to normal
  python tools/experiment.py report                build docs/results.md, results.csv and results.png
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.artifacts import score  # noqa: E402
from core.truth import load_ground_truth  # noqa: E402

LIMIT_S = 600
CONDITIONS = {
    "auto_report": "auto autoquit",
    "auto_explore": "auto autoquit explore_first",
    "manual": "",                 # no way to know a manual run is "done": it always runs the full clock
    "supervised": "autoquit",     # ends when the robot is back at base with exploration complete
}
NAMES = {"manual": "Manual teleop", "auto_report": "Autonomy, return to comms",
         "auto_explore": "Autonomy, never turn back", "supervised": "Supervised autonomy"}
WEBOTS = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Webots/msys64/mingw64/bin/webots.exe"


def build(label, extra, out="mine.wbt", seed=0, nowindow=False):
    cmd = [sys.executable, str(ROOT / "tools" / "build_world.py"), "--out", out, "--seed", str(seed),
           "--args", f"label={label} limit={LIMIT_S} {extra}".strip()]
    if nowindow:
        cmd.append("--nowindow")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def webots_pids():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq webots-bin.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return {int(line.split('","')[1]) for line in out.splitlines() if "webots-bin" in line}


def run_auto(trials):
    for cond in ("auto_report", "auto_explore"):
        for k in range(1, trials + 1):
            build(cond, CONDITIONS[cond], out="experiment.wbt", seed=k, nowindow=True)
            before, known = webots_pids(), set(p.name for p in (ROOT / "logs").glob("run_*"))
            subprocess.Popen([str(WEBOTS), "--batch", "--minimize", "--mode=fast",
                              str(ROOT / "worlds" / "experiment.wbt")])
            print(f"{cond} trial {k}/{trials} running...", flush=True)
            flag, t0 = None, time.time()
            while time.time() - t0 < 900:
                time.sleep(5)
                new = [p for p in (ROOT / "logs").glob(f"run_*_{cond}") if p.name not in known]
                if new and (new[-1] / "finished.flag").exists():
                    flag = new[-1]
                    break
            time.sleep(2)
            for pid in webots_pids() - before:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            time.sleep(3)
            print(f"  -> {flag.name if flag else 'TIMED OUT (no result)'}", flush=True)
    for f in ("experiment.wbt", ".experiment.wbproj"):
        (ROOT / "worlds" / f).unlink(missing_ok=True)
    report()


def trial_result(run, gt):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    limit = summary.get("time_limit_s") or LIMIT_S
    end = float((run / "finished.flag").read_text()) if (run / "finished.flag").exists() else summary["sim_time_s"]
    end = min(end, limit)
    reports = json.loads((run / "artifacts.json").read_text(encoding="utf-8"))
    in_time = [r for r in reports if r.get("delivered_s") is not None and r["delivered_s"] <= limit]
    found = [r for r in reports if (r.get("confirmed_s") or 1e9) <= limit]
    sc = score(in_time, gt["artifacts"], gt.get("scoring_radius_m", 5.0))
    with open(run / "trajectory.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if float(r["t"]) <= limit]
    mapped = float(rows[-1]["mapped_free_m2"]) if rows else 0.0
    delays = [r["delivered_s"] - r["confirmed_s"] for r in in_time]
    comms = summary.get("comms", {})
    return {
        "condition": summary.get("label"), "run": run.name,
        "mission_time_s": round(end), "finished_early": end < limit - 1,
        "coverage_pct": round(100 * min(mapped, gt["open_area_m2"]) / gt["open_area_m2"], 1),
        "found": len(found), "points": sc["points"], "possible": sc["possible"],
        "mean_error_m": sc["mean_error_m"],
        "mean_delay_s": round(sum(delays) / len(delays), 1) if delays else None,
        "max_delay_s": round(max(delays), 1) if delays else None,
        "out_of_contact_s": comms.get("time_out_of_contact_s"),
        "returns_to_comms": comms.get("returns_to_comms"),
        "interventions": summary.get("interventions"),
        "curve": [(float(r["t"]), float(r["mapped_free_m2"])) for r in rows],
    }


def report():
    gt = load_ground_truth()
    trials = []
    for run in sorted((ROOT / "logs").glob("run_*")):
        try:   # a run that was cut off mid-save can leave an empty or half-written file: skip it
            label = json.loads((run / "summary.json").read_text(encoding="utf-8")).get("label")
            if label in CONDITIONS and (run / "finished.flag").exists():
                trials.append(trial_result(run, gt))
        except (OSError, ValueError, KeyError) as e:
            print(f"skipping {run.name}: {type(e).__name__}")
    if not trials:
        raise SystemExit("No labelled experiment runs found in logs/.")

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    cols = [c for c in trials[0] if c != "curve"]
    with open(docs / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: t[c] for c in cols} for t in trials])

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(v, nd=1):
        return "-" if v is None else f"{v:.{nd}f}"

    lines = [f"# Results ({LIMIT_S // 60}-minute mission clock)", "",
             "Points = artifact reports that reached the base station before the clock stopped and are "
             "within 5 m of the real object (the SubT rule). Means over the trials of each condition.", "",
             "| Condition | Trials | Coverage % | Found | Points (of 8) | Mean error m | Mean report delay s "
             "| Out of contact s | Mission time s | Interventions |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cond in ("manual", "auto_explore", "auto_report", "supervised"):
        ts = [t for t in trials if t["condition"] == cond]
        if not ts:
            continue
        lines.append(f"| {NAMES[cond]} | {len(ts)} | {fmt(mean([t['coverage_pct'] for t in ts]))} "
                     f"| {fmt(mean([t['found'] for t in ts]))} | {fmt(mean([t['points'] for t in ts]))} "
                     f"| {fmt(mean([t['mean_error_m'] for t in ts]), 2)} | {fmt(mean([t['mean_delay_s'] for t in ts]))} "
                     f"| {fmt(mean([t['out_of_contact_s'] for t in ts]), 0)} "
                     f"| {fmt(mean([t['mission_time_s'] for t in ts]), 0)} "
                     f"| {fmt(mean([t['interventions'] for t in ts]))} |")
    lines += ["", "## Every trial", "", "| Condition | Run | Coverage % | Found | Points | Mission time s |", "|---|---|---|---|---|---|"]
    lines += [f"| {NAMES[t['condition']]} | {t['run']} | {t['coverage_pct']} | {t['found']} | {t['points']} "
              f"| {t['mission_time_s']} |" for t in trials]
    (docs / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"manual": "#7f8c8d", "auto_explore": "#2f80ed", "auto_report": "#e67e22", "supervised": "#27ae60"}
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    seen = set()
    for t in trials:
        c = t["condition"]
        ax.plot([p[0] / 60 for p in t["curve"]], [100 * min(p[1], gt["open_area_m2"]) / gt["open_area_m2"] for p in t["curve"]],
                color=colours[c], lw=1.8, alpha=0.85, label=NAMES[c] if c not in seen else None)
        seen.add(c)
    ax.set_xlabel("mission time (simulated minutes)")
    ax.set_ylabel("mine mapped (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exploration over time, every trial")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(docs / "results.png")
    print("\n".join(lines[:4 + 2 + len(NAMES)]))
    print(f"\nWrote docs/results.md, docs/results.csv, docs/results.png ({len(trials)} trials)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["auto", "setup", "report"])
    ap.add_argument("condition", nargs="?", choices=["manual", "supervised", "normal"])
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    if a.action == "auto":
        run_auto(a.trials)
    elif a.action == "report":
        report()
    else:
        if a.condition in (None, "normal"):
            subprocess.run([sys.executable, str(ROOT / "tools" / "build_world.py")], check=True)
            print("mine.wbt is back to normal. Reload the world in Webots (Ctrl+Shift+R).")
        else:
            build(a.condition, CONDITIONS[a.condition], seed=int(time.time()) % 1000)
            how = "drive with W/A/S/D only; do not press T" if a.condition == "manual" else \
                "press T to start autonomy, then step in from the console whenever you judge it useful"
            print(f"mine.wbt is set up for a '{a.condition}' trial with a {LIMIT_S // 60}-minute clock.\n"
                  f"Reload the world in Webots (Ctrl+Shift+R), press Play, and {how}.\n"
                  f"The console prints 'mission clock stopped' when the trial is over. Reload to run another.")


if __name__ == "__main__":
    main()

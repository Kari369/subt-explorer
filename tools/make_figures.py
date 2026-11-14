"""Render the article's tables as images (Medium has no tables) into docs/figures/.

Usage (from the project folder):  python tools/make_figures.py
"""

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

HEAD_BG, HEAD_FG = "#1f2a36", "white"
ROW_A, ROW_B, EDGE = "#ffffff", "#f3f5f7", "#d5dae0"
STATUS = {"Reproduced": "#1e8449", "Simplified": "#b9770e", "Left out": "#922b21"}


# Warm "mine" theme, matched to the article's Scope table
MINE = {"head_bg": "#4b3e31", "head_fg": "#ecd6a8", "row_a": "#e6d4ae", "row_b": "#dbc49b",
        "edge": "#c9b087", "text": "#3a2e22", "bold_first": True}


def table_image(path, columns, rows, widths, wrap, title=None, colour_col=None, bold_rows=(), theme=None):
    """widths: relative column widths; wrap: max characters per line for each column."""
    cells = [[textwrap.fill(str(v), w) for v, w in zip(r, wrap)] for r in rows]
    head = [textwrap.fill(c, w) for c, w in zip(columns, wrap)]
    line_counts = [max(c.count("\n") + 1 for c in r) for r in [head] + cells]
    line_h, pad = 0.26, 0.2
    heights = [n * line_h + pad for n in line_counts]
    total_w = 11.0
    fig_h = sum(heights) + (0.6 if title else 0.15)
    fig, ax = plt.subplots(figsize=(total_w, fig_h), dpi=170)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    xs = [0.0]
    for w in widths:
        xs.append(xs[-1] + w / sum(widths))
    y = fig_h - (0.55 if title else 0.08)
    if title:
        ax.text(0, fig_h - 0.12, title, fontsize=13, weight="bold", va="top", color="#1f2a36")
    t = theme or {"head_bg": HEAD_BG, "head_fg": HEAD_FG, "row_a": ROW_A, "row_b": ROW_B, "edge": EDGE,
                  "text": "#1b1f24", "bold_first": False}
    for i, (row, h) in enumerate(zip([head] + cells, heights)):
        is_head = i == 0
        bg = t["head_bg"] if is_head else (t["row_a"] if i % 2 else t["row_b"])
        for j, text in enumerate(row):
            if theme:   # horizontal rules only, like the Scope table
                ax.add_patch(plt.Rectangle((xs[j], y - h), xs[j + 1] - xs[j], h, facecolor=bg, lw=0))
                ax.plot([xs[j], xs[j + 1]], [y - h, y - h], color=t["edge"], lw=0.8)
            else:
                ax.add_patch(plt.Rectangle((xs[j], y - h), xs[j + 1] - xs[j], h, facecolor=bg,
                                           edgecolor=t["edge"], lw=0.8))
            colour = t["head_fg"] if is_head else t["text"]
            weight = "bold" if is_head or (i - 1) in bold_rows or (t["bold_first"] and j == 0) else "normal"
            if not is_head and j == colour_col:
                key = next((k for k in STATUS if text.startswith(k)), None)
                if key:
                    colour, weight = STATUS[key], "bold"
            ax.text(xs[j] + 0.008, y - h / 2, text, fontsize=9.5, va="center", ha="left",
                    color=colour, weight=weight, linespacing=1.25)
        y -= h
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / path, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote docs/figures/{path}")


def scope():
    rows = [
        ("Lidar mapping", "Reproduced, in 2D", "SubT teams mapped in 3D"),
        ("Autonomous exploration", "Reproduced", "My own frontier-based planner, not Team Explorer's TARE"),
        ("Path planning and following", "Reproduced", "Dijkstra search on a safety-padded map, with a pure-pursuit follower"),
        ("Obstacle sensing", "Simplified", "A 5-layer lidar flattened into one 2D scan, so low obstacles are still seen"),
        ("Object detection and localization", "Simplified", "Webots' built-in camera recognition, with sensor noise added; "
         "repeated sightings are averaged; scored with SubT's 5 m rule"),
        ("Comms-limited reporting", "Simplified", "Signal fades with distance and rock; reports queue while the link is down; "
         "the robot can drive back into range to deliver them. No real radios or relay nodes"),
        ("Human supervision", "Reproduced", "One operator, one robot. The operator sees only what the link delivered, "
         "and can drive, click waypoints or call the robot home"),
        ("Robot-initiated alerts", "Reproduced", "Link lost and restored, stuck, low battery, exploration complete"),
        ("Battery", "Simplified", "A simple timer; it never ran low in my trials"),
        ("Position estimation (SLAM)", "Left out", "The simulator supplies the robot's position"),
        ("Robot fleets, drones, legged robots", "Left out", "Out of scope for one week"),
    ]
    table_image("table_scope.png", ("Capability", "Status", "Notes"), rows, (3, 1.8, 6), (30, 12, 68), colour_col=1)


def results():
    names = {"manual": "Manual teleop", "auto_explore": "Autonomy, never turn back",
             "auto_report": "Autonomy, return to comms", "supervised": "Supervised autonomy"}
    with open(ROOT / "docs" / "results.csv", encoding="utf-8") as f:
        trials = list(csv.DictReader(f))

    def mean(cond, key):
        vals = [float(t[key]) for t in trials if t["condition"] == cond and t[key] not in ("", "None")]
        return sum(vals) / len(vals) if vals else 0.0

    rows = []
    for cond in ("manual", "auto_explore", "auto_report", "supervised"):
        n = sum(t["condition"] == cond for t in trials)
        rows.append((names[cond], n, f"{mean(cond, 'coverage_pct'):.1f}%", f"{mean(cond, 'points'):.1f}",
                     f"{mean(cond, 'mean_delay_s'):.0f} s", f"{mean(cond, 'out_of_contact_s'):.0f} s",
                     f"{mean(cond, 'mission_time_s') / 60:.1f} min", f"{mean(cond, 'interventions'):.1f}"))
    table_image("table_results.png",
                ("Condition", "Trials", "Mine mapped", "Points (of 8)", "Mean report delay", "Time out of contact",
                 "Mission time", "Interventions"),
                rows, (3.6, 0.9, 1.2, 1.1, 1.3, 1.4, 1.2, 1.7), (28, 8, 8, 8, 10, 12, 8, 14), theme=MINE)


def comparison():
    rows = [
        ("Platform", "Wheeled, tracked and legged robots, plus drones", "One wheeled robot in Webots"),
        ("Position estimation", "Lidar SLAM on real sensors", "Supplied by the simulator"),
        ("Mapping", "3D", "2D occupancy grid"),
        ("Exploration", "TARE, GBPlanner and others", "My own frontier-based planner"),
        ("Object reports", "Trained detectors on several sensors, with human review",
         "Built-in camera recognition with added noise; averaged sightings; 5 m scoring"),
        ("Communications", "Mesh radios and dropped relay nodes", "Simulated signal that fades with range and rock; no relays"),
        ("Human role", "One supervisor for a whole fleet", "One supervisor, one robot"),
        ("Mission", "60 minutes on a course of tunnel, urban and cave sections", "10 minutes in one 666 m² mine"),
    ]
    table_image("table_comparison.png", ("Capability", "SubT teams (2021)", "My version"), rows,
                (2.2, 4.2, 4.6), (22, 46, 50), theme=MINE)


def teams():
    rows = [
        ("CERBERUS", "ETH Zurich, NTNU and partners", "1st, 23 points (won on tie-break)"),
        ("CSIRO Data61", "CSIRO and partners", "2nd, 23 points"),
        ("MARBLE", "University of Colorado Boulder and partners", "3rd, 18 points"),
        ("Explorer", "Carnegie Mellon and Oregon State", "17 points; CMU reports it explored more of the course than any other team"),
    ]
    table_image("table_teams.png", ("Team", "Who", "SubT Final result"), rows, (2, 4, 5), (20, 42, 52))


if __name__ == "__main__":
    scope()
    results()
    comparison()
    teams()

"""Draw a top-down map of the mine from config/ground_truth.json.

Shows the tunnels, the start / base station and every hidden object, with world
coordinates on the axes. Useful for finding objects in Webots and for the article.

Usage (from the project folder):  python tools/plot_map.py
Writes docs/mine_map.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STYLE = {
    "survivor": ("#ff7a00", "o", "Survivor"),
    "fire_extinguisher": ("#e53935", "s", "Fire extinguisher"),
    "equipment_case": ("#2f80ed", "D", "Equipment case"),
    "gas_hazard": ("#f2c94c", "^", "Gas hazard"),
}


def main():
    gt = json.loads((ROOT / "config" / "ground_truth.json").read_text(encoding="utf-8"))
    cell, layout = gt["cell_size"], gt["layout"]
    rows, cols = len(layout), len(layout[0])

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=130)
    fig.patch.set_facecolor("#16130f")
    ax.set_facecolor("#3a332b")
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            if ch != "#":
                ax.add_patch(Rectangle((c * cell, -(r + 1) * cell), cell, cell, color="#d9c7a7", lw=0))

    s = gt["start"]
    ax.plot(s["x"], s["y"], marker="*", ms=20, color="#27ae60", mec="white", zorder=5)
    ax.annotate("START / base station", (s["x"], s["y"]), xytext=(6, 10), textcoords="offset points",
                color="white", fontsize=10, weight="bold")

    seen = set()
    for a in gt["artifacts"]:
        color, marker, label = STYLE[a["class"]]
        ax.plot(a["x"], a["y"], marker=marker, ms=12, color=color, mec="black", zorder=6,
                label=label if a["class"] not in seen else None)
        seen.add(a["class"])
        ax.annotate(f"{a['name']}\n({a['x']:.1f}, {a['y']:.1f})", (a["x"], a["y"]), xytext=(8, -4),
                    textcoords="offset points", color="black", fontsize=7.5, weight="bold")

    ax.set_xlim(0, cols * cell)
    ax.set_ylim(-rows * cell, 0)
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    ax.set_xlabel("x (m, east)", color="white")
    ax.set_ylabel("y (m, north)", color="white")
    ax.set_title(f"Mine layout: {gt['open_area_m2']:.0f} m² of tunnel, {len(gt['artifacts'])} hidden objects",
                 color="white", fontsize=13, weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, facecolor="#16130f",
              labelcolor="white", fontsize=9, frameon=False)

    out = ROOT / "docs" / "mine_map.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

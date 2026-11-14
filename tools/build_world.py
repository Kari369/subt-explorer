"""Generate the underground mine world for the SubT-style exploration project.

The mine is described as a character grid (LAYOUT below). Each cell is CELL metres
square: '#' is solid rock, '.' is tunnel, 'S' is the start / base station.
Running this script writes:
  worlds/mine.wbt            the Webots world
  config/ground_truth.json   artifact positions, layout and base station (for scoring)

Usage (from the project folder):
  python tools/build_world.py            # with ceiling (normal runs)
  python tools/build_world.py --open     # no ceiling (for overview shots)
"""

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBOTS = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"
# Textures live in worlds/textures so the world works offline (copied from Cyberbotics' Webots repo).
ROCK_TEX = "textures/rock.jpg"
ROCK_NORMAL = "textures/rough_concrete_normal.jpg"

CELL = 3.0          # metres per grid cell (tunnel width)
WALL_H = 3.2        # tunnel height
SEED = 7

LAYOUT = [
    "###############",
    "#S.......#....#",
    "#.###.##.#.##.#",
    "#.#.....#...#.#",
    "#.#.###.###.#.#",
    "#...#.....#...#",
    "###.#.###.#.###",
    "#.....#.......#",  # col 10 here is the only link between the east and west halves
    "#.###.#.#####.#",
    "#.#.......#...#",
    "###############",
]

# Artifacts hidden in the mine: (row, col, class, side it leans against)
ARTIFACTS = [
    (9, 5, "survivor", "S"),
    (1, 12, "survivor_lying", "N"),
    (5, 8, "fire_extinguisher", "N"),
    (7, 2, "fire_extinguisher", "S"),
    (3, 4, "equipment_case", "N"),
    (9, 12, "equipment_case", "S"),
    (7, 8, "gas_hazard", "N"),
    (3, 10, "gas_hazard", "S"),
]

RECOGNITION_COLORS = {
    "survivor": "1 0.45 0",
    "fire_extinguisher": "0.9 0.1 0.1",
    "equipment_case": "0.2 0.6 1",
    "gas_hazard": "1 0.9 0",
}

DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}


def is_open(r, c):
    return 0 <= r < len(LAYOUT) and 0 <= c < len(LAYOUT[0]) and LAYOUT[r][c] != "#"


def cell_center(r, c):
    """Grid cell -> world (x, y). Row 0 is the north edge; x points east, y north."""
    return c * CELL + CELL / 2, -(r * CELL + CELL / 2)


def check_connected():
    start = next((r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch == "S")
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS.values():
            n = (r + dr, c + dc)
            if is_open(*n) and n not in seen:
                seen.add(n)
                stack.append(n)
    all_open = {(r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch != "#"}
    missing = all_open - seen
    if missing:
        raise SystemExit(f"Layout has unreachable tunnel cells: {sorted(missing)}")
    for r, c, cls, _ in ARTIFACTS:
        if not is_open(r, c):
            raise SystemExit(f"Artifact {cls} at {(r, c)} is inside rock")
    return start, len(all_open)


def fmt(*v):
    return " ".join(f"{x:.4g}" for x in v)


def header():
    protos = [
        "objects/backgrounds/protos/TexturedBackground.proto",
        "robots/robotnik/summit_xl_steel/protos/SummitXlSteel.proto",
        "objects/rocks/protos/Rock.proto",
        "objects/factory/fire_extinguisher/protos/FireExtinguisher.proto",
        "objects/obstacles/protos/OilBarrel.proto",
        "objects/factory/containers/protos/MetalStorageBox.proto",
        "objects/factory/containers/protos/WoodenBox.proto",
        "objects/lights/protos/ConstructionLamp.proto",
        "humans/pedestrian/protos/Pedestrian.proto",
        "appearances/protos/DryMud.proto",
        "appearances/protos/RoughPine.proto",
        "appearances/protos/RustyMetal.proto",
    ]
    out = ["#VRML_SIM R2025a utf8", ""]
    out += [f'EXTERNPROTO "{WEBOTS}/{p}"' for p in protos]
    out.append("")
    return out


def world_settings(start_xy, open_top):
    x, y = start_xy
    fog = "" if open_top else """Fog {
  color 0.06 0.055 0.05
  fogType "EXPONENTIAL"
  visibilityRange 28
}"""
    return f"""WorldInfo {{
  info [ "SubT-style underground exploration: a Summit-XL Steel explores an unknown mine." ]
  title "Underground mine exploration"
  basicTimeStep 32
  contactProperties [
    ContactProperties {{
      material1 "InteriorWheelMat"
      coulombFriction [ 0, 2, 0 ]
      frictionRotation -0.785398 0
      bounce 0
      forceDependentSlip [ 10, 0 ]
      softCFM 0.0001
    }}
    ContactProperties {{
      material1 "ExteriorWheelMat"
      coulombFriction [ 0, 2, 0 ]
      frictionRotation 0.785398 0
      bounce 0
      forceDependentSlip [ 10, 0 ]
      softCFM 0.0001
    }}
  ]
}}
Viewpoint {{
  orientation 0 1 0 0.28
  position {fmt(x - 3.2, y, 1.75)}
  near 0.1
  exposure 1.5
  follow "explorer"
  followType "Mounted Shot"
}}
Background {{
  skyColor [ 0.12 0.1 0.09 ]
  luminosity 0.3
}}
{fog}
"""


def rock_walls():
    """One rock block per rock cell that touches a tunnel (others are never seen)."""
    out, first = [], True
    for r, row in enumerate(LAYOUT):
        for c, ch in enumerate(row):
            if ch != "#":
                continue
            touches = any(is_open(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            if not touches:
                continue
            x, y = cell_center(r, c)
            app = ("DEF ROCK_APP PBRAppearance { baseColor 0.62 0.56 0.5 roughness 1 metalness 0 "
                   f'baseColorMap ImageTexture {{ url [ "{ROCK_TEX}" ] }} '
                   f'normalMap ImageTexture {{ url [ "{ROCK_NORMAL}" ] }} normalMapFactor 2.5 '
                   "textureTransform TextureTransform { scale 1.3 1.3 } }") if first else "USE ROCK_APP"
            geom = f"DEF WALL_BOX Box {{ size {fmt(CELL, CELL, WALL_H)} }}" if first else "USE WALL_BOX"
            first = False
            out.append(
                f"Solid {{ translation {fmt(x, y, WALL_H / 2)} name \"rock_{r}_{c}\" "
                f"children [ Shape {{ appearance {app} geometry {geom} }} ] boundingObject USE WALL_BOX }}"
            )
    return out


def floor_and_ceiling(open_top):
    w, h = len(LAYOUT[0]) * CELL, len(LAYOUT) * CELL
    cx, cy = w / 2, -h / 2
    out = [
        f"""Solid {{
  translation {fmt(cx, cy, -0.1)}
  name "mine_floor"
  children [ Shape {{ appearance DryMud {{ textureTransform TextureTransform {{ scale {fmt(w / 4, h / 4)} }} }}
    geometry DEF FLOOR_BOX Box {{ size {fmt(w, h, 0.2)} }} }} ]
  boundingObject USE FLOOR_BOX
}}"""
    ]
    if not open_top:
        out.append(
            f"Solid {{ translation {fmt(cx, cy, WALL_H + 0.1)} name \"mine_ceiling\" "
            f"children [ Shape {{ appearance USE ROCK_APP geometry Box {{ size {fmt(w, h, 0.2)} }} }} ] }}"
        )
    return out


def corridor_axis(r, c):
    """'EW' or 'NS' for a straight corridor cell, else None (junction, bend, dead end)."""
    n, s, e, w = is_open(r - 1, c), is_open(r + 1, c), is_open(r, c + 1), is_open(r, c - 1)
    if e and w and not n and not s:
        return "EW"
    if n and s and not e and not w:
        return "NS"
    return None


def wall_rocks(rng):
    """Boulders half-buried in the walls so the tunnels are not perfect boxes."""
    out, i = [], 0
    for r, row in enumerate(LAYOUT):
        for c, ch in enumerate(row):
            if ch == "#":
                continue
            x, y = cell_center(r, c)
            for side, (dr, dc) in DIRS.items():
                if is_open(r + dr, c + dc) or rng.random() > 0.6:
                    continue
                along = rng.uniform(-1.0, 1.0)
                push = CELL / 2 + 0.15
                if side in "NS":
                    rx, ry = x + along, y - dr * push
                else:
                    rx, ry = x + dc * push, y + along
                scale = rng.uniform(7.5, 12)
                rz = rng.uniform(0.2, 1.3)
                rot = fmt(rng.uniform(-1, 1), rng.uniform(-1, 1), 1, rng.uniform(0, 6.28))
                shade = rng.uniform(0.45, 0.65)
                out.append(
                    f"Rock {{ translation {fmt(rx, ry, rz)} rotation {rot} name \"boulder_{i}\" "
                    f"scale {scale:.3g} color {fmt(shade, shade * 0.93, shade * 0.85)} }}"
                )
                i += 1
    return out


def timber_and_rails():
    """Timber support sets, rails and a cable along straight corridors (visual dressing)."""
    out, rails_defined, timber_defined, duct_defined = [], False, False, False
    for r, row in enumerate(LAYOUT):
        for c, ch in enumerate(row):
            axis = corridor_axis(r, c) if ch != "#" else None
            if not axis:
                continue
            x, y = cell_center(r, c)
            post_off = CELL / 2 - 0.12
            rail_app = "USE RAIL_APP" if rails_defined else "DEF RAIL_APP RustyMetal { }"
            rails_defined = True
            if (r + c) % 2 == 0:
                wood = "USE TIMBER" if timber_defined else \
                    "DEF TIMBER RoughPine { colorOverride 0.5 0.36 0.24 textureTransform TextureTransform { scale 0.3 2 } }"
                timber_defined = True
                if axis == "EW":
                    posts = [(x, y + post_off), (x, y - post_off)]
                    beam_size = fmt(0.22, CELL, 0.22)
                else:
                    posts = [(x + post_off, y), (x - post_off, y)]
                    beam_size = fmt(CELL, 0.22, 0.22)
                post = f"Box {{ size {fmt(0.22, 0.22, WALL_H - 0.3)} }}"
                for k, (px, py) in enumerate(posts):
                    out.append(
                        f"Solid {{ translation {fmt(px, py, (WALL_H - 0.3) / 2)} name \"post_{r}_{c}_{k}\" "
                        f"children [ Shape {{ appearance {wood} geometry {post} }} ] boundingObject {post} }}"
                    )
                    wood = "USE TIMBER"
                out.append(
                    f"Pose {{ translation {fmt(x, y, WALL_H - 0.4)} children [ Shape {{ appearance USE TIMBER "
                    f"geometry Box {{ size {beam_size} }} }} ] }}"
                )
            # rails (no bounding object: purely visual, the robot rolls over them)
            for off in (-0.45, 0.45):
                if axis == "EW":
                    t, size = (x, y + off, 0.02), fmt(CELL, 0.07, 0.04)
                else:
                    t, size = (x + off, y, 0.02), fmt(0.07, CELL, 0.04)
                out.append(
                    f"Pose {{ translation {fmt(*t)} children [ Shape {{ appearance {rail_app} geometry Box {{ size {size} }} }} ] }}"
                )
                rail_app = "USE RAIL_APP"
            # power cable along one wall near the roof
            if axis == "EW":
                t, rot = (x, y + CELL / 2 - 0.25, WALL_H - 0.35), "0 1 0 1.5708"
            else:
                t, rot = (x + CELL / 2 - 0.25, y, WALL_H - 0.35), "1 0 0 1.5708"
            out.append(
                f"Pose {{ translation {fmt(*t)} rotation {rot} children [ Shape {{ appearance PBRAppearance "
                f"{{ baseColor 0.05 0.05 0.05 roughness 0.6 }} geometry Cylinder {{ height {CELL} radius 0.03 }} }} ] }}"
            )
            # yellow ventilation duct along the opposite side of the roof (every mine has one)
            if axis == "EW":
                t = (x, y - CELL / 2 + 0.45, WALL_H - 0.45)
            else:
                t = (x - CELL / 2 + 0.45, y, WALL_H - 0.45)
            duct_app = "USE DUCT_APP" if duct_defined else \
                "DEF DUCT_APP PBRAppearance { baseColor 0.85 0.62 0.12 roughness 0.85 metalness 0 }"
            duct_defined = True
            out.append(
                f"Pose {{ translation {fmt(*t)} rotation {rot} children [ Shape {{ appearance {duct_app} "
                f"geometry Cylinder {{ height {CELL + 0.02} radius 0.24 subdivision 20 }} }} ] }}"
            )
    return out


def work_lights():
    """Warm caged lamps hanging from the roof, roughly every other cell."""
    out = []
    count = 0
    for r, row in enumerate(LAYOUT):
        for c, ch in enumerate(row):
            if ch == "#" or (r * 3 + c) % 4 != 0 or count >= 22:
                continue
            x, y = cell_center(r, c)
            count += 1
            out.append(
                f"PointLight {{ location {fmt(x, y, WALL_H - 0.45)} color 1 0.72 0.42 intensity 1.6 "
                f"attenuation 0 0 0.35 radius 8 castShadows FALSE }}"
            )
            out.append(
                f"Pose {{ translation {fmt(x, y, WALL_H - 0.35)} children [ Shape {{ appearance PBRAppearance "
                f"{{ baseColor 1 0.8 0.5 emissiveColor 1 0.7 0.35 emissiveIntensity 3 }} "
                f"geometry Sphere {{ radius 0.07 subdivision 2 }} }} ] }}"
            )
    return out


def base_station(start):
    r, c = start
    x, y = cell_center(r, c)
    bx, by = x - 0.9, y + 0.9  # tucked into the north-west corner of the start cell
    return [
        f"""DEF BASE_STATION Solid {{
  translation {fmt(bx, by, 0)}
  name "base_station"
  children [
    Pose {{ translation 0 0 0.9 children [ Shape {{ appearance RustyMetal {{ }} geometry Cylinder {{ height 1.8 radius 0.04 }} }} ] }}
    Pose {{ translation 0 0 0.25 children [ Shape {{ appearance PBRAppearance {{ baseColor 0.9 0.6 0.05 roughness 0.5 }} geometry Box {{ size 0.45 0.35 0.5 }} }} ] }}
    Pose {{ translation 0 0 1.85 children [ Shape {{ appearance PBRAppearance {{ baseColor 0.1 1 0.3 emissiveColor 0.1 1 0.3 emissiveIntensity 4 }} geometry Sphere {{ radius 0.06 }} }} ] }}
    PointLight {{ location 0 0 1.9 color 0.2 1 0.4 intensity 0.6 attenuation 0 0 1 radius 4 }}
  ]
  boundingObject Pose {{ translation 0 0 0.25 children [ Box {{ size 0.45 0.35 0.5 }} ] }}
}}""",
        f"ConstructionLamp {{ translation {fmt(x - 1.1, y - 1.0, 0.07)} rotation 0 0 1 0.6 spotLightIntensity 1.5 spotLightColor 1 0.9 0.75 }}",
        f"WoodenBox {{ translation {fmt(x - 1.05, y - 0.2, 0.3)} name \"supply crate\" size 0.6 0.6 0.6 }}",
    ]


COLLAPSES = [(2, 8), (9, 1)]   # dead-end cells filled with fallen rock and broken timber
MINE_CARTS = [(1, 6)]           # straight east-west corridor cells


def dressing(rng):
    """Story props: collapsed tunnel ends and an abandoned mine cart."""
    out = []
    for r, c in COLLAPSES:
        x, y = cell_center(r, c)
        for i in range(7):
            rx, ry = x + rng.uniform(-1.0, 1.0), y + rng.uniform(-1.0, 1.0)
            rot = fmt(rng.uniform(-1, 1), rng.uniform(-1, 1), 1, rng.uniform(0, 6.28))
            shade = rng.uniform(0.4, 0.55)
            out.append(f"Rock {{ translation {fmt(rx, ry, rng.uniform(0.2, 0.7))} rotation {rot} "
                       f"name \"rubble_{r}_{c}_{i}\" scale {rng.uniform(8, 12):.3g} "
                       f"color {fmt(shade, shade * 0.93, shade * 0.85)} }}")
        for i in range(2):  # snapped timbers leaning on the pile
            out.append(f"Pose {{ translation {fmt(x + rng.uniform(-0.6, 0.6), y + rng.uniform(-0.6, 0.6), 0.6)} "
                       f"rotation {fmt(rng.uniform(-1, 1), rng.uniform(-1, 1), 0.3, rng.uniform(0.6, 1.2))} "
                       "children [ Shape { appearance USE TIMBER geometry Box { size 0.2 0.2 2.2 } } ] }")
    for r, c in MINE_CARTS:
        x, y = cell_center(r, c)
        wheels = " ".join(
            f"Pose {{ translation {fmt(dx, dy, 0.14)} rotation 1 0 0 1.5708 children [ Shape {{ appearance USE RAIL_APP "
            f"geometry Cylinder {{ height 0.06 radius 0.14 }} }} ] }}"
            for dx in (-0.4, 0.4) for dy in (-0.45, 0.45))
        out.append(
            # derailed against the north wall, leaving room to drive past on the south side
            f"Solid {{ translation {fmt(x + 0.6, y + 0.85, 0)} rotation 0 0 1 0.3 name \"mine_cart\" children [ "
            "Pose { translation 0 0 0.55 children [ Shape { appearance RustyMetal { } geometry Box { size 1.2 0.85 0.65 } } ] } "
            f"{wheels} ] boundingObject Pose {{ translation 0 0 0.5 children [ Box {{ size 1.2 0.85 0.8 }} ] }} }}"
        )
    return out


SURVIVOR_EXTRAS = (
    # hard hat with a working cap lamp, so survivors read clearly in the dark
    "Pose { translation 0.02 0 0.58 children [ Shape { appearance PBRAppearance { baseColor 1 0.85 0.05 roughness 0.35 } "
    "geometry Sphere { radius 0.13 subdivision 2 } } ] } "
    "Pose { translation 0.14 0 0.6 children [ Shape { appearance PBRAppearance { baseColor 1 1 0.9 emissiveColor 1 1 0.85 "
    "emissiveIntensity 5 } geometry Cylinder { height 0.03 radius 0.03 } } ] } "
)


def glow_stick(x, y):
    """A green chemlight on the floor next to a survivor: the glow you'd spot down a tunnel."""
    return (f"Pose {{ translation {fmt(x, y, 0.03)} rotation 0 0 1 0.7 children [ Shape {{ appearance PBRAppearance "
            "{ baseColor 0.3 1 0.3 emissiveColor 0.2 1 0.2 emissiveIntensity 6 } geometry Capsule { height 0.14 radius 0.012 } } ] } "
            f"PointLight {{ location {fmt(x, y, 0.15)} color 0.3 1 0.35 intensity 0.8 attenuation 0 0 1.2 radius 4 }}")


def artifacts():
    out, truth = [], []
    counts = {}
    for r, c, cls, side in ARTIFACTS:
        x, y = cell_center(r, c)
        dr, dc = DIRS[side]
        lean = 1.05
        ax, ay = x + dc * lean, y - dr * lean
        base_cls = "survivor" if cls.startswith("survivor") else cls
        counts[base_cls] = counts.get(base_cls, 0) + 1
        name = f"{base_cls}_{counts[base_cls]}"
        color = RECOGNITION_COLORS[base_cls]
        yaw = {"N": -1.5708, "S": 1.5708, "E": 3.1416, "W": 0.0}[side]  # face into the tunnel
        if cls in ("survivor", "survivor_lying"):
            if cls == "survivor":
                pose = f"translation {fmt(ax, ay, 1.27)} rotation 0 0 1 {yaw}"
            else:
                pose = f"translation {fmt(ax, ay, 0.2)} rotation 0 1 0 1.5708"
            body = (f"Pedestrian {{ {pose} name \"{name}\" "
                    "controller \"<none>\" shirtColor 0.85 0.4 0.05 pantsColor 0.15 0.15 0.2 enableBoundingObject TRUE "
                    f"bodySlot [ Solid {{ name \"{name}_tag\" model \"survivor\" recognitionColors [ {color} ] "
                    "children [ Pose { translation 0 0 0.1 children [ Shape { appearance PBRAppearance "
                    "{ baseColor 1 0.45 0 emissiveColor 0.4 0.15 0 roughness 0.4 } geometry Box { size 0.26 0.4 0.45 } } ] } "
                    f"{SURVIVOR_EXTRAS}] }} ] }}")
            # glow stick on the tunnel side of the survivor
            body += "\n" + glow_stick(ax - dc * 0.6 + 0.3, ay + dr * 0.6)
        else:
            inner = {
                "fire_extinguisher": "FireExtinguisher { enablePhysics FALSE }",
                "equipment_case": "MetalStorageBox { }",
                "gas_hazard": "OilBarrel { translation 0 0 0.44 }",
            }[cls]
            body = (f"Solid {{ translation {fmt(ax, ay, 0)} rotation 0 0 1 {yaw} name \"{name}\" model \"{cls}\" "
                    f"recognitionColors [ {color} ] children [ {inner} ] }}")
        out.append(body)
        truth.append({"name": name, "class": base_cls, "x": round(ax, 3), "y": round(ay, 3), "cell": [r, c]})
    return out, truth


def robot(start, auto=False, nowindow=False, extra=(), yaw=0.0):
    x, y = cell_center(*start)
    flags = [f for f, on in (("auto", auto), ("nowindow", nowindow)) if on] + list(extra)
    args = "controllerArgs [ " + " ".join(f'"{f}"' for f in flags) + " ]" if flags else ""
    return f"""DEF EXPLORER SummitXlSteel {{
  translation {fmt(x, y, 0.118)}
  rotation 0 0 1 {yaw:.4f}
  name "explorer"
  controller "mine_explorer"
  {args}
  bodySlot [
    Pose {{ translation 0 0 0.12 children [ Shape {{ appearance DEF MAST_APP PBRAppearance {{ baseColor 0.12 0.12 0.13 roughness 0.4 metalness 0.6 }} geometry Cylinder {{ height 0.24 radius 0.035 }} }} ] }}
    Lidar {{
      translation 0 0 0.27
      name "lidar"
      horizontalResolution 360
      fieldOfView 6.2832
      numberOfLayers 5
      verticalFieldOfView 0.6
      minRange 0.08
      maxRange 15
      noise 0.003
      children [ Shape {{ appearance USE MAST_APP geometry Cylinder {{ height 0.08 radius 0.055 }} }} ]
    }}
    Pose {{ translation 0 0 0.35 children [ Shape {{ appearance PBRAppearance {{ baseColor 1 0.5 0 emissiveColor 1 0.45 0 emissiveIntensity 3 }} geometry Cylinder {{ height 0.05 radius 0.03 }} }} ] }}
    Display {{
      name "map_display"
      width 480
      height 480
    }}
    Camera {{
      translation 0.34 0 0.06
      name "artifact_camera"
      fieldOfView 1.2
      width 640
      height 360
      exposure 1.6
      recognition Recognition {{
        maxRange 14
        maxObjects 10
        frameColor 1 0.55 0
        frameThickness 2
      }}
    }}
    SpotLight {{ location 0.36 0.2 0.02 direction 1 0 -0.12 color 1 0.95 0.85 intensity 4 beamWidth 0.45 cutOffAngle 0.75 attenuation 0 0 0.15 radius 30 castShadows TRUE }}
    SpotLight {{ location 0.36 -0.2 0.02 direction 1 0 -0.12 color 1 0.95 0.85 intensity 4 beamWidth 0.45 cutOffAngle 0.75 attenuation 0 0 0.15 radius 30 }}
    Pose {{ translation 0.37 0.2 0.02 children [ DEF HEADLAMP Shape {{ appearance PBRAppearance {{ baseColor 1 1 0.95 emissiveColor 1 1 0.9 emissiveIntensity 4 }} geometry Box {{ size 0.02 0.09 0.04 }} }} ] }}
    Pose {{ translation 0.37 -0.2 0.02 children [ USE HEADLAMP ] }}
  ]
}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="leave the ceiling off for overview shots")
    ap.add_argument("--auto", action="store_true", help="robot starts in autonomous mode")
    ap.add_argument("--nowindow", action="store_true", help="don't open the separate Live map window")
    ap.add_argument("--cinecam", default="", help="add the film camera, with these controller args")
    ap.add_argument("--seed", type=int, default=0, help="Webots random seed (sensor noise differs per seed)")
    ap.add_argument("--args", default="", help="extra controller arguments, space separated")
    ap.add_argument("--out", default="mine.wbt", help="world file name inside worlds/")
    args = ap.parse_args()

    rng = random.Random(SEED)
    start, n_open = check_connected()
    art_nodes, truth = artifacts()

    lines = header()
    lines.append(world_settings(cell_center(*start), args.open).replace(
        "basicTimeStep 32", f"basicTimeStep 32\n  randomSeed {args.seed}"))
    lines += rock_walls()
    lines += floor_and_ceiling(args.open)
    lines += wall_rocks(rng)
    lines += timber_and_rails()
    lines += dressing(rng)
    lines += work_lights()
    lines += base_station(start)
    lines += art_nodes
    # Trials differ by seed: the robot starts facing a different way (east, south, west, north) and
    # the artifact sensor noise is drawn from a different random stream.
    yaw = -(args.seed % 4) * math.pi / 2
    lines.append(robot(start, args.auto, args.nowindow, args.args.split() + [f"seed={args.seed}"], yaw))

    if args.cinecam:
        cargs = " ".join(f'"{a}"' for a in args.cinecam.split())
        lines.append(f"""DEF CINECAM Robot {{
  translation 22.5 -16.5 30
  name "cinecam"
  controller "cinecam"
  controllerArgs [ {cargs} ]
  supervisor TRUE
  children [
    Camera {{
      name "cine"
      fieldOfView 1.05
      width 1280
      height 720
      exposure 1.8
      antiAliasing TRUE
      far 120
    }}
  ]
}}""")

    world = ROOT / "worlds" / args.out
    world.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bx, by = cell_center(*start)
    gt = {
        "cell_size": CELL,
        "layout": LAYOUT,
        "open_area_m2": n_open * CELL * CELL,
        "start": {"cell": list(start), "x": bx, "y": by, "yaw": 0.0},
        "base_station": {"x": bx - 0.9, "y": by + 0.9},
        "artifacts": truth,
        "scoring_radius_m": 5.0,
    }
    (ROOT / "config" / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"Wrote {world.relative_to(ROOT)}: {n_open} tunnel cells ({n_open * CELL * CELL:.0f} m2), "
          f"{len(truth)} artifacts")


if __name__ == "__main__":
    main()

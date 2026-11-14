"""Artifact detection -> localization -> de-duplication -> report, plus SubT-style scoring.

The camera's recognition gives, for each object in view, its class and where it is
relative to the camera. This module:
  1. converts that to a position on the map using the robot's pose,
  2. merges repeated sightings of the same object into one "track" (running average),
  3. confirms a track once it has been seen enough times, which produces a report,
  4. scores reports against ground truth: a report counts only if an object of the
     same class really is within 5 m (the DARPA SubT rule), each real object once.

To keep the task honest, sensor noise is added to every sighting (range and bearing),
so a single sighting is NOT accurate and the averaging in step 2 actually matters.

Pure Python / numpy, no simulator imports.
"""

import math
import random

MERGE_RADIUS_M = 2.5       # sightings of the same class closer than this are the same object
CONFIRM_SIGHTINGS = 6      # sightings needed before a track is reported
MAX_DETECT_RANGE_M = 10.0  # ignore detections further than this (too uncertain)
RANGE_NOISE = 0.06         # std dev, as a fraction of range
BEARING_NOISE = math.radians(2.0)


class Track:
    def __init__(self, tid, cls, x, y, t):
        self.id, self.cls = tid, cls
        self.x, self.y = x, y
        self.n = 1
        self.first_seen = self.last_seen = t
        self.confirmed_at = None
        self.reported = False

    def add(self, x, y, t):
        self.n += 1
        self.x += (x - self.x) / self.n     # running mean: noise shrinks as 1/sqrt(n)
        self.y += (y - self.y) / self.n
        self.last_seen = t

    def as_report(self):
        return {"id": self.id, "class": self.cls, "x": round(self.x, 2), "y": round(self.y, 2),
                "sightings": self.n, "first_seen_s": round(self.first_seen, 1),
                "confirmed_s": round(self.confirmed_at, 1) if self.confirmed_at is not None else None}


class ArtifactTracker:
    def __init__(self, camera_offset=(0.34, 0.0), seed=1):
        self.camera_offset = camera_offset
        self.tracks = []
        self._rng = random.Random(seed)

    def observe(self, pose, detections, t):
        """detections: [{'class': str, 'rel': (forward, left, up)}] in the camera frame.

        Returns the list of tracks newly confirmed by this call (i.e. new reports).
        """
        x, y, yaw = pose
        cx = x + math.cos(yaw) * self.camera_offset[0] - math.sin(yaw) * self.camera_offset[1]
        cy = y + math.sin(yaw) * self.camera_offset[0] + math.cos(yaw) * self.camera_offset[1]
        new_reports = []
        for det in detections:
            fwd, left = det["rel"][0], det["rel"][1]
            rng = math.hypot(fwd, left)
            if rng > MAX_DETECT_RANGE_M or rng < 0.3:
                continue
            rng *= 1.0 + self._rng.gauss(0.0, RANGE_NOISE)
            bearing = math.atan2(left, fwd) + self._rng.gauss(0.0, BEARING_NOISE)
            ax = cx + rng * math.cos(yaw + bearing)
            ay = cy + rng * math.sin(yaw + bearing)

            same = [tr for tr in self.tracks if tr.cls == det["class"]
                    and math.hypot(tr.x - ax, tr.y - ay) < MERGE_RADIUS_M]
            if same:
                tr = min(same, key=lambda k: math.hypot(k.x - ax, k.y - ay))
                tr.add(ax, ay, t)
            else:
                tr = Track(len(self.tracks) + 1, det["class"], ax, ay, t)
                self.tracks.append(tr)
            if tr.confirmed_at is None and tr.n >= CONFIRM_SIGHTINGS:
                tr.confirmed_at = t
                new_reports.append(tr)
        return new_reports

    def confirmed(self):
        return [tr for tr in self.tracks if tr.confirmed_at is not None]

    def reports(self):
        return [tr.as_report() for tr in self.confirmed()]


def score(reports, truth, radius=5.0):
    """SubT scoring. Each true artifact can be claimed by at most one report of the same class.

    Returns a dict with points, per-report results and what was missed.
    """
    pairs = []
    for i, rep in enumerate(reports):
        for j, art in enumerate(truth):
            if rep["class"] == art["class"]:
                d = math.hypot(rep["x"] - art["x"], rep["y"] - art["y"])
                if d <= radius:
                    pairs.append((d, i, j))
    pairs.sort()
    used_r, used_t, matches = set(), set(), []
    for d, i, j in pairs:
        if i in used_r or j in used_t:
            continue
        used_r.add(i)
        used_t.add(j)
        matches.append({"report_id": reports[i]["id"], "artifact": truth[j]["name"],
                        "class": truth[j]["class"], "error_m": round(d, 2)})
    errors = [m["error_m"] for m in matches]
    return {
        "points": len(matches),
        "possible": len(truth),
        "mean_error_m": round(sum(errors) / len(errors), 2) if errors else None,
        "max_error_m": max(errors) if errors else None,
        "matches": matches,
        "false_reports": [reports[i]["id"] for i in range(len(reports)) if i not in used_r],
        "missed": [truth[j]["name"] for j in range(len(truth)) if j not in used_t],
    }

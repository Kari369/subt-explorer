"""Offline test for core/comms.py and core/supervisor.py.

Run:  python tests/test_comms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.comms import CommsModel  # noqa: E402
from core.supervisor import MAX_HELD_REPORTS, Supervisor  # noqa: E402
from core.truth import load_ground_truth  # noqa: E402


def make(policy):
    gt = load_ground_truth()
    b = gt["base_station"]
    return Supervisor(CommsModel((b["x"], b["y"]), gt["layout"], gt["cell_size"]), policy)


def rep(i):
    return {"id": i, "class": "survivor", "x": 30.0 + i, "y": -20.0}


def run():
    near, far = (6.0, -4.5, 0.0), (40.5, -28.5, 0.0)   # next to base / far south-east corner

    sup = make("report_first")
    assert sup.comms.effective_distance(*near[:2]) < 5 < 40 < sup.comms.effective_distance(*far[:2])

    # in contact: a report is delivered immediately
    sup.update(0.0, near, True)
    sup.report_artifact(rep(1), 1.0)
    sup.update(1.2, near, True)
    assert len(sup.inbox) == 1 and sup.inbox[0]["delivered_s"] - sup.inbox[0]["created_s"] < 1.0

    # go deep: the link drops, a LINK_LOST alert is raised but cannot be delivered yet
    ev = sup.update(60.0, far, True)
    assert not sup.link_up and not ev, "a dropout should not alert immediately"
    ev = sup.update(66.0, far, True)
    assert any("LOST" in e for e in ev)
    assert [m["payload"].get("code") for m in sup.outbox] == ["LINK_LOST"]

    # reports found out of contact are held; holding enough of them triggers a return to comms
    for i in range(MAX_HELD_REPORTS):
        assert sup.detour_target() is None
        sup.report_artifact(rep(10 + i), 70.0 + i)
        sup.update(71.0 + i, far, True)
    assert sup.detour_target() == near[:2], "should head for the last place it had signal"

    # the breadcrumb must be far enough inside coverage that arriving there really restores the link
    probe = make("report_first")
    probe.comms.up = False
    assert probe.comms.update(*sup.detour_target()), "breadcrumb is too close to the edge of coverage"

    # back in range: everything is delivered, late, and exploration resumes
    ev = sup.update(200.0, near, True)
    assert sup.link_up and not sup.outbox and sup.detour_target() is None
    st = sup.stats()
    assert st["artifact_reports_delivered"] == 1 + MAX_HELD_REPORTS
    assert st["reports_delayed_over_5s"] == MAX_HELD_REPORTS and st["max_report_delay_s"] > 100
    assert st["alerts"] == ["LINK_LOST", "LINK_RESTORED"] and st["returns_to_comms"] == 1

    # explore_first never turns back
    sup2 = make("explore_first")
    sup2.update(0.0, near, True)
    sup2.update(10.0, far, True)
    for i in range(5):
        sup2.report_artifact(rep(i), 20.0 + i)
        sup2.update(21.0 + i, far, True)
    assert sup2.detour_target() is None and len(sup2.outbox) == 6

    print(f"OK: link drops deep in the mine, {MAX_HELD_REPORTS} held reports trigger a return to comms, "
          f"late reports delivered with max delay {st['max_report_delay_s']} s")


if __name__ == "__main__":
    run()

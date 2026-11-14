"""Mission supervisor: what the robot tells the human, and what it does when it can't.

Everything the robot wants the operator to know becomes a message: an artifact
report, or an alert the robot raises on its own initiative (link lost, stuck, low
battery, exploration complete...). Messages go into an outbox. While the radio link
is up the outbox empties straight to the base station; while it is down, messages
wait, and the delay each one suffers is recorded.

The supervisor also decides how the robot behaves out of contact. Policies:
  "report_first"  : if it is holding undelivered artifact reports for too long (or too
                    many), it drives back to the last place it had signal, delivers
                    them, then carries on exploring. This is the "return to comms"
                    behaviour SubT teams used.
  "explore_first" : it never turns back; everything is delivered when it finally
                    returns to base.

Pure Python, no simulator imports.
"""

import math

ARTIFACT, ALERT = "ARTIFACT", "ALERT"

MAX_HELD_REPORTS = 2          # report_first: turn back when holding this many undelivered reports...
MAX_HOLD_S = 90.0             # ...or when the oldest has waited this long
BATTERY_LIFE_S = 25 * 60.0    # full charge lasts this long while moving
BATTERY_IDLE_FACTOR = 0.3     # standing still drains this fraction as fast
LOW_BATTERY = 0.25
LINK_ALERT_AFTER_S = 5.0      # ignore dropouts shorter than this
GOOD_SIGNAL = 0.2            # signal quality (0..1) that counts as safely inside coverage


class Supervisor:
    def __init__(self, comms, policy="report_first"):
        self.comms = comms
        self.policy = policy
        self.outbox = []            # messages waiting for a link
        self.inbox = []             # messages the base station has received
        self.messages = []          # every message ever created (for the log)
        self.link_up = True
        self.last_link_pose = None  # last place the robot had signal (the "breadcrumb")
        self.battery = 1.0
        self.low_battery_sent = False
        self.returning_to_comms = False
        self.comms_returns = 0
        self.link_losses = 0
        self.time_out_of_contact = 0.0
        self._down_since, self._down_pose, self._lost_alerted = 0.0, (0.0, 0.0), False
        self._last_t = None

    # ---- creating messages ---------------------------------------------------
    def _send(self, kind, t, text, payload=None):
        msg = {"n": len(self.messages) + 1, "kind": kind, "text": text, "payload": payload or {},
               "created_s": round(t, 1), "delivered_s": None}
        self.messages.append(msg)
        self.outbox.append(msg)
        return msg

    def report_artifact(self, report, t):
        return self._send(ARTIFACT, t, f"{report['class']} at ({report['x']:.1f}, {report['y']:.1f})", report)

    def alert(self, code, t, text):
        return self._send(ALERT, t, f"{code}: {text}", {"code": code})

    # ---- the periodic update -------------------------------------------------
    def update(self, t, pose, moving):
        """Call a few times a second. Returns a list of human-readable event strings."""
        events = []
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        was_up = self.link_up
        self.link_up = self.comms.update(pose[0], pose[1])
        # Breadcrumb: the last place with a comfortably strong signal (like reading a radio's
        # signal bars). Aiming for the very edge of coverage would leave the robot parked
        # just outside it, so the edge itself is not good enough.
        if self.link_up and (self.last_link_pose is None or self.comms.signal(pose[0], pose[1]) >= GOOD_SIGNAL):
            self.last_link_pose = (pose[0], pose[1])
        # A brief dropout at the edge of coverage is not worth the operator's attention:
        # the LINK_LOST alert is only raised once the link has been down for a few seconds.
        if was_up and not self.link_up:
            self._down_since, self._down_pose, self._lost_alerted = t, (pose[0], pose[1]), False
        if not self.link_up and not self._lost_alerted and t - self._down_since >= LINK_ALERT_AFTER_S:
            self._lost_alerted = True
            self.link_losses += 1
            self._send(ALERT, self._down_since,
                       f"LINK_LOST: lost contact at ({self._down_pose[0]:.1f}, {self._down_pose[1]:.1f}); "
                       f"continuing autonomously", {"code": "LINK_LOST"})
            events.append("link LOST: base station can no longer see the robot")
        if not was_up and self.link_up and self._lost_alerted:
            self.alert("LINK_RESTORED", t, f"back in contact at ({pose[0]:.1f}, {pose[1]:.1f}) "
                                           f"after {t - self._down_since:.0f} s")
            events.append("link RESTORED")
        if not self.link_up:
            self.time_out_of_contact += dt

        self.battery = max(0.0, self.battery - dt / BATTERY_LIFE_S * (1.0 if moving else BATTERY_IDLE_FACTOR))
        if self.battery < LOW_BATTERY and not self.low_battery_sent:
            self.low_battery_sent = True
            self.alert("LOW_BATTERY", t, f"{100 * self.battery:.0f}% left; abandoning exploration and returning to base")
            events.append("LOW BATTERY: returning to base")

        if self.link_up and self.outbox:
            for msg in self.outbox:
                msg["delivered_s"] = round(t, 1)
                self.inbox.append(msg)
                delay = msg["delivered_s"] - msg["created_s"]
                late = f" (delayed {delay:.0f} s)" if delay > 1.0 else ""
                events.append(f"delivered to base: {msg['text']}{late}")
            self.outbox = []

        # out-of-contact behaviour
        held = [m for m in self.outbox if m["kind"] == ARTIFACT]
        if self.link_up:
            if self.returning_to_comms:
                events.append("reports delivered; resuming exploration")
            self.returning_to_comms = False
        elif self.policy == "report_first" and held and not self.returning_to_comms:
            oldest = t - min(m["created_s"] for m in held)
            if len(held) >= MAX_HELD_REPORTS or oldest >= MAX_HOLD_S:
                self.returning_to_comms = True
                self.comms_returns += 1
                events.append(f"holding {len(held)} undelivered report(s): returning to comms range to deliver")
        return events

    # ---- what the navigation layer should do ---------------------------------
    def detour_target(self):
        """Where to drive to regain contact, or None to carry on."""
        return self.last_link_pose if self.returning_to_comms else None

    def must_return_home(self):
        return self.low_battery_sent

    # ---- summaries -----------------------------------------------------------
    def delivered_artifacts(self):
        return [m for m in self.inbox if m["kind"] == ARTIFACT]

    def stats(self):
        arts = [m for m in self.messages if m["kind"] == ARTIFACT]
        delays = [m["delivered_s"] - m["created_s"] for m in arts if m["delivered_s"] is not None]
        return {
            "policy": self.policy,
            "artifact_reports_created": len(arts),
            "artifact_reports_delivered": len(delays),
            "reports_delayed_over_5s": sum(d > 5.0 for d in delays),
            "mean_report_delay_s": round(sum(delays) / len(delays), 1) if delays else None,
            "max_report_delay_s": round(max(delays), 1) if delays else None,
            "returns_to_comms": self.comms_returns,
            "link_losses": self.link_losses,
            "time_out_of_contact_s": round(self.time_out_of_contact, 1),
            "alerts": [m["payload"]["code"] for m in self.messages if m["kind"] == ALERT],
            "battery_left_pct": round(100 * self.battery),
        }

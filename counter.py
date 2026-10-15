"""
counter.py
----------
Line-crossing logic. Pure Python (no OpenCV / no YOLO) so it is easy to test.

A vehicle is counted ONCE, at the moment its tracked centre point crosses the
virtual line. Its class is decided by majority vote over all frames it was
seen in, which removes most "car <-> truck" flicker.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class CrossingEvent:
    track_id: int
    vehicle_class: str
    direction: str      # "IN" or "OUT"
    frame: int
    time_sec: float


class LineCounter:
    def __init__(self, frame_w, frame_h, orientation="horizontal",
                 position=0.55, invert_direction=False):
        if orientation not in ("horizontal", "vertical"):
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        self.orientation = orientation
        self.invert = invert_direction

        if orientation == "horizontal":
            self.line_pos = int(frame_h * position)
            self.p1, self.p2 = (0, self.line_pos), (frame_w, self.line_pos)
        else:
            self.line_pos = int(frame_w * position)
            self.p1, self.p2 = (self.line_pos, 0), (self.line_pos, frame_h)

        self._last_axis = {}                       # track_id -> last coordinate
        self._class_votes = defaultdict(Counter)   # track_id -> Counter(class)
        self._counted = set()

        self.count_in = 0
        self.count_out = 0
        self.by_class = Counter()
        self.events = []

    # -- helpers ---------------------------------------------------------
    def centroid(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _axis(self, cx, cy):
        return cy if self.orientation == "horizontal" else cx

    # -- main entry ------------------------------------------------------
    def update(self, track_id, box, class_name, frame_idx=0, time_sec=0.0):
        """Feed one tracked box. Returns the centroid (cx, cy)."""
        cx, cy = self.centroid(box)
        axis = self._axis(cx, cy)
        self._class_votes[track_id][class_name] += 1

        prev = self._last_axis.get(track_id)
        self._last_axis[track_id] = axis

        if prev is None or track_id in self._counted:
            return cx, cy

        forward = prev < self.line_pos <= axis
        backward = prev > self.line_pos >= axis
        if not (forward or backward):
            return cx, cy

        moving_forward = forward != self.invert   # XOR with invert flag
        direction = "IN" if moving_forward else "OUT"
        final_class = self._class_votes[track_id].most_common(1)[0][0]

        if direction == "IN":
            self.count_in += 1
        else:
            self.count_out += 1
        self.by_class[final_class] += 1
        self._counted.add(track_id)
        self.events.append(
            CrossingEvent(track_id, final_class, direction, frame_idx, round(time_sec, 2))
        )
        return cx, cy

    @property
    def total(self):
        return self.count_in + self.count_out

    def summary(self):
        return {
            "total": self.total,
            "in": self.count_in,
            "out": self.count_out,
            "by_class": dict(self.by_class),
        }

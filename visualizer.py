"""
visualizer.py
-------------
All OpenCV drawing code: boxes, labels, trails, counting line, dashboard.
Sizes scale with the frame height so the overlay looks good on any resolution.
"""

from collections import defaultdict, deque

import cv2

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)


class Visualizer:
    def __init__(self, frame_h, show_trails=True, trail_length=30):
        self.scale = max(frame_h / 720.0, 0.6)
        self.show_trails = show_trails
        self.trails = defaultdict(lambda: deque(maxlen=trail_length))

    # -- per-vehicle -----------------------------------------------------
    def draw_vehicle(self, frame, box, track_id, class_name, centroid):
        color = config.CLASS_COLORS.get(class_name, WHITE)
        x1, y1, x2, y2 = map(int, box)
        thick = max(1, int(2 * self.scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        label = f"{class_name} | ID {track_id}"
        fs = 0.5 * self.scale
        (tw, th), _ = cv2.getTextSize(label, FONT, fs, thick)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 4, top + th + 2), FONT, fs, WHITE, thick, cv2.LINE_AA)

        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(frame, (cx, cy), max(3, int(4 * self.scale)), color, -1)

        if self.show_trails:
            self.trails[track_id].append((cx, cy))
            pts = list(self.trails[track_id])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(2 * self.scale)))

    # -- overlays --------------------------------------------------------
    def draw_line(self, frame, counter):
        cv2.line(frame, counter.p1, counter.p2, config.LINE_COLOR, max(2, int(3 * self.scale)))

    def draw_dashboard(self, frame, counter, live_count):
        s = self.scale
        rows = [(f"{name}s: {counter.by_class.get(name, 0)}",
                 config.CLASS_COLORS.get(name, WHITE))
                for name in ("Car", "Bus", "Truck", "Motorcycle")]
        rows.append((f"Total: {counter.total}", WHITE))
        rows.append((f"IN: {counter.count_in}   OUT: {counter.count_out}", config.LINE_COLOR))
        rows.append((f"In view now: {live_count}", (200, 200, 200)))

        line_h = int(30 * s)
        panel_w, panel_h = int(290 * s), int(line_h * len(rows) + 16 * s)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = int(line_h * 0.85)
        for text, color in rows:
            cv2.putText(frame, text, (int(12 * s), y), FONT, 0.65 * s, color,
                        max(1, int(2 * s)), cv2.LINE_AA)
            y += line_h

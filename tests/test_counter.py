import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from counter import LineCounter  # noqa: E402


def box(cx, cy, w=20, h=20):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class LineCounterTests(unittest.TestCase):
    def setUp(self):
        self.c = LineCounter(1000, 1000, "horizontal", 0.5)   # line at y=500

    def test_counts_in_once(self):
        for y in (400, 450, 490, 510, 560, 600):
            self.c.update(1, box(100, y), "Car")
        self.assertEqual((self.c.count_in, self.c.count_out, self.c.total), (1, 0, 1))

    def test_counts_out(self):
        for y in (600, 550, 510, 490, 450):
            self.c.update(2, box(100, y), "Bus")
        self.assertEqual((self.c.count_in, self.c.count_out), (0, 1))
        self.assertEqual(self.c.by_class["Bus"], 1)

    def test_no_double_count_when_jittering(self):
        for y in (480, 520, 490, 530, 495, 505, 470):
            self.c.update(3, box(100, y), "Car")
        self.assertEqual(self.c.total, 1)

    def test_vehicle_that_never_crosses(self):
        for y in (100, 150, 200, 300):
            self.c.update(4, box(100, y), "Car")
        self.assertEqual(self.c.total, 0)

    def test_class_majority_vote(self):
        for y, cls in ((400, "Truck"), (450, "Car"), (480, "Car"), (520, "Car")):
            self.c.update(5, box(100, y), cls)
        self.assertEqual(self.c.by_class["Car"], 1)
        self.assertEqual(self.c.by_class["Truck"], 0)

    def test_vertical_line_and_invert(self):
        c = LineCounter(1000, 1000, "vertical", 0.5, invert_direction=True)
        for x in (400, 480, 520, 600):
            c.update(6, box(x, 300), "Motorcycle")
        self.assertEqual((c.count_in, c.count_out), (0, 1))   # inverted naming

    def test_events_recorded(self):
        for i, y in enumerate((450, 480, 520)):
            self.c.update(7, box(100, y), "Car", frame_idx=i, time_sec=i * 0.5)
        self.assertEqual(len(self.c.events), 1)
        self.assertEqual(self.c.events[0].direction, "IN")


if __name__ == "__main__":
    unittest.main()

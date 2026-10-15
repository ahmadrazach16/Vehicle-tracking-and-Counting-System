"""
config.py
---------
Central configuration for the Vehicle Tracking & Counting System.
Command-line arguments in main.py override these defaults.
"""

# ---------------------------------------------------------------------------
# Model / detection
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = "yolo11n.pt"      # nano: fastest. Try yolo11s.pt / yolo11m.pt for accuracy
CONFIDENCE = 0.25                 # lower = more detections (good for dense traffic)
IOU = 0.5
IMG_SIZE = 640
VID_STRIDE = 1                    # process every Nth frame (2 = twice as fast)

# Fast preset used by:  python main.py --fast   (great for CPU-only laptops)
FAST_IMG_SIZE = 480
FAST_VID_STRIDE = 2

# COCO class ids -> display names
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
TRACKER = "botsort.yaml"          # or "bytetrack.yaml"

# ---------------------------------------------------------------------------
# Counting line
# ---------------------------------------------------------------------------
LINE_ORIENTATION = "horizontal"   # "horizontal" or "vertical"
LINE_POSITION = 0.55              # fraction of frame height (or width if vertical)
# A vehicle moving toward larger y (down) / larger x (right) counts as "IN".
# Set True if you want the opposite naming for your camera.
INVERT_DIRECTION = False

# ---------------------------------------------------------------------------
# Drawing (BGR colors)
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "Car": (255, 128, 0),
    "Motorcycle": (0, 215, 255),
    "Bus": (0, 200, 0),
    "Truck": (60, 60, 255),
}
LINE_COLOR = (0, 165, 255)
SHOW_TRAILS = False
TRAIL_LENGTH = 30

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_DIR = "input"
OUTPUT_DIR = "output"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
SAMPLE_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
)

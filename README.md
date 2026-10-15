# 🚗 Vehicle Detection, Tracking & Counting System

Real-time traffic analytics with **YOLO11** + **BoT-SORT**: detect vehicles, give each one a persistent ID, and count it as it crosses a virtual line — with IN/OUT directions, per-class totals, movement trails and CSV/JSON reports.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![YOLO11](https://img.shields.io/badge/Detection-YOLO11n-orange)
![BoT-SORT](https://img.shields.io/badge/Tracking-BoT--SORT-green)
![OpenCV](https://img.shields.io/badge/Video-OpenCV-red)

## ✨ Features

- Detects **cars, buses, trucks and motorcycles**
- **Multi-object tracking** (BoT-SORT) with unique, persistent IDs
- **Virtual counting line** (horizontal or vertical) with **IN / OUT** direction
- **Class-specific counts** using majority voting per track (no car↔truck flicker)
- Movement **trails**, live dashboard and "vehicles in view now"
- Exports **annotated video + CSV (one row per vehicle) + JSON summary**
- Progress bar with **ETA**, safe **Ctrl+C** (partial results still saved)
- `--fast` mode for CPU-only laptops, automatic GPU use when available
- Unit-tested counting logic

## 📁 Project structure

```
vehicle-tracking-system/
├── main.py              # CLI + full pipeline
├── counter.py           # line-crossing logic (pure Python, unit-tested)
├── visualizer.py        # boxes, trails, line, dashboard
├── config.py            # every tunable setting
├── download_sample.py   # grabs a free test video
├── tests/test_counter.py
├── requirements.txt
├── input/               # put your videos here
└── output/              # annotated video, CSV, JSON
```

## 🚀 Quick start

```bash
pip install -r requirements.txt
python download_sample.py        # optional: free test video -> input/traffic.mp4
python main.py                   # auto-uses the first video in input/
```

Slow computer? `python main.py --fast`

## ⚙️ Useful options

| Option | Meaning |
|---|---|
| `--source PATH` | video file (quotes if it has spaces), webcam `0`, or RTSP URL |
| `--fast` | smaller image size + skip every 2nd frame (≈4× faster on CPU) |
| `--line 0.4` | move the counting line (fraction of frame) |
| `--orientation vertical` | vertical counting line |
| `--invert` | swap IN / OUT naming |
| `--conf 0.2` | detection confidence |
| `--weights yolo11s.pt` | bigger, more accurate model |
| `--max-frames 100` | quick test run |
| `--show` | live preview |

## 🎥 Best camera angle

Use footage from a **fixed, elevated camera (CCTV / overpass style)**. Models trained on COCO, like YOLO11, struggle with pure top-down drone footage because cars look like small rectangles from directly above.

## 🧠 How it works

1. **Detect** – YOLO11n finds vehicles in every frame.
2. **Track** – BoT-SORT links detections across frames → unique IDs.
3. **Count** – when a track's centre point crosses the line it is counted once; its class is the majority vote of all its frames.
4. **Report** – annotated video, `*_events.csv`, `*_summary.json`.

## 🔮 Ideas to extend

Speed estimation · multi-zone counting · Streamlit dashboard · heatmaps · custom-trained model for aerial footage.

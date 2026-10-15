"""
Vehicle Tracking & Counting System
===================================
YOLO11 detection + BoT-SORT tracking + virtual-line counting.

Quick start:
    python download_sample.py      # optional: get a test video
    python main.py                 # auto-picks the first video in input/
    python main.py --fast          # ~4x faster on CPU-only machines
    python main.py --source "my video.mp4" --show

Outputs (in output/):
    <name>_annotated.mp4   annotated video
    <name>_events.csv      one row per counted vehicle
    <name>_summary.json    final totals
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

import config
from counter import LineCounter
from visualizer import Visualizer


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Vehicle Tracking & Counting System")
    p.add_argument("--source", default=None,
                   help="video path, webcam index (0) or stream URL. Default: first video in input/")
    p.add_argument("--output", default=None, help="output video path")
    p.add_argument("--weights", default=config.MODEL_WEIGHTS)
    p.add_argument("--conf", type=float, default=config.CONFIDENCE)
    p.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    p.add_argument("--stride", type=int, default=config.VID_STRIDE,
                   help="process every Nth frame (2 = 2x faster)")
    p.add_argument("--fast", action="store_true",
                   help=f"preset for slow PCs: imgsz={config.FAST_IMG_SIZE}, stride={config.FAST_VID_STRIDE}")
    p.add_argument("--width", type=int, default=1280,
                   help="resize output video to this width (0 = keep original). Big speed-up for 4K videos")
    p.add_argument("--line", type=float, default=config.LINE_POSITION,
                   help="line position as a fraction of the frame (0-1)")
    p.add_argument("--orientation", choices=["horizontal", "vertical"],
                   default=config.LINE_ORIENTATION)
    p.add_argument("--invert", action="store_true", default=config.INVERT_DIRECTION,
                   help="swap IN / OUT naming")
    p.add_argument("--no-trails", action="store_true", help="hide movement trails")
    p.add_argument("--device", default=None, help="'cpu', '0' (first GPU) ... default: auto")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N processed frames (quick test)")
    p.add_argument("--show", action="store_true", help="live preview window (press q to quit)")
    return p.parse_args()


def find_default_video():
    folder = Path(config.INPUT_DIR)
    videos = sorted(p for p in folder.glob("*")
                    if p.suffix.lower() in config.VIDEO_EXTENSIONS) if folder.exists() else []
    return str(videos[0]) if videos else None


def resolve_source(arg):
    if arg is None:
        found = find_default_video()
        if not found:
            sys.exit("[ERROR] No video found.\n"
                     f"  -> Put a video inside the '{config.INPUT_DIR}/' folder, or run:\n"
                     "       python download_sample.py\n"
                     "  -> or pass one yourself:  python main.py --source \"path/to/video.mp4\"")
        return found
    if arg.isdigit():
        return int(arg)
    if not arg.startswith(("rtsp://", "http://", "https://")) and not Path(arg).exists():
        sys.exit(f"[ERROR] Video file not found: {arg}\n"
                 f"        Current folder: {Path.cwd()}\n"
                 "        Tip: wrap paths that contain spaces in quotes.")
    return arg


def pick_device(user_choice):
    if user_choice:
        return user_choice
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def fmt_time(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.fast:
        args.imgsz, args.stride = config.FAST_IMG_SIZE, config.FAST_VID_STRIDE
    args.stride = max(1, args.stride)

    source = resolve_source(args.source)
    stem = Path(source).stem if isinstance(source, str) and Path(source).exists() else "stream"
    out_video = args.output or str(Path(config.OUTPUT_DIR) / f"{stem}_annotated.mp4")
    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    base = Path(out_video).with_suffix("")

    # ---- video metadata ---------------------------------------------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] OpenCV could not open the video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    total_steps = total_src_frames // args.stride if total_src_frames > 0 else 0

    # Work on a smaller canvas for big videos (4K drawing/writing is very slow)
    scale = 1.0
    out_w, out_h = width, height
    if args.width and width > args.width:
        scale = args.width / width
        out_w = args.width
        out_h = int(height * scale) // 2 * 2

    device = pick_device(args.device)
    print("=" * 62)
    print(" Vehicle Tracking & Counting System")
    print("=" * 62)
    print(f" Source   : {source}  ({width}x{height} @ {fps:.1f} fps)")
    print(f" Model    : {args.weights}   device={device}")
    print(f" Settings : imgsz={args.imgsz}  stride={args.stride}  conf={args.conf}  "
          f"output={out_w}x{out_h}")
    print(f" Output   : {out_video}")
    print(" Press Ctrl+C at any time - partial results are still saved.")
    print("=" * 62)

    from ultralytics import YOLO      # imported late so --help stays instant
    model = YOLO(args.weights)

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (out_w, out_h))
    counter = LineCounter(out_w, out_h, args.orientation, args.line, args.invert)
    viz = Visualizer(out_h, show_trails=not args.no_trails and config.SHOW_TRAILS,
                     trail_length=config.TRAIL_LENGTH)

    extra = {"half": True} if device != "cpu" else {}
    results = model.track(
        source=source,
        classes=list(config.VEHICLE_CLASSES),
        conf=args.conf, iou=config.IOU, imgsz=args.imgsz,
        tracker=config.TRACKER, persist=True, stream=True,
        vid_stride=args.stride, device=device,
        verbose=False, **extra,
    )

    steps, t0 = 0, time.time()
    try:
        for result in results:
            steps += 1
            frame = (cv2.resize(result.orig_img, (out_w, out_h)) if scale != 1.0
                     else result.orig_img.copy())
            time_sec = steps * args.stride / fps
            live = 0

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for box, tid, cid in zip(boxes.xyxy.cpu().numpy(),
                                         boxes.id.cpu().numpy().astype(int),
                                         boxes.cls.cpu().numpy().astype(int)):
                    name = config.VEHICLE_CLASSES.get(int(cid))
                    if name is None:
                        continue
                    live += 1
                    box = box * scale
                    centroid = counter.update(int(tid), box, name, steps, time_sec)
                    viz.draw_vehicle(frame, box, int(tid), name, centroid)

            viz.draw_line(frame, counter)
            viz.draw_dashboard(frame, counter, live)
            writer.write(frame)

            if steps % 10 == 0:
                elapsed = time.time() - t0
                speed = steps / elapsed
                if total_steps:
                    eta = (total_steps - steps) / speed if speed else 0
                    print(f"[{steps / total_steps * 100:5.1f}%] frame {steps}/{total_steps} | "
                          f"{speed:.2f} fps | ETA {fmt_time(eta)} | counted: {counter.total}")
                else:
                    print(f"frame {steps} | {speed:.2f} fps | counted: {counter.total}")

            if args.show:
                cv2.imshow("Vehicle Tracking & Counting (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and steps >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user - saving what has been processed so far...")
    finally:
        writer.release()
        cv2.destroyAllWindows()

    # ---- reports ----------------------------------------------------------
    elapsed = time.time() - t0
    summary = counter.summary()
    summary.update(frames_processed=steps, avg_fps=round(steps / elapsed, 2) if elapsed else 0,
                   video=str(source))

    with open(f"{base}_events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "vehicle_class", "direction", "frame", "time_sec"])
        for e in counter.events:
            w.writerow([e.track_id, e.vehicle_class, e.direction, e.frame, e.time_sec])
    with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 62)
    print(" RESULT")
    print("=" * 62)
    print(f" Frames processed : {steps}   (avg {summary['avg_fps']} fps)")
    print(f" Total counted    : {counter.total}   (IN {counter.count_in} / OUT {counter.count_out})")
    for cls in ("Car", "Bus", "Truck", "Motorcycle"):
        print(f"   {cls:<11}: {counter.by_class.get(cls, 0)}")
    print(f" Video   -> {out_video}")
    print(f" CSV     -> {base}_events.csv")
    print(f" Summary -> {base}_summary.json")
    if counter.total == 0:
        print("\n [TIP] Nothing crossed the line. Check the annotated video: if boxes are "
              "visible,\n       move the line with --line 0.4 (or --orientation vertical). "
              "If there are no boxes,\n       try a lower --conf 0.15 or a video with a "
              "CCTV / elevated angle (not straight top-down).")


if __name__ == "__main__":
    main()
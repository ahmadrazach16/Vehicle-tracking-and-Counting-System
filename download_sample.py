"""
download_sample.py
------------------
Downloads a free, watermark-free traffic video into input/traffic.mp4
so you can test the project immediately.

    python download_sample.py
"""

import os
import urllib.request

import config


def main():
    os.makedirs(config.INPUT_DIR, exist_ok=True)
    dest = os.path.join(config.INPUT_DIR, "traffic.mp4")
    if os.path.exists(dest):
        print(f"[INFO] {dest} already exists - nothing to do.")
        return

    print(f"[INFO] Downloading sample video...\n       {config.SAMPLE_VIDEO_URL}")
    req = urllib.request.Request(config.SAMPLE_VIDEO_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        print("        Download any traffic video manually and put it inside the input/ folder.")
        return
    print(f"[DONE] Saved to {dest}")


if __name__ == "__main__":
    main()

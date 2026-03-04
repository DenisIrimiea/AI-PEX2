#!/usr/bin/env python3
"""
rover_data_processor.py

Processes RealSense .bag files and matching telemetry .csv files.
Outputs binary preprocessed images labeled with throttle/steering/heading.

Expected layout:
  AI-PEX2/
    rover_data_processor.py
    Cripple/
      rover_data/               <-- .bag + .csv + .log
      rover_data_processed/     <-- output folders created here
"""

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
import os
from imutils.video import FPS


# ---------- PATHS (relative to this script) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # AI-PEX2
SOURCE_PATH = os.path.join(SCRIPT_DIR, "Cripple", "rover_data")
DEST_PATH = os.path.join(SCRIPT_DIR, "Cripple", "rover_data_processed")
# -----------------------------------------------------


# ---------------- IMAGE SETTINGS ----------------
resize_W, resize_H = 160, 120
crop_top_pixels = 160         # remove top 160px from 480px height
white_L, white_H = 200, 255   # threshold for "white tape"
morph_k = 3                   # morphology kernel size
# ------------------------------------------------


def load_telem_file(path: str):
    """Load telemetry CSV into dict keyed by frame index."""
    lookup = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("index")
            if idx is not None:
                lookup[idx] = row
    return lookup


def preprocess_frame(color_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB frame to a binary mask highlighting white tape.
    Returns uint8 image of shape (resize_H, resize_W), values 0 or 255.
    """
    # RealSense provides RGB; convert to OpenCV-friendly BGR
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    # Crop bottom region
    h, w = bgr.shape[:2]
    top = min(max(crop_top_pixels, 0), h - 1)
    cropped = bgr[top:, :]

    # Resize for speed
    resized = cv2.resize(cropped, (resize_W, resize_H), interpolation=cv2.INTER_AREA)

    # Grayscale + blur
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold for white
    bw = cv2.inRange(gray, white_L, white_H)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    return bw


def process_bag_file(source_file: str, skip_if_exists: bool = True):
    fps = None
    pipeline = None
    playback = None

    try:
        print(f"\nProcessing {source_file}...")

        file_stem = os.path.basename(source_file).replace(".bag", "")
        out_dir = os.path.join(DEST_PATH, file_stem)

        # Skip if already processed
        if skip_if_exists and os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            print(f"  -> {file_stem} already processed. Skipping.")
            return

        os.makedirs(out_dir, exist_ok=True)

        csv_path = source_file.replace(".bag", ".csv")
        if not os.path.exists(csv_path):
            print(f"  -> Missing CSV for {file_stem}, skipping.")
            return

        telem_lookup = load_telem_file(csv_path)

        # ---- RealSense playback setup ----
        config = rs.config()
        pipeline = rs.pipeline()

        # Point to bag file
        config.enable_device_from_file(source_file, repeat_playback=False)

        # IMPORTANT FIX:
        # Do NOT force enable_stream() for playback.
        # Let the bag's recorded streams drive the configuration.
        pipeline.start(config)

        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)  # IMPORTANT FIX: avoid realtime stalls

        align = rs.align(rs.stream.color)
        fps = FPS().start()

        # ---- Read frames until EOF / timeout ----
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                # Typically EOF or stream stall; stop cleanly
                print(f"  -> [wait_for_frames] {e} (stopping {file_stem})")
                break

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_number = color_frame.frame_number
            row = telem_lookup.get(str(frame_number))
            if row is None:
                continue

            throttle = row.get("throttle", "0")
            steering = row.get("steering", "0")
            heading = row.get("heading", "0")

            color_rgb = np.asanyarray(color_frame.get_data())
            bw = preprocess_frame(color_rgb)

            filename = f"{int(frame_number):09d}_{throttle}_{steering}_{heading}_bw.png"
            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, bw)

            fps.update()

    except Exception as e:
        print(f"[Process Error] {e}")

    finally:
        try:
            if fps:
                fps.stop()
        except Exception:
            pass

        try:
            if pipeline:
                pipeline.stop()
        except Exception:
            pass

        print(f"Finished {source_file}. FPS: {fps.fps() if fps else 'N/A'}")


def main():
    os.makedirs(DEST_PATH, exist_ok=True)

    if not os.path.isdir(SOURCE_PATH):
        print(f"Source folder not found: {SOURCE_PATH}")
        print("Expected: AI-PEX2/rover_data/")
        return

    bag_files = sorted([f for f in os.listdir(SOURCE_PATH) if f.endswith(".bag")])
    if not bag_files:
        print(f"No .bag files found in {SOURCE_PATH}")
        return

    print(f"Found {len(bag_files)} bag(s) in {SOURCE_PATH}")
    for bag in bag_files:
        process_bag_file(os.path.join(SOURCE_PATH, bag))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
rover_data_processor.py
"""

import argparse
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import csv
import os
from imutils.video import FPS


# Paths for source and destination data (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # AI-PEX2-main
SOURCE_PATH = os.path.join(SCRIPT_DIR, "rover_data")
DEST_PATH = os.path.join(SOURCE_PATH, "processed_data")

# Parameters for image processing
# Define the range of white values to be considered for binary conversion
white_L, white_H = 200, 255

# Resize dimensions (quarter-ish of 640x480)
resize_W, resize_H = 160, 120

# Crop from top to focus on relevant part of the image
crop_top_pixels = 160         # remove top 160px from 480px height

# Morphology kernel size
morph_k = 3


def load_telem_file(path: str):
    """
    Loads telemetry data from a CSV file into a dict keyed by frame index (as strings).
    Helps with speed of data processing.
    """
    lookup = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get("index")
            if idx is not None:
                lookup[str(idx)] = row
    return lookup


def preprocess_frame(color_rgb: np.ndarray) -> np.ndarray:

    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    # Crop bottom region
    h, _ = bgr.shape[:2]
    top = min(max(crop_top_pixels, 0), h - 1)
    cropped = bgr[top:, :]

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


def process_bag_file(source_file, dest_folder=None, skip_if_exists=True):
    """
    Processes a single .bag file, extracting frames and converting them to grayscale and binary images.
    Saves these images to a specified destination directory.
    """
    fps = None
    pipeline = None
    playback = None

    try:
        print(f"Processing {source_file}...")

        file_name = os.path.basename(source_file).replace(".bag", "")
        dest_path = os.path.join(dest_folder or DEST_PATH, file_name)

        # Keep original behavior: skip only if folder exists AND has files inside
        if skip_if_exists and os.path.isdir(dest_path) and len(os.listdir(dest_path)) > 0:
            print(f"{file_name} previously processed; skipping.")
            return

        os.makedirs(dest_path, exist_ok=True)

        csv_path = source_file.replace(".bag", ".csv")
        if not os.path.exists(csv_path):
            print(f"  -> Missing CSV for {file_name}, skipping.")
            return

        frm_lookup = load_telem_file(csv_path)

        # Setup RealSense pipeline (KEEP working behavior)
        config, pipeline = rs.config(), rs.pipeline()
        config.enable_device_from_file(source_file, repeat_playback=False)

        # Don't enable_stream() for playback; let bag drive it.
        pipeline.start(config)

        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)

        alignedFs = rs.align(rs.stream.color)
        fps = FPS().start()

        # Processing loop (KEEP working EOF/timeout logic)
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"  -> [wait_for_frames] {e} (stopping {file_name})")
                break

            aligned_frames = alignedFs.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue

            # Frame number
            frm_num = int(color_frame.get_frame_number())

            # Skip if no telemetry data for frame
            row = frm_lookup.get(str(frm_num))
            if row is None:
                continue

            # Extract throttle, steering, and heading data
            throttle = row.get("throttle", "0")
            steering = row.get("steering", "0")
            heading = row.get("heading", "0")

            color_rgb = np.asanyarray(color_frame.get_data())

            # Image processing using OpenCV (your working pipeline)
            Img_frame_placeholder = preprocess_frame(color_rgb)

            # Save processed images WITH LABELS in the name
            bw_frm_name = f"{int(frm_num):09d}_{throttle}_{steering}_{heading}_bw.png"
            cv2.imwrite(os.path.join(dest_path, bw_frm_name), Img_frame_placeholder)

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

        if fps:
            print(f"Finished {source_file}. FPS: {fps.fps():.2f}")
        else:
            print(f"Finished {source_file}. FPS: N/A")


def main():
    """
    Main function to process one bag or all .bag files in the source directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        type=str,
        default=None,
        help="Process a specific .bag file in rover_data/ (example: cloning-20260225-102624.bag). "
             "If omitted, processes all .bag files in rover_data/."
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reprocess even if output folder already exists and has files."
    )
    args = parser.parse_args()

    os.makedirs(DEST_PATH, exist_ok=True)

    if not os.path.isdir(SOURCE_PATH):
        print(f"Source folder not found: {SOURCE_PATH}")
        print("Expected: <project_root>/rover_data/")
        return

    skip_if_exists = not args.no_skip

    # If a specific bag was provided
    if args.bag:
        bag_path = args.bag
        if not os.path.isabs(bag_path):
            bag_path = os.path.join(SOURCE_PATH, bag_path)

        if not os.path.exists(bag_path):
            print(f"Bag file not found: {bag_path}")
            return

        process_bag_file(bag_path, dest_folder=None, skip_if_exists=skip_if_exists)
        return

    # Otherwise process all bags
    bag_files = sorted([f for f in os.listdir(SOURCE_PATH) if f.endswith(".bag")])
    if not bag_files:
        print(f"No .bag files found in {SOURCE_PATH}")
        return

    print(f"Found {len(bag_files)} bag(s) in {SOURCE_PATH}")
    for filename in bag_files:
        process_bag_file(os.path.join(SOURCE_PATH, filename), dest_folder=None, skip_if_exists=skip_if_exists)


if __name__ == "__main__":
    main()
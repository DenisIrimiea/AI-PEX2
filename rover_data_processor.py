"""
rover_data_processor.py

"""

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
import os
from imutils.video import FPS

# Paths for source and destination data
SOURCE_PATH = "/media/usafa/data/rover_data"
DEST_PATH = "/media/usafa/data/rover_data_processed"

# Parameters for image processing
# Define the range of white values to be considered for binary conversion
white_L, white_H = 200, 255
# Resize dimensions (quarter of 640x480)
resize_W, resize_H = 160, 120
# Crop dimensions to focus on relevant parts of the image
crop_W, crop_B, crop_T = 160, 120, 40  # Crop from top third down

def load_telem_file(path):
    """
    Loads telemetry data from a CSV file to associate video frames with sensor data like throttle and steering.
    """
    with open(path, "r") as f:
        dict_reader = csv.DictReader(f)
        return list(dict_reader)

def process_bag_file(source_file, dest_folder=None, skip_if_exists=True):
    """
    Processes a single .bag file, extracting frames and converting them to grayscale and binary images.
    Saves these images to a specified destination directory.
    """
    try:
        print(f"Processing {source_file}...")
        file_name = os.path.basename(source_file.replace(".bag", ""))
        dest_path = os.path.join(dest_folder or DEST_PATH, file_name)
        
        if skip_if_exists and os.path.isdir(dest_path):
            print(f"{file_name} previously processed; skipping.")
            return

        os.makedirs(dest_path, exist_ok=True)
        frm_lookup = load_telem_file(source_file.replace(".bag", ".csv"))

        # Setup RealSense pipeline
        config, pipeline = rs.config(), rs.pipeline()
        rs.config.enable_device_from_file(config, source_file, False)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pipeline.start(config)
        
        # Allow time for pipeline to stabilize
        time.sleep(1)
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(True)
        alignedFs = rs.align(rs.stream.color)
        fps = FPS().start()

        # Processing loop
        while playback.current_status() == rs.playback_status.playing:
            try:
                playback.pause()  # Pause before getting frames
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                aligned_frames = alignedFs.process(frames)
                color_frame = aligned_frames.get_color_frame()
                
                # Skip if no telemetry data for frame
                frm_num = color_frame.frame_number
                result = [entry for entry in frm_lookup if entry["index"] == str(frm_num)]
                if not result: 
                    playback.resume()  # Resume before continuing
                    continue
                
                # Extract throttle, steering, and heading data
                throttle, steering, heading = result[0]["throttle"], result[0]["steering"], result[0]["heading"]
                color_frame = np.asanyarray(color_frame.get_data())

                # TODO: Image processing using OpenCV:
                Img_frame_placeholder = None
                # Save processed images WITH LABELS in the name
                bw_frm_name = f"{int(frm_num):09d}_{throttle}_{steering}_{heading}_bw.png"
                cv2.imwrite(os.path.join(dest_path, bw_frm_name), Img_frame_placeholder)
                fps.update()
                
                playback.resume()  # Resume after processing

            except Exception as e:
                print(e)
                playback.resume()  # Make sure to resume even on error
                continue
    except Exception as e:
        print(e)
    finally:
        # Cleanup and stats
        if fps: fps.stop()
        if playback and playback.current_status() == rs.playback_status.playing:
            playback.pause()
            if pipeline: pipeline.stop()
        print(f"Finished {source_file}. FPS: {fps.fps() if fps else 'N/A'}")

def main():
    """
    Main function to process all .bag files in the source directory.
    """
    for filename in filter(lambda f: f.endswith(".bag"), os.listdir(SOURCE_PATH)):
        process_bag_file(os.path.join(SOURCE_PATH, filename))

if __name__ == "__main__":
    main()

"""
Rover Data Collection Program

This script records video data from a RealSense camera while logging telemetry 
data from a rover controlled via ArduPilot. The program connects to the rover 
using MAVLink, listens for RC channel updates, and logs throttle, steering, and heading
information into a CSV file alongside the recorded video stream (.bag format). 
The collected data can be used for further analysis and training machine learning models.

Key Features:
- Connects to an ArduPilot-based rover via MAVLink.
- Captures video frames from an Intel RealSense camera.
- Logs telemetry data (throttle, steering, heading) with timestamps and frame index.


"""

from imutils.video import FPS
from dronekit import connect
import argparse
import pyrealsense2.pyrealsense2 as rs
from pymavlink import mavutil
import time
import logging
import sys
import utilities.drone_lib as dl
import csv
import os

# Default settings for telemetry connection and data storage
DEFAULT_BAUD = 115200  # Default baud rate for telemetry connection
DEFAULT_DATA_PATH = '/media/usafa/data/rover_data/'  # Default directory for storing rover data
# Experimenting with different telemetry ports
# DEFAULT_PORT = "/dev/ttyUSB0"  # USB connection
# DEFAULT_PORT = "/dev/ttyACM0"  # Serial connection
DEFAULT_PORT = "127.0.0.1:14550"  # Default telemetry connection port (UDP)
connection = None  # Global variable for drone connection


def prepare_log_file(log_file):
    """
    Set up logging configuration to log messages to both a file and console.
    """
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Ensure handlers are cleared to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


# Appends samples to your training data file
def append_ardu_data(idx, throttle, steering, heading, file_path):
    """
    Append telemetry data to the specified CSV data file.
    """
    try:
        with open(file_path, "a") as f:
            f.write(f"{idx},{throttle},{steering},{heading}\n")
    except Exception as e:
        logging.error(f"Error writing telemetry data to {file_path}: {e}", exc_info=True)

# Utility function related to the callback hack below.
def set_rc(vehicle, chnum, value):
    """
    Set a specific RC channel value on the vehicle.
    """
    vehicle._channels._update_channel(str(chnum), value)

# A callback hack to ensure channel values 
# from the autopilot are updated in realtime.
def device_channel_msg(device):
    """
    Capture and update vehicle channel values for reference.
    """
    @device.on_message('RC_CHANNELS')
    def RC_CHANNEL_listener(vehicle, name, message):
        set_rc(vehicle, 1, message.chan1_raw)
        set_rc(vehicle, 2, message.chan2_raw)
        set_rc(vehicle, 3, message.chan3_raw)
        set_rc(vehicle, 4, message.chan4_raw)
        set_rc(vehicle, 5, message.chan5_raw)
        set_rc(vehicle, 6, message.chan6_raw)
        set_rc(vehicle, 7, message.chan7_raw)
        set_rc(vehicle, 8, message.chan8_raw)
        set_rc(vehicle, 9, message.chan9_raw)
        set_rc(vehicle, 10, message.chan10_raw)
        set_rc(vehicle, 11, message.chan11_raw)
        set_rc(vehicle, 12, message.chan12_raw)
        set_rc(vehicle, 13, message.chan13_raw)
        set_rc(vehicle, 14, message.chan14_raw)
        set_rc(vehicle, 15, message.chan15_raw)
        set_rc(vehicle, 16, message.chan16_raw)
        vehicle.notify_attribute_listeners('channels', vehicle.channels)


def collect_data(bag_file):
    """
    Main data collection loop: records video and telemetry data.
    """
    state_update_interval = 30  # Log state every 30 frames

    try:
        file_name = bag_file.replace(".bag", ".csv")
        logging.info(f"Recording data to: {bag_file}")

        # Open CSV file for writing telemetry data
        with open(file_name, 'w', newline='') as data_file:
            writer = csv.writer(data_file)
            writer.writerow(['index', 'throttle', 'steering', 'heading'])  # Write header

        # Configure RealSense pipeline to save video
        pipeline = rs.pipeline()
        config = rs.config()
        # TODO: enable recodring to file
        # TODO: enable 8-bit rgb color stream stream, 640x480 @ 30 fps
        
        #Start the stream
        pipeline.start(config)

        fps = FPS().start()
        logging.info("RealSense recording started.")
    except Exception as e:
        logging.error("Error initializing recording.", exc_info=True)
        return

    # Loop while the vehicle is armed
    while connection and connection.armed:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # TODO: 
            # (1) collect throttle, steering, heading (not required), and frame number
            # example: throttle = int(connection.channels.get('3', 1500))  # Default neutral throttle
            # heading = getattr(connection, 'heading', 0)
            # get frame number from your frame's "frame_number" property - might want to cast to be numeric
            
            frm_num=0
            # Append details to your comma delimited file that's paired with your video bag
            # append_ardu_data(frm_num, throttle, steering_mix, heading, file_name)
            

            if frm_num % state_update_interval == 0:
                dl.display_rover_state(connection)
            
            fps.update()
        except Exception as e:
            logging.error("Error during data collection loop.", exc_info=True)
            break
    
    # Stop recording
    logging.info("Stopping recording...")
    pipeline.stop()
    fps.stop()
    logging.info("Waiting for video to flush...")
    time.sleep(7)
    logging.info(f"Elapsed time: {fps.elapsed():.2f}s, Approx FPS: {fps.fps():.2f}")


if __name__ == "__main__":
    prepare_log_file("rover_collect.log")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--output", type=str, help="Path to output file(s).")
    parser.add_argument("-port", "--port", type=str, help="Telemetry port for Ardupilot.")
    args = parser.parse_args()
    
    port = args.port if args.port else DEFAULT_PORT
    storage_root = args.output if args.output else DEFAULT_DATA_PATH
    os.makedirs(storage_root, exist_ok=True)
    
    print(f"Connecting to autopilot on {port} at {DEFAULT_BAUD} baud...")
    connection = dl.connect_device(port, DEFAULT_BAUD, timeout=60)
    if not connection:
        logging.error("Failed to connect to the autopilot.")
        sys.exit(1)
    
    device_channel_msg(connection)
    dl.display_rover_state(connection)
    
    while True:
        print("Arm vehicle to start recording. (CTRL-C to exit)")
        while not connection.armed:
            time.sleep(1)
        
        bag_file = os.path.join(storage_root, time.strftime("cloning_%Y%m%d-%H%M%S") + ".bag")
        prepare_log_file(bag_file.replace(".bag", ".log"))
        collect_data(bag_file)
        print("Recording stopped.")

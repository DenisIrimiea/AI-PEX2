"""
rover_driver.py
"""

import os
import pyrealsense2.pyrealsense2 as rs
import time
import numpy as np
import cv2
import keras
import utilities.drone_lib as dl


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(SCRIPT_DIR, "models",     "rover_model_01_ver01_epoch0014_val_loss0.0151.h5"
)


MIN_STEERING, MAX_STEERING = 1176, 2006
MIN_THROTTLE, MAX_THROTTLE = 985, 1766



white_L, white_H = 200, 255  # White color range
resize_W, resize_H = 160, 120  # Resized image dimensions

crop_top_pixels = 160
morph_k = 3

def get_model(filename):
    """Load and compile the TensorFlow Keras model."""
    model = keras.models.load_model(filename, compile=False)
    model.compile()
    print("Loaded Model")
    return model

def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)

def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min

def denormalize(steering, throttle):
    """Denormalize steering and throttle values to the rover's command range."""
    steering = invert_min_max_norm(steering, MIN_STEERING, MAX_STEERING)
    throttle = invert_min_max_norm(throttle, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def initialize_pipeline(brg=False):
    """Initialize the RealSense pipeline for video capture."""
    pipeline = rs.pipeline()
    config = rs.config()

    # ✅ keep BGR
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    return pipeline

def get_video_data(pipeline):
    """Capture a video frame, preprocess it, and prepare it for model prediction."""
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    if not color_frame:
        return None

    image = np.asanyarray(color_frame.get_data())  # BGR (640x480)

    # ✅ REQUIRED preprocessing to match training pipeline:
    # crop bottom region (remove top crop_top_pixels)
    h, _ = image.shape[:2]
    top = min(max(crop_top_pixels, 0), h - 1)
    cropped = image[top:, :]

    resized = cv2.resize(cropped, (resize_W, resize_H), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bw = cv2.inRange(gray, white_L, white_H)  # 0/255

    # morph cleanup (same as processor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # normalize to [0,1] and add channel+batch dims => (1,120,160,1)
    bw = bw.astype(np.float32) / 255.0
    bw = np.expand_dims(bw, axis=-1)
    bw = np.expand_dims(bw, axis=0)

    return bw

def set_rover_data(rover, steering, throttle):
    """Set rover control commands, ensuring they're within the valid range."""
    steering, throttle = check_inputs(int(steering), int(throttle))
    rover.channels.overrides = {"1": steering, "3": throttle}
    print(f"Steering: {steering}, Throttle: {throttle}")

def check_inputs(steering, throttle):
    """Check and clamp the steering and throttle inputs to their allowed ranges."""
    steering = int(np.clip(steering, MIN_STEERING, MAX_STEERING))
    throttle = int(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))
    return steering, throttle

def main():

    """Main function to drive the rover based on model predictions."""
   
    # ✅ FIX: match your earlier working connection style (port + baud)
    port = "/dev/ttyACM0"
    baud = 115200
    rover = dl.connect_device(port, baud)

    # Load the trained model
    model = get_model(MODEL_NAME)

    if model is None:
        print("Unable to load CNN model!")
        rover.close()
        print("Terminating program...")
        exit()
        
    while True:
        print("Arm vehicle to start mission.")
        print("(CTRL-C to stop process)")
        while not rover.armed:
            time.sleep(1)
        
        # Initialize video capture
        pipeline = initialize_pipeline()
        
        while rover.armed:
            processed_image = get_video_data(pipeline)
            if processed_image is None:
                print("No image from camera.")
                continue

            output = model.predict(processed_image, verbose=0)  # shape (1,2)

            steering, throttle = denormalize(output[0][0], output[0][1])
            set_rover_data(rover, steering, throttle)

        pipeline.stop()
        time.sleep(1)
        pipeline = None
        rover.close()
        print("Done.")

if __name__ == "__main__":
    main()
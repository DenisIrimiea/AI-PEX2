"""
rover_driver.py
"""

import pyrealsense2.pyrealsense2 as rs
import time
import numpy as np
import cv2
import keras
import utilities.drone_lib as dl

# Path to the trained model weights
MODEL_NAME = "models/V6.6_SGD_Batch115_664Steps/rover_model06_ver06_epoch0095_val_loss0.005095.h5"

# Rover driving command limits
MIN_STEERING, MAX_STEERING = 1000, 2000
MIN_THROTTLE, MAX_THROTTLE = 1500, 2000

"""
HINT:  Get values to the above by querying your own rover...
rover.parameters['RC3_MAX']
rover.parameters['RC3_MIN']
rover.parameters['RC1_MAX']
rover.parameters['RC1_MIN']
"""

# Image processing parameters
white_L, white_H = 200, 255  # White color range
resize_W, resize_H = 160, 120  # Resized image dimensions
crop_W, crop_B, crop_T = 160, 120, 40  # Crop box dimensions

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
    throttle = invert_min_max_norm(steering, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def initialize_pipeline(brg=False):
    """Initialize the RealSense pipeline for video capture."""
    pipeline = rs.pipeline()
    config = rs.config()

    if brg:
        config.enable_stream(rs.stream.color, 
                             640, 480, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, 
                             rs.format.rgb, 30)
    pipeline.start(config)
    return pipeline

def get_video_data(pipeline):
    """Capture a video frame, preprocess it, and prepare it for model prediction."""
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    if not color_frame:
        return None

    image = np.asanyarray(color_frame.get_data())
    
    #TODO: process your incoming frame so that it is 
    #      in the form required to feed into your CNN.
    # Maybe resize
    # perhaps convert to gray
    # turn into B&W (using cv.inRange)
    # Perform cropping (if any)
    # etc...

    return image

def set_rover_data(rover, steering, throttle):
    """Set rover control commands, ensuring they're within the valid range."""
    
    # May uncomment below to force a specific range, if your model is 
    # sometimes outputting weird ranges (should not be needed)
    #steering, throttle = check_inputs(int(steering), int(throttle))
    
    rover.channels.overrides = {"1": steering, "3": throttle}
    print(f"Steering: {steering}, Throttle: {throttle}")


def check_inputs(steering, throttle):
    """Check and clamp the steering and throttle inputs to their allowed ranges."""
    steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
    throttle = np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def main():

    """Main function to drive the rover based on model predictions."""
   
    # Setup and connect to the rover
    rover = dl.connect_device("/dev/ttyUSB0")

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

            # Predict steering and throttle from the processed image
            #TODO: get model predictions

            #output = get_prdictions
            
            # Note that you may denormalize values now, 
            # if you trained your model on normalized values.
            #steering, throttle = denormalize(output[0][0], output[0][1])
            
            # Next, send predicted values to rover to be executed
            # Note: this is where your model drives!
            #set_rover_data(rover, steering, throttle)

        # stop recording
        pipeline.stop()
        time.sleep(1)
        pipeline = None
        rover.close()
        print("Done.")

if __name__ == "__main__":
    main()

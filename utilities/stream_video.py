# Documentation is in rover_recorder.py

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from imutils.video import FPS

#initialize camera settings
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)


def stream_video():
#stream video until exit condition ('q' is pressed)
    fps = FPS().start()
    while True:

        frame = pipeline.wait_for_frames() #get frame
        depth_frame = frame.get_depth_frame() #get color and depth frames for viewing
        color_frame = frame.get_color_frame()

        if not depth_frame or not color_frame: #loop again if both frames are not available
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        #print('stream')
        cv2.imshow('color', color_img) #dislplay images
        cv2.imshow('depth', depth_image)
        #keep track of fps
        fps.update()

#exit condition
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    try:

        # stop recording
        if fps is not None:
            fps.stop()
            print("Elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            
    except Exception as e:
        print("Unexpected error during cleanup.", exc_info=True)

stream_video()

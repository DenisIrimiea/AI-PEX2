import pyrealsense2 as rs
import numpy as np
import cv2 
"""
configure camera
"""
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
pipeline.start(config)

frame = pipeline.wait_for_frames().get_color_frame()
if frame:
    img = np.asanyarray(frame.get_data())
    print('frame captured', img.shape)
    cv2.imwrite('test_image.jpg', img)


pipeline.stop()


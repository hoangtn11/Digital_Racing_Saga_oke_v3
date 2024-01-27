# import cv2
# from platform_modules.camera import Camera
# import global_storage as gs
# import signal
# import sys


import cv2
from platform_modules.camera import Camera
import global_storage as gs
import signal
import sys
import time  # Import the time module for measuring time intervals
from lane.lane_line_detection import laneDetector
# from lane.lane_line_segmentation import laneDetector
from utils.queue_handle import *
# Init camera
camera = Camera()
camera.start()

# lane = laneDetector()
# lane.start()

# Variables for fps calculation
start_time = time.time()
frame_count = 0

# Grab images and show
while not gs.exit_signal:
    if not gs.rgb_frames.empty():
        rgb = gs.rgb_frames.get()
        # mask = get_fast(gs.mask_img)
        cv2.imshow("RGB", rgb)
        cv2.waitKey(1)
        # Increment frame count
        frame_count += 1
        
        # Calculate fps every second
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # Check if one second has passed
            fps = frame_count / elapsed_time
            #print(f"FPS: {fps:.2f}")
            
            # Reset counters for the next second
            start_time = time.time()
            frame_count = 0

camera.join()
# lane.join()

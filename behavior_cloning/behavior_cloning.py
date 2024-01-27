import cv2
import numpy as np
import threading
import time
import global_storage as gs
from utils.param import Param

class E2E(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        print("-------------Init E2E")
        # List of class names
        self.param = Param()
        # {'3_way_intersection': 0, 'left': 1, 'middle_roundabout': 2, 'outer_roundabout': 3, 'right': 4, 'straight': 5, 'unknown': 6}
        self.onnx_session = self.param.e2e_model
        self.session_input_name = self.onnx_session.get_inputs()[0].name
        self.session_output_name = self.onnx_session.get_outputs()[0].name
        
    def map_range(self, value, from_min, from_max, to_min, to_max):
        # First, normalize the value from the input range to [0, 1]
        normalized_value = (value - from_min) / (from_max - from_min)
        
        # Then, map the normalized value to the output range
        mapped_value = normalized_value * (to_max - to_min) + to_min
        
        return mapped_value
    def run(self):
        while not gs.exit_signal:
            if gs.e2e_images.empty():
                time.sleep(0.1)
                continue
            print("*** E2E process")

            image = gs.e2e_images.get()
            # print(image.shape)
            resized = cv2.resize(image, (40, 40))
            #cv2.imshow("input", resized)
            #cv2.waitKey(1)

            img = resized.astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = img.reshape(img.shape[0], 40, 40, 1)
            # print(img.shape)
            result = self.onnx_session.run([self.session_output_name], {self.session_input_name: img})
            
            # steering_angle = self.map_range(float(result[0]), -1, 1 , -60, 60)
            steering_angle = float(result[0]) * 100
            print(steering_angle)
            # if (steering_angle > 20):
            #     steering_angle = 60
            # elif (steering_angle < -20):
            #     steering_angle = -60
            gs.e2e_steering = steering_angle
            print("                             ***** E2E steering angle: ", steering_angle)
            # if gs.show_Object:
            #     cv2.imshow('Detection object', draw)
            #     cv2.waitKey(1)
            

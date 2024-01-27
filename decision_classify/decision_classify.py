import cv2
import numpy as np
import threading
import time
import global_storage as gs
from utils.param import Param

class DesisionClassifier(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # List of class names
        self.param = Param()
        # {'3_way_intersection': 0, 'left': 1, 'middle_roundabout': 2, 'outer_roundabout': 3, 'right': 4, 'straight': 5, 'unknown': 6}
        self.classes = ["3_way_intersection","left", "middle_roundabout", "outer_roundabout", "right", "straight", "unknown"]
        self.onnx_session = self.param.decision_classifier_onnx
        
    def run(self):
        while not gs.exit_signal:
            if gs.dc_images.empty():
                time.sleep(0.1)
                continue
            image = gs.dc_images.get()
            # print(image.shape)
            image = cv2.resize(image, (32, 32))
            rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

            # Copy the grayscale channel to all three channels of the RGB image
            rgb_image[:, :, 0] = image
            rgb_image[:, :, 1] = image
            rgb_image[:, :, 2] = image

            draw = rgb_image.copy()
            # # Detect faces using Haar Cascade
    
            # # Preprocess the grayscale face image
            gray_face_resized = cv2.resize(draw, (32, 32))
            gray_face_normalized = gray_face_resized.astype('float32') / 255.0
            gray_face_normalized = np.expand_dims(gray_face_normalized, axis=0)  # Add batch dimension

            # Perform prediction
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            predictions = self.onnx_session.run([output_name], {input_name: gray_face_normalized})
            # print("pre: ", predictions)

            # Get predicted class and confidence
            predicted_class = np.argmax(predictions)
            # confidence = predictions[0][0][predicted_class]
            # print("Confidence:", confidence)
            print("+++++++++++++++Predicted class:", self.classes[predicted_class])


            # gs.decision_class = self.classes[predicted_class]
            gs.decision_class  = predicted_class
            
            # if gs.show_Object:
            #     cv2.imshow('Detection object', draw)
            #     cv2.waitKey(1)
            

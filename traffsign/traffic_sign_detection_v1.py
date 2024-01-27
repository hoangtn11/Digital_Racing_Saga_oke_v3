import cv2
import threading
import numpy as np
import time
import global_storage as gs
from utils.param import Param
from utils.detection import ObjectFinder
# import onnxruntime as ort
import numpy as np

class SignDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.param = Param()
        self.object_finder = ObjectFinder()
        # self.classes = ['unknown', 'left', 'no_left', 'right', 'no_right', 'straight', 'stop']
        self.classes = ['left', 'unknown', 'stop', 'unknown', 'unknown', 'unknown', 'right']
        self.class_indices_reverse = {0: '01_left', 1: '02_no_left', 2: '03_right', 3: '04_no_right', 4: '05_straight', 5: '06_stop'}
        self.session = self.param.traffic_sign_session
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self):  
        
        
        while not gs.exit_signal:
       
            img = gs.rgb_sign_frame.get()  
            # img = cv2.resize(img, (640, 480))
            draw = img.copy()
            points = self.object_finder.get_boxes_from_mask(img)

            # Preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = img / 255.0
            print("CCCCCCCCCCCCCCCCCCCC")

            # Classify signs using CNN
            detetc_signs = []
            for bbox in points:
                # Crop sign area
                x, y, w, h = bbox
                sub_image = img[y:y+h, x:x+w]
                if sub_image.shape[0] < 20 or sub_image.shape[1] < 20:
                    continue
                # Preprocess
                sub_image = cv2.resize(sub_image, (32, 32))

                # sub_image = input_image.resize((32, 32))  # Resize to match model input shape
                sub_image = np.array(sub_image)  # Convert to numpy array

                input_image = sub_image.astype(np.float32) / 255.0  # Normalize pixel values (modify if needed)
                    # Make predictions using the loaded ONNX model
                input_data = {self.input_name: input_image[np.newaxis, ...]}  # Add batch dimension
                outputs = self.session.run([self.output_name], input_data)

                # Get the predicted class index and confidence score for the predicted class
                predicted_class_index = np.argmax(outputs[0], axis=1)
                confidence_score = outputs[0][0][predicted_class_index[0]]

                # Map the predicted class index to label name using the class_indices_reverse dictionary
                predicted_label = self.class_indices_reverse[predicted_class_index[0]]
                
                cls = predicted_class_index
                
                print("predicted_class_index: ",predicted_class_index)
                print("outputs[0]: ",outputs[0])
                print("score: ", score)
                score = confidence_score
                print("========================cls: ", cls)
                # if cls == 1 or cls == 3 or cls == 4 or cls == 6 or score < 0.5:
                #     continue

                detetc_signs.append(self.classes[cls[0]])    
                
                # Draw prediction result
                if draw is not None and gs.show_trafficSign:
                    text = self.classes[cls[0]] + ' ' + str(round(score, 2))
                    cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 4)
                    cv2.putText(draw, text, (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            gs.signs = detetc_signs
            # print("detetc_signs: ", detetc_signs)
            if gs.show_trafficSign:
                cv2.imshow("Traffic signs", draw)
                cv2.waitKey(1)


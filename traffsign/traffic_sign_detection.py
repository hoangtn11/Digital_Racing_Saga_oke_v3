import cv2
import threading
import numpy as np
import time
import global_storage as gs
from utils.param import Param
from utils.detection import ObjectFinder


class SignDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.param = Param()
        self.object_finder = ObjectFinder()
        self.sign_classifier = self.param.traffic_sign_model
        # self.classes = ['unknown', 'left', 'no_left', 'right', 'no_right', 'straight', 'stop']
        # self.classes = ['left', 'unknown', 'right', 'unknown', 'unknown', 'stop', 'unknown']
        self.classes = ['unknown', 'right', 'left', 'stop']
        

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

            sub_image = np.expand_dims(sub_image, axis=0)

            # Use CNN to get prediction
            self.sign_classifier.setInput(sub_image)
            preds = self.sign_classifier.forward()
            preds = preds[0]
            cls = preds.argmax()
            score = preds[cls]

            # if cls == 1 or cls == 3 or cls == 4 or cls == 6 or score < 0.5:
            #     continue

            # detetc_signs.append([self.classes[cls]])
            
            # Draw prediction result
            if draw is not None and gs.show_trafficSign:
                text = self.classes[cls] + ' ' + str(round(score, 2))
                cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 4)
                cv2.putText(draw, text, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        gs.signs = detetc_signs
        # print("detetc_signs: ", detetc_signs)
        if gs.show_trafficSign:
            cv2.imshow("Traffic signs", draw)
            cv2.waitKey(1)


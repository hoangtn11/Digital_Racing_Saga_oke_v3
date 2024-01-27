import cv2
import onnxruntime as ort
import numpy as np

class Param:
    def __init__(self):
        self.minThrottle = 0.25
        self.maxThrotle = 0.35
        self.steering = 0
        
        self.minTurnTimeSign = 2
        self.maxTurnTimeSign = 3
        
        self.minTurnTime90 = 2
        self.maxTurnTime90 = 3
        
        self.minTurnTimeXuyen = 3
        self.maxTurnTimeXuyen = 4
        
        self.stoptime = 4
        
        self.maxCountTurnRight = 2
        self.maxCountTurnLeft = 2
        
        self.maxCountSign = 2
        self.maxCountDecision = 3
        self.maxCountObject = 3
        
        # Initalize traffic sign classifier
        self.traffic_sign_model = cv2.dnn.readNetFromONNX("models/traffic_sign_classifier_4.onnx")
        # self.traffic_sign_session = ort.InferenceSession('models/traffic_sign_classifier_3.onnx')
        # Load HaarCascade
        # self.cascade = cv2.CascadeClassifier('object/car.xml')
        # Filter for object classifier to fill
        # self.onnx_session = ort.InferenceSession('models/classification_model_v3.onnx')
        self.decision_classifier_onnx = ort.InferenceSession('models/decision_classifier.onnx')
        self.lane_segment_model_onnx = ort.InferenceSession('models/enet_v2.onnx')
        # self.e2e_model = ort.InferenceSession('models/e2e_obstacle_avoidance.onnx')
    

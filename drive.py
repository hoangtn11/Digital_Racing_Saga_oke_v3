from traffsign.traffic_sign_detection import SignDetector
from decision_classify.decision_classify import DesisionClassifier
# from lane.lane_line_segmentation import laneDetector
from lane.lane_line_detection_v1 import laneDetector

from controler.carcontroler import CarController
# from behavior_cloning.behavior_cloning import E2E
from platform_modules.button_reader import ButtonReader
from platform_modules.car_guard import CarGuard
from platform_modules.camera import Camera
from platform_modules.lcd_display import LCDDisplay
from platform_modules.motor_controller import MotorController


def main():
    # Start thread camera
    camera = Camera()
    camera.start()
    
    # Start thread lane detection
    laneDetect = laneDetector()
    laneDetect.start()
    
    # Start thread sign detection
    # signDetect = SignDetector()
    # signDetect.start()
    
    # Start thread decision classify
    decisionClass = DesisionClassifier()
    decisionClass.start()
    
    # Init motor controller
    motor_controller = MotorController()
    motor_controller.start()
    
    # Init button reader
    button_reader = ButtonReader()
    button_reader.start()
    
    # Init LCD to display
    lcd_display = LCDDisplay()
    lcd_display.start()
    
    # Car guard
    # Stop car when hitting obstacle or when user presses button 4
    guard = CarGuard()
    guard.start()
    
    
    # Start thread for end to end learning
    # e2e = E2E()
    # e2e.start()
    
    carcontrol = CarController()
    carcontrol.start()

    camera.join()
    # signDetect.join()
    decisionClass.join()
    laneDetect.join()
    motor_controller.join()
    guard.join()
    button_reader.join()
    # lcd_display.join()
    carcontrol.join()   
    # e2e.join() 
if __name__ == '__main__':
    main()

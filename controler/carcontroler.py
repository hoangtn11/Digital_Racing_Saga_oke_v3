import numpy as np
import cv2
import traceback
import time
import threading
from utils.param import Param
import global_storage as gs
from utils.PID_Fuzzy import Fuzzy_PID, PID
from utils.queue_handle import *
import config as cf

class CarController(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.rgb = None
        self.image = None
        # Set ROI
        self.roi1 = 0.92
        self.roi2 = 0.5
        
        # Set Turning
        self.param = Param()
        self.turning_time = 0
        self.seeSignTime = 0
        self.last_sign_time = 0
        self.lastSignDetection = ''
        self.lastSign = ''
        self.haveObject = 0
        self.turnStatus = 0
        
        # Init turn
        self.countTurnLeft1 = 0
        self.countTurnRight1 = 0
        self.countTurnLeft2 = 0
        self.countTurnRight2 = 0
        
        # Init turn for sign
        self.countSignLeft = 0
        self.countSignRight = 0
        self.countSignStop = 0
        
        # Init steer and throttle
        self.throttle = self.param.maxThrotle
        self.steering_angle = self.param.steering
        
        # Init image size
        self.im_height, self.im_width = 240, 320
        self.center = self.im_width // 2
        
        # Init for decision classify model
        self.countDecisionRight = 0
        self.countDecisionLeft = 0
        self.countDecisionStraight = 0
        self.countDecisionThree = 0
        self.countDecisionXuyen = 0
        self.decisionclass = None
        
        # Init for PID control
        
        # self.pid_controller = Fuzzy_PID(15,0,1,0,1, 0)
        self.pid_controller = PID(2,0,0)
        # self.pid_controller.setKp(15, 0)
        # self.pid_controller .setKi(1, 0)
        # self.pid_controller.setKd(1, 0)
        self.pid_controller.setSampleTime(0.015) # Set the sample time (adjust as needed)
        setpoint = 0.0
        self.pid_controller.setSetPoint(setpoint)
    
    # Convert mask image or rgb image to birdview image
    def birdview_transform(self, img):
        """
            Apply bird-view transform to the image
        """
        IMAGE_H = 240
        IMAGE_W = 320
        src = np.float32([[0, IMAGE_H], [320, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
        dst = np.float32([[120, IMAGE_H], [320 - 120, IMAGE_H], [-80, 0], [IMAGE_W+80, 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
        return warped_img
    
    # Find two point left and right of two line in the road
    def find_left_right_points(self):
        
        self.img_birdview = self.birdview_transform(self.image)
        
        if (gs.show_draw):
            self.rgb[:, :] = self.birdview_transform(self.rgb)
 
        interested_line_y = int(self.im_height * self.roi1)
        interested_line_y2 = int(self.im_height * self.roi2)
        interested_line_x = int(self.im_width * 0.5)
        
        if self.rgb is not None and gs.show_draw:
            cv2.line(self.rgb, (interested_line_x, 0),
                    (interested_line_x, self.im_height), (255, 0, 255), 2)
                    
            cv2.line(self.rgb, (0, interested_line_y),
                    (self.im_width, interested_line_y), (0, 0, 255), 2)
            
            cv2.line(self.rgb, (0, interested_line_y2),
                    (self.im_width, interested_line_y2), (0, 0, 255), 2)     
            
        interested_line = self.img_birdview[interested_line_y, :]
        interested_line2 = self.img_birdview[interested_line_y2, :]
        
        # interested_mid = self.img_birdview[:, interested_line_x]

        # Define a helper function for finding the left and right points
        def find_point(interested_line):
            left_point, right_point = -1, -1

            # Search for left point
            for x in range(self.center, 0, -1):
                if interested_line[x] > 0:
                    left_point = x
                    break

            # Search for right point
            for x in range(self.center + 1, self.im_width):
                if interested_line[x] > 0:
                    right_point = x
                    break

            return left_point, right_point

        # Optimize the search for interested_line
        self.left_point, self.right_point = find_point(interested_line)
        self.haveLeft = 1 if self.left_point != -1 else 0
        self.haveRight = 1 if self.right_point != -1 else 0

        # Optimize the search for interested_line2
        self.left_point2, self.right_point2 = find_point(interested_line2)
        self.haveLeft2 = 1 if self.left_point2 != -1 else 0
        self.haveRight2 = 1 if self.right_point2 != -1 else 0
        
        
        # if (self.haveLeft != 0 and self.haveRight !=0):
        #     self.len_line = 2
        # elif (self.haveLeft == 0 and self.haveRight ==0):
        #     self.len_line = 0
        # else:
        #     self.len_line = 1
         
        self.len_line = 2 if self.haveLeft != 0 and self.haveRight != 0 else (0 if self.haveLeft == 0 and self.haveRight == 0 else 1)

        if gs.show_birdview:
            cv2.imshow("img_birdview", self.img_birdview)
            cv2.waitKey(1)
            
        # ============================================================================
        # print("leftpoin 1: ", self.left_point)
        # print("rightpoint 1: ", self.right_point)
        # print("leftpoin 2: ", self.left_point2)
        # print("rightpoint 2: ", self.right_point)
        
        # if abs(self.left_point - self.right_point2) < 30:
        #     self.right_point = self.left_point
        #     self.left_point = -1
        # elif abs(self.right_point - self.left_point2) < 30:
        #     self.left_point = self.right_point 
        #     self.right_point = -1
        
        # When it have error make two point too near   
        # if self.left_point != -1 and self.right_point != -1:          
        #     if (abs(self.right_point-self.left_point)<30):
        #         print("chummm")
        #         self.left_point = 130
        #         self.right_point = 185

        # # Predict right point when only see the left point
        # if self.left_point != -1 and self.right_point == -1:
        #     if (self.left_point > 120 ) and (self.left_point < 160):
        #         print("one left -1 ")
        #         self.right_point = self.left_point + 130
        #     else:
        #         print("one left - 2")
        #         self.right_point = self.left_point + 130

        # # Predict left point when only see the right point
        # if self.right_point != -1 and self.left_point == -1:
        #     if (self.right_point >150) and (self.right_point < 230):
        #         self.left_point = self.right_point - 130
        #         print("one right -1")
        #     else:
        #         self.left_point = self.right_point - 130
        #         print("one right -2")
        
        # Predict right point when only see the left point
        if self.left_point != -1 and self.right_point == -1:
            self.right_point = self.left_point + 60
            print("-------one left-----")
           

        # Predict left point when only see the right point
        if self.right_point != -1 and self.left_point == -1:
            self.left_point = self.right_point - 60
            print("-------one right-----")
        # =====================================================================================
        
        # Draw everything to rgb birdview image
        if self.rgb is not None and gs.show_draw:
            if self.left_point != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.left_point, interested_line_y), 5, (255, 255, 0), -1)
            if self.right_point != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.right_point, interested_line_y), 5, (0, 255, 0), -1)
            if self.left_point2 != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.left_point2, interested_line_y2), 5, (255, 255, 0), -1)
            if self.right_point2 != -1:
                self.rgb = cv2.circle(
                    self.rgb, (self.right_point2, interested_line_y2), 5, (0, 255, 0), -1)
        if gs.show_draw:
            cv2.imshow('Result', self.rgb)
            cv2.waitKey(1)
        
        
    def calculate_control_signal(self):
        
        # Find left/right points  
        self.find_left_right_points()            
            
        if self.left_point != -1 and self.right_point != -1:
            # if abs(self.left_point - self.right_point ) >= 130:      
            #     # print("+++ 2 line to 3 line ++++")
            #     self.right_point += 20  
                
            middle_point = (self.right_point + self.left_point) // 2
            
            x_offset = self.center - middle_point
            
            self.pid_controller.update(x_offset)
            
            # Get the calculated steering angle from the PID controller
            steering_angle_pid = self.pid_controller.output
            
            # Normalize the steering angle to the range -1 to 1
            steering_angle_normalized = -float(steering_angle_pid) 
            if (gs.emergency_stop == True):
                self.pid_controller.clear_stop()
                # print("******** clear PID")
        else:
            steering_angle_normalized = 0

        self.steering_angle = steering_angle_normalized
        self.throttle = self.param.maxThrotle
    
    def clear_countTurn1(self):
        self.countTurnLeft1 = 0
        self.countTurnRight1 = 0
    
    def clear_countTurn2(self):
        self.countTurnLeft2 = 0
        self.countTurnRight2 = 0
    
    def clear_sign(self):
        self.countSignLeft = 0      
        self.countSignRight = 0
        self.countSignStop = 0
        
    def clear_decision(self):
        self.countDecisionThree = 0
        self.countDecisionLeft = 0
        self.countDecisionRight = 0
        self.countDecisionXuyen = 0
        self.countDecisionStraight = 0
    
        
    def final_decision_control(self, rgb, mask):
        self.rgb = rgb
        self.image = mask
        # To get PID steering
        self.calculate_control_signal()
        # ================================SIGN=========================================
        # To Get Sign 
        signs_value = gs.signs[:]
        if signs_value != [] and self.lastSignDetection == '':
            for sign in signs_value:
                # if sign == 'left':
                #     self.countSignLeft += 1
                # elif sign == 'right':
                #     self.countSignRight += 1
                if sign == 'stop' and self.lastSign != 'stop':
                    self.countSignStop += 1       
        
        if self.countSignLeft >= self.param.maxCountSign:
            self.lastSignDetection = 'left'
            self.seeSignTime = time.time()
            self.clear_sign()
        elif self.countSignRight >= self.param.maxCountSign:
            self.lastSignDetection = 'right'
            self.seeSignTime = time.time()
            self.clear_sign()
        elif self.countSignStop >= self.param.maxCountSign:
            self.lastSignDetection = 'stop'
            self.seeSignTime = time.time()
            self.clear_sign()
                
        print("Last_Sign_Detection:",self.lastSignDetection)
        # ==============================CLASSIFY======================================
        # To Get decision classify
        self.decisionclass = gs.decision_class
        if self.lastSignDetection == '' and self.decisionclass != None and self.turning_time == 0:
            # if self.decisionclass == 0: # Nga ba
            #     # self.countDecisionThree += 1
            #     self.countDecisionLeft = 0
            #     self.countDecisionRight = 0
            #     self.countDecisionXuyen = 0
            #     self.countDecisionStraight = 0
            if self.decisionclass == 1: # LEFT
                self.countDecisionLeft += 1
                self.countDecisionThree = 0
                self.countDecisionRight = 0
                self.countDecisionXuyen = 0
                self.countDecisionStraight = 0
            elif self.decisionclass == 4: # RIGHT
                self.countDecisionRight += 1
                self.countDecisionThree = 0
                self.countDecisionXuyen = 0
                self.countDecisionStraight = 0
                self.countDecisionLeft = 0
            elif  self.decisionclass == 5: # STRAIGHT
                self.countDecisionStraight += 1
                self.countDecisionRight = 0
                self.countDecisionThree = 0
                self.countDecisionXuyen = 0
                self.countDecisionLeft = 0
            # elif self.decisionclass == 2 or self.decisionclass == 3: # Vong Xuyen
            #     # self.countDecisionXuyen += 1
            #     self.countDecisionRight = 0
            #     self.countDecisionThree = 0
            #     self.countDecisionXuyen = 0
            #     self.countDecisionLeft = 0
            #     self.countDecisionStraight = 0
    
        # if see the sign will decreace the throttle
        # if self.steering_angle != 0 and self.lastSignDetection != '' and self.turning_time == 0:  
        #     self.throttle = self.param.minThrottle 
           
        # will go ahead when see the sign when it don't see one or two line
        # if self.lastSignDetection != '' and self.turning_time == 0 and (self.haveLeft == 0 or self.haveRight ==0) and signs_value ==[]:
        #     self.steering_angle = self.param.steering # 0
        # self.lastSignDetection = 'right'
        # ====================================END====================================================
        # ==============================RIGHT TRAFFIC============================
        # Count the time not have right point
        # if (self.haveLeft == 1 and self.haveRight == 0  and self.turning_time == 0 and self.lastSignDetection == ''):
        #     self.countTurnRight1 += 1
        #     # print("++++++++++++++=countTurnRight", self.countTurnRight)
        # else:
        #     self.countTurnRight1 = 0       
            
        # # Set turning time for right -------- traffsign
        # if  (self.countTurnRight1 >= self.param.maxCountTurnRight and self.turning_time == 0 \
        # and self.lastSignDetection == 'right' and signs_value ==[] and self.haveRight ==0):
        #     self.turning_time = self.param.maxTurnTimeSign
        #     self.last_sign_time = time.time()
        #     self.turnStatus = 1
        #     self.clear_sign()
        #     self.clear_countTurn1()
        #     print("******************turn right 1***************** ")
        # ----------------------------RIGHT CLASSIFY--------------------------------- 
        # if (self.haveLeft == 1 and self.haveRight == 0  and self.turning_time == 0 and self.lastSignDetection == ''):
        #     self.countTurnRight2 += 1
        #     # print("++++++++++++++=countTurnRight", self.countTurnRight)
        # else:
        #     self.countTurnRight2 = 0  
            
        # if (self.countDecisionRight >= self.param.maxCountDecision and self.countTurnRight2 >= self.param.maxCountTurnRight \
        # and self.turning_time == 0 and self.lastSignDetection == ''):
            
        #     self.turning_time = self.param.maxTurnTime90
        #     self.lastSignDetection = 'right'
        #     self.last_sign_time = time.time()
        #     self.turnStatus = 2
            
        #     self.clear_decision()
        #     self.clear_countTurn2()
        #     print("*******************turn right 2************")

        # ============================= LEFT SIGN=====================================
        # Count the time not have right point
        # if (self.haveLeft == 1 and self.haveRight == 0  and self.turning_time == 0 and self.lastSignDetection == ''):
        #     self.countTurnLeft1 += 1
        #     # print("++++++++++++++=countTurnRight", self.countTurnRight)
        # else:
        #     self.countTurnLeft1 = 0
            
        # # set turning time for left -------- traffsign
        # if  (self.turning_time == 0 and self.lastSignDetection == 'left' and signs_value ==[] and self.haveLeft ==0):
        #     self.turning_time = self.param.maxTurnTimeSign
        #     self.last_sign_time = time.time()
        #     self.turnStatus = 1
        #     print("******************turn left 1 *******************")
        
        # -----------------------------LEFT CLASSIFY-------------------------------- 
        #Count the time not have left point
        if (self.haveLeft == 0 and self.haveRight == 1 and self.turning_time == 0 and self.lastSignDetection == ''):
            self.countTurnLeft2 += 1
            # print("++++++++++++++=CountTurnLeft", self.CountTurnLeft)
        else:
            self.countTurnLeft2 = 0
            
        if (self.countDecisionLeft >= self.param.maxCountDecision and self.countTurnLeft2 >= self.param.maxCountTurnLeft \
        and self.turning_time == 0 and self.lastSignDetection == ''):
            self.turning_time = self.param.maxTurnTime90
            self.lastSignDetection = 'left'
            self.last_sign_time = time.time()
            self.turnStatus = 2
            
            self.clear_decision()
            self.clear_countTurn2()
            print("*******************turn left 2 **************************")
        
        # ===============================STOP===============================
        # set turning time for stop sign imediately
        if self.lastSignDetection == 'stop' and self.turning_time == 0:
            self.turning_time = self.param.stoptime
            self.last_sign_time = time.time()
            self.turnStatus = 1
            print("************************ stop *************************")
        # ================================Executing======================================
        # if (self.countDecisionStraight >= 2):
        #     self.steering_angle = 0
        
        # if have turnning time and last sign detection
        if (time.time() - self.last_sign_time) >= 0 and (time.time() - self.last_sign_time) <= self.turning_time and self.lastSignDetection != '':         
            if self.turnStatus == 1: # Turning for sign
                # if self.lastSignDetection == 'left':
                #     self.steering_angle = -50
                # elif self.lastSignDetection == 'right':
                #     self.steering_angle = 50
                if self.lastSignDetection == 'stop':
                    self.throttle = 0
                    self.steering_angle = 0
            elif self.turnStatus == 2: # Turing for 90*
                if self.lastSignDetection == 'left':
                    self.steering_angle = -30
                    self.clear_decision()               
                elif self.lastSignDetection == 'right':
                    self.steering_angle = 30
                    self.clear_decision()
                    
            # clear all when have two line
            if (self.len_line >= 2) and (time.time() - self.last_sign_time) >= self.param.minTurnTime90:
                self.turning_time = 0
                self.lastSign = self.lastSignDetection
                self.lastSignDetection = ''
                self.seeSignTime = 0
                self.last_sign_time = 0
                self.countDecision = 0
                self.haveObject = 0
                self.turnStatus = 0
                self.clear_decision()
                self.clear_countTurn1()
                self.clear_countTurn2()
                self.clear_sign()
                print("clear all 1")
        # clear all when out of time
        elif ((time.time() - self.last_sign_time) < 0 or (time.time() - self.last_sign_time) >= self.turning_time) and self.lastSignDetection != '' and self.turning_time != 0 and self.len_line == 2 :
            self.turning_time = 0
            self.lastSign = self.lastSignDetection
            self.lastSignDetection = ''
            self.seeSignTime = 0
            self.last_sign_time = 0
            self.haveObject = 0
            self.turnStatus = 0
            self.countDecision = 0
            self.clear_decision()
            self.clear_countTurn1()
            self.clear_countTurn2()
            self.clear_sign()
            print("clear all 2")
        
        # Reset when see sign but don't turn
        if (self.lastSignDetection != '' and self.turning_time == 0 and (time.time() - self.seeSignTime) >= 4) or gs.emergency_stop == True:
            self.turning_time = 0
            self.lastSign = self.lastSignDetection
            self.lastSignDetection = ''
            self.seeSignTime = 0
            self.last_sign_time = 0
            self.haveObject = 0
            self.turnStatus = 0
            self.countDecision = 0
            self.clear_countTurn1()
            self.clear_countTurn2()
            self.clear_decision()
            self.clear_sign()
            print("New round")    


    def run(self):
        start_time = time.time()
        frame_count = 0
        while not gs.exit_signal:
            try:
                # mask = gs.current_img
                mask = get_fast(gs.mask_img)
                rgb =  get_fast(gs.rgb_frames)
                
                if (gs.show_rgb):
                    cv2.imshow("rgb", rgb)
                    cv2.waitKey(1)
                
                if gs.show_mask:
                    cv2.imshow('Result', mask)
                    cv2.waitKey(1)
            
                self.final_decision_control(rgb, mask)
                # Lưu ý đừng cho số lớn hơn 630
                # Quá cao sẽ hư 
                if self.throttle != 0:
                    gs.speed = 10
                    cf.THROTTLE_NEUTRAL = 620
                else:
                    gs.speed = 0
                    cf.THROTTLE_NEUTRAL = 614
                self.steering_angle = max(-60, min(60, self.steering_angle))
                # gs.steer = gs.e2e_steering
                gs.steer = self.steering_angle
                print("Throtle: " + str(gs.speed) + " - Steering: " + str(gs.steer))
                
                frame_count += 1
                # Calculate fps every second
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # Check if one second has passed
                    fps = frame_count / elapsed_time
                    gs.fps = fps
                    print(f"FPS: {fps:.2f}")
                    # Reset counters for the next second
                    start_time = time.time()
                    frame_count = 0
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error) # An exception occurred:
                traceback.print_exc()
                continue

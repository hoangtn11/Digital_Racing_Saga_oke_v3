#!/usr/bin/env python3
import Jetson.GPIO as  gpio
import sys
import config as cf
import time
import threading
import global_storage as gs

class CarGuard(threading.Thread):

    def __init__(self, stop_on_distance_sensor_covered=True, stop_on_button_2=True, start_on_button_1 = True, stop_duration=None):
        threading.Thread.__init__(self)
        self.stop_on_distance_sensor_covered = stop_on_distance_sensor_covered
        self.stop_on_button_2 = stop_on_button_2
        self.start_on_button_1 = start_on_button_1
        self.stop_duration = stop_duration
        self.stop_time_begin = time.time()

    def run(self):
        while not gs.exit_signal:
            condition1 = gs.button_ss1 and self.stop_on_distance_sensor_covered
            condition2 = gs.button_2 and self.stop_on_button_2  
            condition3 = gs.button_1 and self.start_on_button_1
            if condition1 or condition2:
                gs.emergency_stop = True
                gs.speed = 0
                self.stop_time_begin = time.time()
                # print("Mem mmm")
            elif condition3:
                gs.emergency_stop = False
                # print("Mem mmm dasdasd")
            if self.stop_duration is not None:
                if time.time() - self.stop_time_begin > self.stop_duration:
                    gs.emergency_stop = False
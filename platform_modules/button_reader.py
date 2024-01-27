#!/usr/bin/env python3
import Jetson.GPIO as  gpio
import RPi.GPIO as rpigpio
import sys
import config as cf
import time
import threading
import global_storage as gs
import signal
import os

class ButtonReader(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        # Set the GPIO pin numbering mode
        gpio.setmode(gpio.BCM)  # or gpio.BCM, depending on your preference
        print("mode: ",gpio.getmode())
        print("jetson infor: ", gpio.JETSON_INFO)
        print("version: ", gpio.VERSION)
        while True:
            val = os.system('sudo chmod -R 777 /sys/class/gpio/gpio*')    
            gpio.setup(cf.GPIO_BUTTON_1, gpio.IN) #Stop
            gpio.setup(cf.GPIO_BUTTON_2, gpio.IN) #Start
            # gpio.setup(cf.GPIO_BUTTON_3, gpio.IN)
            # gpio.setup(cf.GPIO_BUTTON_4, gpio.IN)
            # Pin Setup:
            rpigpio.setmode(rpigpio.BCM)  # BCM pin-numbering scheme from Raspberry Pi
            rpigpio.setup(cf.GPIO_SENSOR_1, rpigpio.IN)  # set pin as an input pin
            # gpio.setup(cf.GPIO_BUTTON_SS1, gpio.IN)
            # gpio.setup(cf.GPIO_BUTTON_SS2, gpio.IN)
            # gpio.setup(cf.GPIO_LED, gpio.OUT)
            break

    def run(self):
        prev_value = None
        while not gs.exit_signal:
            gs.button_1 = gpio.input(cf.GPIO_BUTTON_1)  # Use gpio.input() instead of gpio.read()
            gs.button_2 = gpio.input(cf.GPIO_BUTTON_2)
            # gs.button_3 = gpio.input(cf.GPIO_BUTTON_3)
            # gs.button_4 = gpio.input(cf.GPIO_BUTTON_4)
            value = rpigpio.input(cf.GPIO_SENSOR_1)
            if value != prev_value:
                if value == rpigpio.HIGH:
                    gs.button_ss1 = False
                else:
                    gs.button_ss1 = True
                prev_value = value
            # gs.button_ss1 = gpio.input(cf.GPIO_BUTTON_SS1)
            # gs.button_ss2 = gpio.input(cf.GPIO_BUTTON_SS2)
            # gpio.output(cf.GPIO_LED, gs.led_state)
            time.sleep(0.1)
        self.clean_up_gpio(None, None)
        print("Exiting from ButtonReader")

    def clean_up_gpio(self, num, stack):
        print("Clean up GPIO...")
        gpio.cleanup(cf.GPIO_BUTTON_1)
        gpio.cleanup(cf.GPIO_BUTTON_2)
        # gpio.cleanup(cf.GPIO_BUTTON_3)
        # gpio.cleanup(cf.GPIO_BUTTON_4)
        rpigpio.cleanup()
        # gpio.cleanup(cf.GPIO_BUTTON_SS1)
        # gpio.cleanup(cf.GPIO_BUTTON_SS2)
        exit(0)
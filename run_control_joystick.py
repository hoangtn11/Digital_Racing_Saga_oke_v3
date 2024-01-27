import signal
import sys
import time
import _thread
from platform_modules.motor_controller import MotorController
from platform_modules.lcd_driver import LCD
from platform_modules.button_reader import ButtonReader
from platform_modules.car_guard import CarGuard
from platform_modules.camera import Camera
from utils.keyboard_getch import _GetJoystick
from platform_modules.lcd_display import LCDDisplay
import global_storage as gs
import config as cf

# LCD show status
lcd_display = LCDDisplay()
lcd_display.start()

camera = Camera()
camera.start()

# Init motor controller
motor_controller = MotorController()
motor_controller.start()

# Init button reader
button_reader = ButtonReader()
button_reader.start()

# Car guard
# Stop car when hitting obstacle or when user presses button 4
guard = CarGuard()
guard.start()

# Manual control using keyboard
getch = _GetJoystick()
print("Use Joystick to control: wasd")
print("Quit: q")
while not gs.exit_signal:
    # List of joystick states
    joystick_states = getch()  # Assuming you receive a list of joystick states
    if joystick_states != []:
        for key in joystick_states:
            if key == "w":
                if gs.speed < 0:
                    # gs.speed = 0
                    gs.speed += 2
                    cf.THROTTLE_NEUTRAL = 614
                else:
                    gs.speed = min(cf.MAX_SPEED, gs.speed + 2)
                    cf.THROTTLE_NEUTRAL = 620
            elif key == "s":
                if gs.speed > 0:
                    gs.speed -= 2
                    cf.THROTTLE_NEUTRAL = 614
                else:
                    gs.speed = max(-cf.MAX_SPEED, gs.speed - 2)
                    cf.THROTTLE_NEUTRAL = 620
            elif key == "a":
                if gs.steer > 0:
                    gs.steer = 0
                else:
                    gs.steer = max(cf.MIN_ANGLE, gs.steer - 7)
            elif key == "d":
                if gs.steer < 0:
                    gs.steer = 0
                else:
                    gs.steer = min(cf.MAX_ANGLE, gs.steer + 7)
            elif key == "i":  # Remove emergency stop state
                print("Emergency stop state: ",gs.emergency_stop)
                gs.emergency_stop = True
                time.sleep(0.5)
            elif key == "x":  # Remove emergency stop state
                print("Emergency stop state: ",gs.emergency_stop)
                gs.emergency_stop = False
                time.sleep(0.5)
            elif key == "v":
                print("Record video: " + str(gs.record_videos))
                gs.record_videos = not gs.record_videos
                time.sleep(1)
            elif key == "q":
                gs.exit_signal = True
                break
            print("Speed: {}  Steer: {} THROTTLE_NEUTRAL: {} emergency_stop: {}".format(gs.speed, gs.steer, cf.THROTTLE_NEUTRAL, gs.emergency_stop))
        getch.listKeys = []
    else:
        gs.steer = 0

camera.join()
motor_controller.join()
guard.join()
button_reader.join()
lcd_display.join()
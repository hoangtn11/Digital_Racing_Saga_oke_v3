#!/usr/bin/env python3

import threading
import global_storage as gs
from platform_modules.lcd_driver import LCD
import config
import time
import config as cf

class LCDDisplay(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.lcd = LCD(config.LCD_ADDRESS)
    
    def run(self):
        while not gs.exit_signal:            
            self.lcd.lcd_clear()
            self.lcd.lcd_display_string("State: " + ("Running" if not gs.emergency_stop else "Stop"), 1)
            self.lcd.lcd_display_string(f"FPS: {gs.fps:.2f}", 2)
            if gs.record_videos:
                self.lcd.lcd_display_string("Recording", 3)
            time.sleep(10)

    

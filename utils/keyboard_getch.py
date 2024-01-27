class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

class _GetJoystick:
    def __init__(self):
        import pygame
        self.listKeys = []

    def __call__(self):
        import pygame
        pygame.init()
        pygame.joystick.init()

        # Initialize joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        # print(f"Joystick: {joystick.get_name()}")

        try:
            pygame.event.pump()

            # Read joystick axes
            # left_stick_x = joystick.get_axis(0)
            left_stick_y = joystick.get_axis(1)
            
            if left_stick_y <= -0.9 :
                # return 'w'
                self.listKeys.append('w')
            elif left_stick_y >= 0.9:
                # return 's'
                self.listKeys.append('s')
                
            
            right_stick_x = joystick.get_axis(2)
            if right_stick_x <= -0.9 :
                # return 'a'
                self.listKeys.append('a')
            elif right_stick_x >= 0.9:
                # return 'd'
                self.listKeys.append('d')
            
            # right_stick_y = joystick.get_axis(3)

            # Map joystick axes to control values
            # forward_backward = -left_stick_y  # Invert for intuitive control
            # steering = right_stick_x

            # Read joystick buttons
            button_A = joystick.get_button(0) # A
            if button_A >= 0.95:
                self.listKeys.append('q')
                
            button_B = joystick.get_button(1) # B
            if button_B >= 0.95:
                self.listKeys.append('x')
                
            button_Y = joystick.get_button(4) # Y
            if button_Y >= 0.95:
                self.listKeys.append('i')
                
            button_X = joystick.get_button(3) # X
            if button_X >= 0.95:
                self.listKeys.append('v')
            # Add more button indices as needed

            # print(f"Forward/Backward: {forward_backward:.2f}, Steering: {steering:.2f}")
            # print(f"Button A: {button_A}, Button B: {button_B}, Button X: {button_X}, Button Y: {button_Y}")
            
            # time.sleep(0.1)  # Add a delay to avoid excessive output

        except KeyboardInterrupt:
            pass
        finally:
            joystick.quit()
        return self.listKeys

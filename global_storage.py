import queue

# Set this to True to exit
exit_signal = False

haveObject = False

#FPS on LCD
fps = 0

# Images from camera
rgb_frames = queue.Queue(5)
rgb_seg_frames = queue.Queue(5)
rgb_sign_frame = queue.Queue(5)

dc_images = queue.Queue(5)

# depth_frames = queue.Queue(5)

current_img = None
mask_img = queue.Queue(5)
record_videos = False

# 6 is straight
decision_class = 6

e2e_images = queue.Queue(5)
e2e_steering = 0


# Emergency STOP
# When the car hits obstacle or when
# user presses emergency stop button
# following value will be set to True
# causing car to brake
emergency_stop = True

show_rgb = False
show_draw = False
show_control = False
show_mask = False
show_trafficSign = False
show_Object = False
show_birdview = False
show_hsvmask = False

signs = []

# Motor states
pause = False
speed = 0
steer = 0

# Buttons
button_1 = False
button_2 = False
# button_3 = False
# button_4 = False
button_ss1 = False
# button_ss2 = False

# Remote control sensor
last_control_signal_time = 0
remote_control_max_speed = 100
remote_control_max_steer_angle = 150
remote_control_speed = 0
remote_control_steer_angle = 0

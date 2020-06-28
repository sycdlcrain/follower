# ~ # USE sudo python3 blink_smooth.py
# ~ import RPi.GPIO as GPIO
# ~ import numpy as np
# ~ import time
# ~ import matplotlib.pyplot as plt

# ~ GPIO.setwarnings(False)
# ~ GPIO.setmode(GPIO.BCM)
# ~ pin_num = 18
# ~ freq = 100 # don't see blinking, for some reason faster frequencies are worse
# ~ percent_on = 100
# ~ count = -1
# ~ step = 1
# ~ GPIO.setup(pin_num, GPIO.OUT)
# ~ p = GPIO.PWM(pin_num , freq)
# ~ p.start(percent_on)

# ~ motor_num = 13
# ~ motor_d1 = 19
# ~ motor_d2 = 26
# ~ GPIO.setup(motor_num, GPIO.OUT)
# ~ motor_freq = 100
# ~ mpwm = GPIO.PWM(motor_num , motor_freq)
# ~ mpwm.start(100)
# ~ GPIO.setup(motor_d1, GPIO.OUT)
# ~ GPIO.setup(motor_d2, GPIO.OUT)

STEERING_PIN = 23
STEERING_FREQ = 100 # don't see blinking, for some reason faster frequencies are worse

MOTOR_PIN = 13
MOTOR_D1 = 19
MOTOR_D2 = 26
MOTOR_FREQ = 100

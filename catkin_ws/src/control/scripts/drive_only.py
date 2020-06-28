# USE sudo python3 blink_smooth.py
import RPi.GPIO as GPIO
import numpy as np
import time
import matplotlib.pyplot as plt

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
pin_num = 18
freq = 100 # don't see blinking, for some reason faster frequencies are worse
percent_on = 100
count = -1
step = 1
GPIO.setup(pin_num, GPIO.OUT)

p = GPIO.PWM(pin_num , freq)
p.start(percent_on)
#GPIO.output(pin_num, GPIO.LOW)

motor_num = 13
motor_d1 = 19
motor_d2 = 26
GPIO.setup(motor_num, GPIO.OUT)
motor_freq = 100
mpwm = GPIO.PWM(motor_num , motor_freq)
mpwm.start(100)
GPIO.setup(motor_d1, GPIO.OUT)
GPIO.setup(motor_d2, GPIO.OUT)

buffer = np.zeros(100)

while 1==1:

	count+=step
	if count>=100:#np.size(buffer):
		count=1
	percent_on = count+0.0
		
	negone_to_one = np.sin((percent_on+0.0)/100*2*np.pi)
	zero_to_one = negone_to_one/2.0 + 0.5

	# VISUALIZE
	VISUALIZE = 0
	if VISUALIZE:
		buffer[count] = zero_to_one

		plt.clf()
		plt.plot(range(np.size(buffer)), buffer,'.')
		plt.show(block=False)
		plt.pause(0.00001)


	#p.ChangeDutyCycle(zero_to_one*100)    #change duty cycle for varying the brightness of LED.
	p.ChangeDutyCycle(zero_to_one*30) 
	print(zero_to_one*30)
	#print(count)
	#p.ChangeDutyCycle(count)


	#GPIO.output(motor_num, True)
	GPIO.output(motor_d1, False)
	GPIO.output(motor_d2, True)    
	mpwm.ChangeDutyCycle(zero_to_one*100) 


	time.sleep(0.03)                       #sleep for 100m second
	
GPIO.cleanup()


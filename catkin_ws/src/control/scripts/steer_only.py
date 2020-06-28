# USE sudo python3 blink_smooth.py
import RPi.GPIO as GPIO
import numpy as np
import time
import matplotlib.pyplot as plt

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
pin_num = 23
freq = 200 # don't see blinking, for some reason faster frequencies are worse
percent_on = 100
count = -1
step = 1
GPIO.setup(pin_num, GPIO.OUT)

p = GPIO.PWM(pin_num , freq)
p.start(percent_on)
#GPIO.output(pin_num, GPIO.LOW)


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
	num = 15+zero_to_one*20
	p.ChangeDutyCycle(num )
	print(num)
	#print(count)
	#p.ChangeDutyCycle(count)

	time.sleep(0.5)                       #sleep for 100m second
	
GPIO.cleanup()


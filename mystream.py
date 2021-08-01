#project is based more on image processing rather than machine learning
#will do image processing on the image captured from the webcam.

import numpy as np 
import cv2
import time

#creating a video capture object
cap=cv2.VideoCapture(0)

#to allow the camera to calibrate itself in the initial 3 secs
time.sleep(3)

#provision to capture the bacground behind the cloak
background=0


#capture background
for i in  range(60): #iterations to capture the background multiple times
	
	retr,background=cap.read() #retr is like a staus variable returning true on success


#execute till the webcapturing window is closed.
while (cap.isOpened()):

	retr, image=cap.read()

	if not retr:
		break



	#convert from BGR to HSV.
	#only the hue will depend on the brightness.
	#this improves the color perception for image proccessing in our favour 

	hsvColor=cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #hue saturation value/brightness #HSV/HSB

	#color range for the cloak, here red.
	red_lower=np.array([0,120, 70])
	red_upper=np.array([10,255,255])

	#develop the mask
	mask=cv2.inRange(hsvColor,red_lower,red_upper)


	red_lower=np.array([170,120,70])
	red_upper=np.array([180,255,255])
	mask2= cv2.inRange(hsvColor,red_lower,red_upper)

	mask=mask+mask2 #plus acts as an OR operator #shades of red are segmented into mask

	#apply morphology funcions 
	# ref: https://docs.opencv.org/4.5.2/d9/d61/tutorial_py_morphological_ops.html

	#Removing Noise from mask
	mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2)

	#increasing smoothness of image using MORPH_DILATE
	mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8),iterations=1)

	#everything except the cloak
	mask2=cv2.bitwise_not(mask)

	#segmentation of color
	result1=cv2.bitwise_and(background,background,mask=mask)

	#substitiute the cloak part.
	result2=cv2.bitwise_and(image,image,mask=mask2)

	#final output
	final=cv2.addWeighted(result1,1,result2,1,0) #linearly add two images
	cv2.imshow('Eureka !!', final)

	#to close the window
	keyPressed=cv2.waitKey(1) & 0xFF
	if keyPressed== ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

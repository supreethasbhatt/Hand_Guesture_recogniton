#from dlib import imutils
import numpy as np
from skimage import measure
import cv2

im = cv2.imread('img2.jpg')
im = cv2.resize(im, (320,320))
frame = im
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

#frame = imutils.resize(im, width = 400)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(hsv, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
# blur the mask to help remove noise, then apply the
# mask to the frame
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(frame, frame, mask = skinMask)
r, newim = cv2.threshold(skin, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("images", np.hstack([frame, skin]))
#cv2.imshow("skin binary", newim)

'''labels = measure.label(newim, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
 
# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 3000:
		mask = cv2.add(mask, labelMask)'''

skinMask = cv2.cvtColor(skinMask, cv2.COLOR_GRAY2BGR)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(imgray, (11, 11), 0)
ret,thresh = cv2.threshold(blurred,100,255,cv2.THRESH_BINARY)
#cv2.imshow('blurred', blurred)
cv2.imshow('thresh', thresh)
# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
cv2.imshow('goodthresh', thresh)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
max_area = 0
for i in range(len(contours)):
	cnt=contours[i]
	area = cv2.contourArea(cnt)
	if(area>max_area):
		max_area=area
		ci=i

cnt=contours[ci]
cv2.drawContours(im, [cnt], 0, (0,255,0), 3)
cv2.imshow('contours', im)

cv2.waitKey(0)
cv2.destroyAllWindows()

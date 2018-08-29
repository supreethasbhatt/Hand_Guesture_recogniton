# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:08:46 2018

@author: divya
"""

import cv2
import cv2.bgsegm
import numpy as np
import copy
import math

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

#fgbg = cv2.createBackgroundSubtractorMOG()
i=0
while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 2, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)


    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        #cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        #cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('ori', thresh)
	cv2.imwrite('pic{}.jpg'.format(i), thresh)
	i+=1

    
    # Keyboard OP
    k = cv2.waitKey(30)
    if k & 0xff == 27:  # press ESC to exit
	print('Here')
        camera.release()
	cv2.destroyAllWindows()
	exit(0)
    elif k & 0xff == ord('b'):  # press 'b' to capture the background
	print('Here')
        bgModel = cv2.bgsegm.createBackgroundSubtractorMOG()
        isBgCaptured = 1
        print '!!!Background Captured!!!'
    elif k & 0xff == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print '!!!Reset BackGround!!!'
    elif k & 0xff == ord('t'):
        triggerSwitch = True
        print '!!!Trigger On!!!'




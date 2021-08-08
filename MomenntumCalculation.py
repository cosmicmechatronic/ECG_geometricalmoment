import cv2
import numpy as np
#import dlib
from matplotlib import pyplot as plt
#import imutils
import os, sys
import glob
import time
import pickle

dtype=np.float32

class MomenntumCalculation():
    def __init__(self, file):
        self.file = file



    def Calculate_moment_Cx(self):

        img = cv2.imread(self.file)
        g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(g, 60, 180)

        #fig, ax = plt.subplots(1, figsize=(12,8))
        #plt.imshow(edge, cmap='Greys')
        #plt.show()
        #plt.close('all')

        contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (0, 255, 0), 5)

        #fig, ax = plt.subplots(1, figsize=(8, 6))
        #plt.imshow(img)
        #plt.show()
        #plt.close('all')

        ret,thresh = cv2.threshold(img,127,255,0)
        #im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)

        try:
            cnt = contours[0]
            M = cv2.moments(cnt)

            try:
                cx = int(M['m10'] / M['m00'])
                #print("Cx: ", cx)
                os.remove(f"{self.file}")


                return cx

            except ZeroDivisionError:
                print("Error of momment calculation")

                cx = 0
                print("Cx: ", cx)
                os.remove(f"{self.file}")
                return cx

        except IndexError:
            gotdata = 'null'

            cx = 0
            #print("Cx: ", cx)
            os.remove(f"{self.file}")
            return cx


        #print("momenty:", M )

        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])









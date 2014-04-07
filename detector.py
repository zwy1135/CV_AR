# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 21:38:18 2014

@author: wy
"""

import cv2
#import scipy.misc as mc
#import pylab as pl
#import time
FLANN_INDEX_LSH = 6
detectorMap = {"sift":cv2.SIFT,"surf":cv2.SURF,"orb":cv2.ORB}


class Detector(object):
    '''
    
    '''
    def __init__(self,dtType,keypointNum = None):
        '''
        
        '''
        self.dtType = dtType
        self.func = detectorMap[dtType]
        if keypointNum:
            self.detector = self.func(keypointNum)
        else:
            self.detector = self.func()
        
    def changeNum(self,keypointNum):
        self.detector = self.dtType(keypointNum)
        print "keypoint's num is changed to %d"%keypointNum
        
    def getType(self):
        return self.dtType
        
    @property
    def detect(self):
        return self.detector.detect
        
    @property
    def compute(self):
        return self.detector.compute
        
    
    def detectAndCompute(self,img):
        keypoints = self.detect(img)
        ds= self.compute(img,keypoints)
        
        return keypoints,ds[-1]
        
        
        



if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    dt = Detector("orb")
    cv2.namedWindow("kpimg")

    while True:
        if not cap.isOpened():
            continue
        ret,img = cap.read()
        cv2.waitKey(5)
        
        try:
            if img.size:
                kp,ds = dt.detectAndCompute(img)
                print ds
                kpimg = img
                kpimg = cv2.drawKeypoints(img,kp,kpimg)
                cv2.imshow("kpimg",kpimg)
        except AttributeError:pass












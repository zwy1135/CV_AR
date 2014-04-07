# -*- coding: utf-8 -*-
"""
Created on Mon Apr 07 18:39:34 2014

@author: wy
"""

import cv2
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


class Matcher(object):
    '''
    
    '''
    def __init__(self,mctype = None):
        '''
        
        '''
        self.mctype = mctype
        if mctype == "bf":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck = True)
        elif mctype == "flann_kdtree":
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=100)
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
            search_params = dict(checks=100)
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
            
    def getType(self):
        return self.mctype
        
    @property
    def match(self):
        return self.matcher.match
        
    @property
    def knnMatch(self):
        return self.matcher.knnMatch
        
    
    
    
if __name__=="__main__":
    from detector import Detector
    
    cap = cv2.VideoCapture(0)
    dt = Detector("orb",300)
    mc = Matcher()#"flann_kdtree")
    cv2.namedWindow("mcimg")
    origin = cv2.imread("image/origin.jpg")
    kp_o,ds_o = dt.detectAndCompute(origin)
    while True:
        if not cap.isOpened():
            continue
        ret,img = cap.read()
        cv2.waitKey(5)
        
        try:
            if img.size:
                kp,ds = dt.detectAndCompute(img)
                mcimg = img
                matches = mc.knnMatch(ds_o,ds,k=2)
                
                matches = matches[:100]
                mcimg = cv2.drawMatchesKnn(origin,kp_o,img,kp,matches,mcimg)
                cv2.imshow("mcimg",mcimg)
        except AttributeError:pass
import math
import sys
import traceback
import json
import numpy as np
import cv2

from ImgProc import ImgTask
import VODEvents

DEBUG=False
# designed to work with the "Input Overlay" and "Mouse" OBS sources
class InputAnalyzer(ImgTask.ImgTask):

    def __init__(self, region, thres, act, deact, key_size, **kwargs):
        super().__init__(**kwargs)
        self.y1 = region[0][1]
        self.y2 = region[1][1]
        self.x1 = region[0][0]
        self.x2 = region[1][0]
        self.detect = []
        self.detect.append({
            'actevent' : act,
            'deactevent' : deact,
            'lowseg': np.array(thres[0]),
            'uppseg': np.array(thres[1]),
            'active': thres[2], })
        self.kern = np.ones((key_size,key_size), np.float32)/(key_size**2)# sum all in square neighborhood
        self.state = False
            
    def requires(self):
        return [ImgTask.IMG_BASE]
        
    def outputs(self):
        return [VODEvents.EVENT_NODE]#, ImgTask.IMG_DEBUG]

    # proc_frame signature must match requires / outputs
    def proc_frame(self, frame):
        subimage = frame[self.y1:self.y2, self.x1:self.x2,:]
        
        res = []
        self.dbg_img = None
        new_state = self.state
        for rule in self.detect:
            value = self.threshold(subimage, rule)
            if DEBUG:
                self.dbg_img = value
            if (value.max()>rule['active']):
                new_state = True
            else:
                new_state = False
        if (not self.state) and (new_state):
            res.append({'topic': rule['actevent']})
        elif (self.state) and (not new_state):
            res.append({'topic': rule['deactevent']})
        self.state = new_state
        return res#, self.dbg_img
        
    def threshold(self, img, rule):
        mask = cv2.inRange(img, rule['lowseg'], rule['uppseg']).astype(np.float32)/255. # 0 or 1 only
        return cv2.filter2D(mask,-1,self.kern)
        

keyb = InputAnalyzer(region=[(83, 863),(302, 1002)],
                    thres=[[65, 170, 170], [105, 230, 230], 0.395],
                    act=VODEvents.KEY_ANY_DOWN, deact=VODEvents.KEY_ANY_UP,
                    key_size=55)
                    
mous = InputAnalyzer(region=[(400, 887),(455, 953)],
                    thres=[[65, 170, 170], [105, 230, 230], 0.495],
                    act=VODEvents.MOUSE_LMB_DOWN, deact=VODEvents.MOUSE_LMB_UP,
                    key_size=55)
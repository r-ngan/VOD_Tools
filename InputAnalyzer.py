import math
import sys
import traceback
import json
from pubsub import pub
import numpy as np
import cv2

import VideoAnalysis
import VODEvents
import VODState
from ImgProc import Delta, BotPose # dependency for generation

# designed to work with the "Input Overlay" and "Mouse" OBS sources
class InputAnalyzer(VideoAnalysis.Analyzer):

    def __init__(self, region, thres, key_size, **kwargs):
        super().__init__(**kwargs)
        self.y1 = region[0][1]
        self.y2 = region[1][1]
        self.x1 = region[0][0]
        self.x2 = region[1][0]
        self.detect = []
        for event,thrdata in thres.items():
            self.detect.append({
                'event' : event,
                'lowseg': np.array(thrdata[0]),
                'uppseg': np.array(thrdata[1]),
                'active': thrdata[2], })
        self.kern = np.ones((key_size,key_size), np.float32)/(key_size**2)# sum all in square neighborhood
        
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        super().proc_frame(timestamp=timestamp, img=img, aux_imgs=aux_imgs)
        kbcrop = aux_imgs['delta'][self.y1:self.y2, self.x1:self.x2,:]
        bx,by = BotPose.get_bot_pose(aux_imgs)
        
        for rule in self.detect:
            value = self.threshold(kbcrop, rule)
            if (value.max()>rule['active']):
                pub.sendMessage(rule['event'], timestamp=timestamp,
                                x=bx,y=by,)
        
    def threshold(self, img, rule):
        mask = cv2.inRange(img, rule['lowseg'], rule['uppseg']).astype(np.float32)/255. # 0 or 1 only
        return cv2.filter2D(mask,-1,self.kern)
        

keyb = InputAnalyzer(region=[(83, 863),(302, 1002)], thres={
                    VODEvents.KEY_ANY_DOWN:[[-70, 110, 110], [70, 999, 999], 0.495],
                    VODEvents.KEY_ANY_UP  :[[-90, -999, -999], [70, -110, -110], 0.495] },
                    key_size  = 55)
                    
mous = InputAnalyzer(region=[(400, 887),(455, 953)], thres={
                    VODEvents.MOUSE_LMB_DOWN: [[-70, 110, 110], [70, 999, 999], 0.495],
                    VODEvents.MOUSE_LMB_UP  : [[-90, -999, -999], [70, -110, -110], 0.495] },
                    key_size  = 55)
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
            
    def requires(self):
        return [ImgTask.VAL_FRAMENUM, 'delta', 'bothead_closest']
        
    def outputs(self):
        if DEBUG:
            return [VODEvents.EVENT_NODE, 'debug']
        else:
            return [VODEvents.EVENT_NODE]

    # proc_frame signature must match requires / outputs
    def proc_frame(self, timestamp, delta, bothead):
        subimage = delta[self.y1:self.y2, self.x1:self.x2,:]
        bx,by = bothead
        
        res = []
        dbg_img = None
        for rule in self.detect:
            value = self.threshold(subimage, rule)
            if DEBUG:
                dbg_img = value
            if (value.max()>rule['active']):
                res.append({'topic': rule['event'],
                            'timestamp':timestamp,
                            'x':bx,'y':by,})
        if DEBUG:
            return res, dbg_img
        else:
            return res
        
    def threshold(self, img, rule):
        mask = cv2.inRange(img, rule['lowseg'], rule['uppseg']).astype(np.float32)/255. # 0 or 1 only
        return cv2.filter2D(mask,-1,self.kern)
        

keyb = InputAnalyzer(region=[(83, 863),(302, 1002)], thres={
                    VODEvents.KEY_ANY_DOWN:[[-70, 110, 110], [70, 999, 999], 0.495],
                    VODEvents.KEY_ANY_UP  :[[-90, -999, -999], [70, -90, -90], 0.395] },
                    key_size  = 55)
                    
mous = InputAnalyzer(region=[(400, 887),(455, 953)], thres={
                    VODEvents.MOUSE_LMB_DOWN: [[-70, 110, 110], [70, 999, 999], 0.495],
                    VODEvents.MOUSE_LMB_UP  : [[-90, -999, -999], [70, -110, -110], 0.495] },
                    key_size  = 55)
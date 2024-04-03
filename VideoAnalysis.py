import math
import sys
import traceback
import json
from pubsub import pub
import numpy as np
import cv2

import VODEvents
import VODState


_WIDTH_KEY = 'width'
_HEIGHT_KEY= 'height'
_DEPTH_KEY = 'depth'
_FR_KEY    = 'frame_rate'

dbg_frame = np.array([])

class Analyzer:

    def __init__(self, xdim=0, ydim=0, depth=0):
        self.xdim = 0
        self.ydim = 0
        self.midx = self.xdim//2
        self.midy = self.ydim//2
        self.depth = 0
        self.ms_fr = 1
        self.game_state = VODState.VOD_RESET
        pub.subscribe(self.initialize, VODEvents.VOD_START)
        pub.subscribe(self.proc_frame, VODEvents.VOD_FRAME)
        
    def initialize(self, topic=pub.AUTO_TOPIC, **data):
        if _WIDTH_KEY in data:
            self.xdim = data[_WIDTH_KEY]
            self.midx = data[_WIDTH_KEY]//2
        if _HEIGHT_KEY in data:
            self.ydim = data[_HEIGHT_KEY]
            self.midy = data[_HEIGHT_KEY]//2
        if _DEPTH_KEY in data:
            self.depth = data[_DEPTH_KEY]
        if _FR_KEY in data:
            self.ms_fr = 1000./data[_FR_KEY]
        self.game_state = VODState.VOD_IDLE
        
    def proc_frame(self, timestamp, img, img_delta):
        pass

def detect_bot(frame):
    return 0
    
def detect_fire(frame):
    return 0
    
def seg_high(frame): #frame in bgr
    #low_seg = np.array([0, 40, 200])
    #upp_seg = np.array([255, 120, 255])
    low_seg = np.array([70, 200, 220])
    upp_seg = np.array([200, 255, 255])
    return cv2.inRange(frame, low_seg, upp_seg)
    
def seg_low(frame): #frame in hsv
    low_seg = np.array([18, 110, 150])
    upp_seg = np.array([40, 255, 255])
    return cv2.inRange(frame, low_seg, upp_seg)
    
def seg_lap(frame): #frame in bgr_16s
    low_seg = np.array([-500, -3500, -3500])
    upp_seg = np.array([1000, -200, -200])
    return cv2.inRange(frame, low_seg, upp_seg)
    

def main(args):
    frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    seg_mask1 = seg_high(frame)
    seg_mask2 = seg_low(frame_data)
    segment = cv2.bitwise_and(frame, frame, mask = seg_mask1 & seg_mask2) 
    lap = cv2.Laplacian(frame, cv2.CV_16S, ksize=5)
    lap = cv2.GaussianBlur(lap, (0,0), 1)
    seg_mask = seg_lap(lap)
    segment = cv2.bitwise_and(frame, frame, mask = seg_mask) 
    frame_data = lap


    print ("ok finished")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
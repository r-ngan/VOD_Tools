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
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        pass
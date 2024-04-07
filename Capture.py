import json
from pubsub import pub
import numpy as np

import VODEvents

CAPTURE_TOPIC = 'capture'
BOT_START = '%s.B1'%(CAPTURE_TOPIC)
BOT_END = '%s.B2'%(CAPTURE_TOPIC)
MOUSE_START = '%s.M1'%(CAPTURE_TOPIC)
MOUSE_END = '%s.M2'%(CAPTURE_TOPIC)
KEY_START = '%s.K1'%(CAPTURE_TOPIC)
KEY_END = '%s.K2'%(CAPTURE_TOPIC)

_WIDTH_KEY = 'width'
_HEIGHT_KEY= 'height'
_DEPTH_KEY = 'depth'
_FR_KEY    = 'frame_rate'

class Capture:
    
    def __init__(self, frame_rate=30.0):
        self.inited = False
        self.ms_frame_rate = 1000./frame_rate
        pub.subscribe(self.initialize, VODEvents.VOD_START)
        pub.subscribe(self.event, CAPTURE_TOPIC)
    
    def initialize(self, **data):
        if _WIDTH_KEY in data:
            self.xdim = data[_WIDTH_KEY]
        if _HEIGHT_KEY in data:
            self.ydim = data[_HEIGHT_KEY]
        if _DEPTH_KEY in data:
            self.depth = data[_DEPTH_KEY]
        if _FR_KEY in data:
            self.ms_frame_rate = 1000./data[_FR_KEY]
        self.reset()
        self.inited = True
        
    def reset(self):
        pass
        
    def is_inited(self):
        return self.inited
        
    def event(self, timestamp, topic=pub.AUTO_TOPIC, **data):
        if not self.is_inited():
            return False
        return True
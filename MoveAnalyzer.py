import math
import time
from pubsub import pub
import numpy as np
import cv2

import VideoAnalysis
import VODEvents
import VODState
from ImgProc import OpticFlow, BotPose # dependency for generation

# uses flow information to determine movement types
class MoveAnalyzer(VideoAnalysis.Analyzer):
        
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.ts_static = None # last seen timestamp
        self.ts_moving = None
        self.moving = False
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        super().proc_frame(timestamp=timestamp, img=img, aux_imgs=aux_imgs)
        bx,by = BotPose.get_bot_pose(aux_imgs)
        
        if self.ts_static is None:
            self.ts_static = timestamp
        if self.ts_moving is None:
            self.ts_moving = timestamp

        if 'moving' in aux_imgs:
            dts = timestamp - self.ts_moving
            if not self.moving: # start to move (TODO differentiate between mouse/kb)
                pub.sendMessage(VODEvents.MOUSE_MOVE_START, timestamp=timestamp,
                                x=bx,y=by,)
                self.moving = True
            self.ts_moving = timestamp
        else:
            dts = timestamp - self.ts_static
            if self.moving: # stopped move (TODO differentiate between mouse/kb)
                pub.sendMessage(VODEvents.MOUSE_MOVE_END, timestamp=timestamp,
                                x=bx,y=by,)
                self.moving = False
            self.ts_static = timestamp
        
        

_ = MoveAnalyzer()
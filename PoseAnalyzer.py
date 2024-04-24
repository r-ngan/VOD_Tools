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
#from ImgProc import OpticFlow, BotPose # dependency for generation

# use bot poses to mark trial start/end
class PoseAnalyzer(VideoAnalysis.Analyzer):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        super().proc_frame(timestamp=timestamp, img=img, aux_imgs=aux_imgs)
        poses = aux_imgs['poses']
        
        if self.game_state == VODState.VOD_BOT_ONSCREEN:
            if len(poses) < 1:
                self.game_state = VODState.VOD_IDLE
                pub.sendMessage(VODEvents.BOT_NONE, timestamp=timestamp)
            
        if self.game_state == VODState.VOD_IDLE: # full screen search
            if len(poses) > 0:
                bx,by = poses[0]
                self.game_state = VODState.VOD_BOT_ONSCREEN
                pub.sendMessage(VODEvents.BOT_APPEAR, timestamp=timestamp,
                                x=bx,y=by,)
    
_ = PoseAnalyzer()
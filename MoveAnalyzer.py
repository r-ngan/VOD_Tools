import math
import time
from pubsub import pub
import numpy as np
import cv2

import VideoAnalysis
import VODEvents
import VODState
from ImgProc import OpticFlow, BotPose # dependency for generation
import FlowAnalyzer

MOUSE_THRESHOLD = 0.05*(np.pi/180) # 0.05degrees movement to trigger
# uses flow information to determine movement types
class MoveAnalyzer(FlowAnalyzer.FlowAnalyzer):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #pub.subscribe(self.reset, VODEvents.BOT_NONE)
        pub.subscribe(self.reset, VODEvents.BOT_APPEAR)
        
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.ts_static = None # last seen timestamp
        self.ts_moving = None
        self.fsm_inmotion = False
        self.track_hist = []
        
    def reset(self, timestamp, x, y, topic=pub.AUTO_TOPIC):
        self.track_hist.clear()
        self.fsm_inmotion = False
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        super().proc_frame(timestamp=timestamp, img=img, aux_imgs=aux_imgs)
        bot_dir = BotPose.get_bot_pose(aux_imgs)
        bx,by = bot_dir
        
        if self.ts_static is None:
            self.ts_static = timestamp
        if self.ts_moving is None:
            self.ts_moving = timestamp
        
        if self.moving: # moving from super
            self.track_hist.append(-self.delta_ang)
            
        mouse_mag = np.linalg.norm(self.delta_ang, axis=-1)
        bot_mag = np.linalg.norm(bot_dir, axis=-1)
        in_dir = np.dot(self.delta_ang, bot_dir)/bot_mag
        print ('%s / %s : %s'%(self.delta_ang, bot_dir, in_dir))
        if mouse_mag > MOUSE_THRESHOLD:
            if in_dir>2e-3: # trigger only if moving towards bot
                dts = timestamp - self.ts_moving
            
                if not self.fsm_inmotion: # start to move
                    print ('start')
                    pub.sendMessage(VODEvents.MOUSE_MOVE_START, timestamp=timestamp,
                                    x=bx,y=by,)
                    self.fsm_inmotion = True
            self.ts_moving = timestamp
        else:
            dts = timestamp - self.ts_static
            dtm = timestamp - self.ts_moving
            if self.fsm_inmotion and dtm>2: # stopped move
                print ('stopped')
                pub.sendMessage(VODEvents.MOUSE_MOVE_END, timestamp=timestamp,
                                x=bx,y=by,)
                self.fsm_inmotion = False
            self.ts_static = timestamp
        
        

instance = MoveAnalyzer()
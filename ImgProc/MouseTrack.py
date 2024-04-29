import math
import time
from pubsub import pub
import numpy as np
import cv2

from ImgProc import ImgTask
import VODEvents
import VODState

MOUSE_THRESHOLD = 0.05*(np.pi/180) # 0.05degrees movement to trigger
# uses flow information to determine movement types
class MouseTrack(ImgTask.ImgTask):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #pub.subscribe(self.reset, VODEvents.BOT_NONE)
        #pub.subscribe(self.reset, VODEvents.BOT_APPEAR)
        
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.ts_static = None # last seen timestamp
        self.ts_moving = None
        self.fsm_inmotion = False
            
    def requires(self):
        return [ImgTask.VAL_FRAMENUM, 'pred_cam', 'bothead_closest']
        
    def outputs(self):
        return [VODEvents.EVENT_NODE]
        
    def proc_frame(self, timestamp, pred_cam, bothead):
        base_ang = (0, pred_cam[0])
        delta_ang = np.array([pred_cam[1], pred_cam[2]])
        bx,by = bothead
        events = []
        
        if self.ts_static is None:
            self.ts_static = timestamp
        if self.ts_moving is None:
            self.ts_moving = timestamp
        
        #if self.moving: # moving from super
        #    self.track_hist.append(-delta_ang)
        #    #self.track_hist.append(mouse_xy)
            
        mouse_mag = np.linalg.norm(delta_ang, axis=-1)
        bot_mag = np.linalg.norm(bothead, axis=-1)
        in_dir = np.dot(delta_ang, bothead)/bot_mag
        #print ('%s / %s : %s'%(delta_ang, bothead, in_dir))
        if mouse_mag > MOUSE_THRESHOLD:
            if in_dir>2e-3: # trigger only if moving towards bot
                dts = timestamp - self.ts_moving
            
                if not self.fsm_inmotion: # start to move
                    events.append({'topic':VODEvents.MOUSE_MOVE_START,
                                    'timestamp':timestamp,
                                    'x':bx,'y':by,})
                    self.fsm_inmotion = True
            self.ts_moving = timestamp
        else:
            dts = timestamp - self.ts_static
            dtm = timestamp - self.ts_moving
            if self.fsm_inmotion and dtm>2: # stopped move
                events.append({'topic':VODEvents.MOUSE_MOVE_END,
                                'timestamp':timestamp,
                                'x':bx,'y':by,})
                self.fsm_inmotion = False
            self.ts_static = timestamp
            
        return events
        

instance = MouseTrack()
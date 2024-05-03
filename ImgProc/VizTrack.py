import math
import time
import numpy as np
import cv2

import VODEvents
from ImgProc import ImgTask
from ImgProc import MotionDisc as Motion

# uses flow information to determine movement types
class VizTrack(ImgTask.ImgTask):
        
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.last_ts = 0
        self.track_hist = []
        
    def clear_track(self):
        self.track_hist.clear()
            
    def requires(self):
        return [ImgTask.VAL_FRAMENUM, ImgTask.IMG_BASE, 'pred_cam', VODEvents.EVENT_NODE]
        
    def outputs(self):
        return [ImgTask.IMG_DEBUG]
        
    def proc_frame(self, timestamp, frame, pred_cam, eventdict):
        events = self.unroll(eventdict)
        for e in events:
            if (e['topic'] == VODEvents.BOT_APPEAR):
                self.clear_track()
                break
        
        base_ang = (0, pred_cam[0])
        delta_ang = np.array([pred_cam[1], pred_cam[2]])
        
        self.track_hist.append(-delta_ang)
        
        dbg_img = np.array(frame)
        self.draw_hist(dbg_img)
            
        return dbg_img
            
    def draw_hist(self, img):
        ang = np.array([0,0])
        angscalar = np.array([self.xdim/Motion.HFOV,
                            self.ydim/Motion.VFOV])
        mid = np.array([self.midx, self.midy])

        for xy in self.track_hist[::-1]:
            mag = np.linalg.norm(xy)
            MAG_LIM = 2.*np.pi/180
            mag = min(MAG_LIM,mag) # cap to limit
            color = [0,128*mag/MAG_LIM,200*mag/MAG_LIM]
            color = tuple(int(x) for x in color)
            newang = ang+xy
            loc = (ang)*angscalar+mid
            newloc = (newang)*angscalar+mid
            x1 = int(loc[0])
            y1 = int(loc[1])
            x2 = int(newloc[0])
            y2 = int(newloc[1])
            cv2.line(img, (x1,y1), (x2,y2), color, int(1.5+2.*mag/MAG_LIM))
            ang = newang
        

instance = VizTrack()
import math
import time
import numpy as np
import cv2

from ImgProc import ImgTask
from ImgProc import BotTrack
from ImgProc.BotFind import pixel, topleft, botright, subimage

DEBUG=True # module is only used for debugging

class VizBots(ImgTask.ImgTask):
            
    def requires(self):
        return [ImgTask.IMG_BASE, 'botposes']
        
    def outputs(self):
        return [ImgTask.IMG_DEBUG]

    # proc_frame signature must match requires / outputs
    def proc_frame(self, img, botposes):
        # botposes only has heads right now, need to rely on BotTrack pose store
        dbg_img = np.array(img)
        BotTrack.instance.draw_bots(dbg_img)
        return dbg_img
        
    # unused
    def draw_bots(self, debug=None):
        COL = [(0,0,255),
                (0,255,255),
                (0,255,0),
                (255,0,255),
                (255,0,0),
                (255,255,0),
                (255,255,255),
                ]
        if debug is None:
            return
        for ix, det in enumerate(self.bots):
            headxy = det.last_head
            kp_list = det.last_pose
            color = COL[ix%len(COL)]
            cv2.rectangle(debug, topleft(det.last_bound), botright(det.last_bound), color=color, thickness=1)
            self.draw_keypoints(debug, kp_list)
            kpx, kpy, headsize = headxy
            circ = (kpx,kpy)
            cv2.circle(debug, pixel(circ), int(headsize), color=(0,0,255), thickness=1)
        
    # unused
    def draw_keypoints(self, img, kp_list):
        # 17 keypoints in COCO dataset
        # 0 = nose
        # 1 = left eye
        # 2 = right eye
        # 3 = left ear
        # 4 = right ear
        # 5 = left shoulder
        # 6 = right shoulder
        # 7 = left elbow
        # 8 = right elbow
        # 9 = left wrist
        # 10 = right wrist
        # 11 = left hip
        # 12 = right hip
        # 13 = left knee
        # 14 = right knee
        # 15 = left ankle
        # 16 = right ankle
        linepairs = [[0,5],
                    [0,6],
                    [5,6],
                    [5,11],
                    [6,12],
                    [11,12],
                    [11,13],
                    [13,15],
                    [12,14],
                    [14,16],]
        pos = kp_list[:,:-1]
        kp_conf = kp_list[:,-1]
            
        for pair in linepairs:
            conf = kp_conf[pair[0]]*kp_conf[pair[1]]
            if conf<0.5:
                continue
            p0 = pos[pair[0]]
            p1 = pos[pair[1]]
            cv2.line(img, pixel(p0), pixel(p1), color=(0,128,255), thickness=2)
        
        for ix, xy in enumerate(pos):
            kpcolor = (0,255,0)
            if ix<5: # nose leye reye lear rear
                kpcolor = (0,0,255)
            elif 5<=ix<9: # lrshoulder lrelbow
                kpcolor = (255,0,255)
            kpx, kpy = xy
            box = (kpy-1,kpy+1,kpx-1,kpx+1)
            #cv2.rectangle(img, topleft(box), botright(box), color=kpcolor, thickness=1)
    
instance = VizBots()
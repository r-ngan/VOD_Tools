import math
import sys
import traceback
import json
from pubsub import pub
import numpy as np
import cv2
from ultralytics import YOLO

import VideoAnalysis
import VODEvents
import VODState
from ImgProc import OpticFlow # dependency for generation

BOT_TIMEOUT = 4
class PoseAnalyzer(VideoAnalysis.Analyzer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.qmodel = YOLO('assets/yolov8m.pt')
        YOLO_KERNEL=32
        DOWNSCALE=3
        self.qdim = ((self.ydim//YOLO_KERNEL)//DOWNSCALE * YOLO_KERNEL,
                    (self.xdim//YOLO_KERNEL)//DOWNSCALE * YOLO_KERNEL,)
        self.finemodel = YOLO('assets/yolov8m-pose.pt')
        self.finedim = ((self.ydim//YOLO_KERNEL) * YOLO_KERNEL,
                        (self.xdim//YOLO_KERNEL) * YOLO_KERNEL,)
        self.filtcls = []
        for ix, cls in self.qmodel.names.items():
            if cls=='person':
                self.filtcls.append(ix)
        self.last_bound = (0,0,0,0)
        self.last_head = (0,0)
        self.bot_track = False
        self.last_bot_ts = 0
        
    def update_track(self, motion):
        y,y2,x,x2 = self.last_bound
        dx = motion[0]
        dy = motion[1]
        self.last_bound = [y+dy,y2+dy,x+dx,x2+dx]
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        super().proc_frame(timestamp=timestamp, img=img, aux_imgs=aux_imgs)
        
        if self.bot_track: # light update based on flow, pose check
            flimg = subimage(aux_imgs['flow'], self.last_bound)
            motion = OpticFlow.get_avg_flow(flimg)
            self.update_track(motion)
            self.last_head += motion
            
            newsubimg = subimage(img, self.last_bound) # bound after update
            kp_list = self.pose_findbot(newsubimg)
            if kp_list is not None: # bot still present
                self.last_bot_ts = timestamp
            else:
                if timestamp - self.last_bot_ts > BOT_TIMEOUT: # bot is probably gone
                    self.bot_track = False
                    self.game_state = VODState.VOD_IDLE
                    pub.sendMessage(VODEvents.BOT_NONE, timestamp=timestamp,
                                    x=self.last_head[0],y=self.last_head[1],)
            
            debug = aux_imgs['debug']
            if debug is not None:
                cv2.rectangle(debug, topleft(self.last_bound), botright(self.last_bound), color=(0,255,0), thickness=1)
                kpx, kpy = self.last_head
                box = (kpy-2,kpy+2,kpx-2,kpx+2)
                cv2.rectangle(debug, topleft(box), botright(box), color=(0,255,0), thickness=1)
        
        if self.game_state == VODState.VOD_IDLE: # full screen search
            bots = self.findbot(img, debug=aux_imgs['debug'])
            if len(bots)>0:
                if not self.bot_track:
                    self.last_bot_ts = timestamp
                    self.last_head = bots[0]['head']
                    self.last_bound = bots[0]['bound']
                self.bot_track = True
                self.game_state = VODState.VOD_BOT_ONSCREEN
                pub.sendMessage(VODEvents.BOT_APPEAR, timestamp=timestamp,
                                x=self.last_head[0],y=self.last_head[1],)
        
    # Based on keypoints, check how confident we got a full body
    def get_body_conf(self, kp_list):
        # items are [x,y,confidence]
        IX_X = 0
        IX_Y = 1
        IX_CONF = 2
        if len(kp_list)<1: # no key points
            return 0
        return sum([it[IX_CONF] for it in kp_list]) / len(kp_list)
        
    def get_weighted_avg(self, kp_list):
        # items are [x,y,confidence]
        IX_X = 0
        IX_Y = 1
        IX_CONF = 2
        if len(kp_list)<1: # no key points
            return None
        weight_func = lambda x: x**2
        weight_sum = sum([weight_func(it[IX_CONF]) for it in kp_list])
        return (sum([it[IX_X]*weight_func(it[IX_CONF]) for it in kp_list])/weight_sum,
                sum([it[IX_Y]*weight_func(it[IX_CONF]) for it in kp_list])/weight_sum)
        
    def findbot(self, img, debug=None):
        results = []
        for b, subimg in self.coarse_findbot(img):
            if debug is not None:
                cv2.rectangle(debug, topleft(b), botright(b), color=(0,0,255), thickness=1)
            kp_list = self.pose_findbot(subimg)
            if kp_list is None:
                continue
                
            # don't rely on face kps, use shoulders and extrapolate a triangle is more stable
            base_kps = kp_list[5:7]
            base = self.get_weighted_avg(base_kps)
            base_dist = math.dist(base_kps[0][:-1], base_kps[1][:-1])
            kpx = b[2]+base[0]
            kpy = b[0]+base[1]-base_dist*0.6
            if debug is not None:
                box = (kpy-2,kpy+2,kpx-2,kpx+2)
                cv2.rectangle(debug, topleft(box), botright(box), color=(255,0,0), thickness=1)
                
            results.append({
                    'head':[kpx,kpy],
                    'bound':b, })
        return results
        
    def pose_findbot(self, img):
        imgsz = np.array(img.shape[:2])
        YOLO_STRIDE = 32
        imgsz = ((imgsz+YOLO_STRIDE-1)//YOLO_STRIDE)*YOLO_STRIDE # YOLO needs multiple of 32
        pose = self.finemodel.predict(img,
                        conf=0.01,
                        imgsz=tuple(imgsz), 
                        classes=self.filtcls,
                        stream=False, verbose=False)[0]
        if pose.keypoints is None:
            return None
        kp_list = pose.keypoints.data[0]
        if self.get_body_conf(kp_list)<0.5: # weed out non-bots
            return None
        return kp_list
        
    def coarse_findbot(self, img):
        MAX_X = self.xdim//5
        MAX_Y = self.ydim//3
        res = self.qmodel.predict(img,
                        conf=0.1, iou=0.4,
                        imgsz=self.qdim, 
                        classes=self.filtcls,
                        stream=False, verbose=False)[0]
                        
        boxes = res.boxes.xywh
        for box in boxes:
            x, y, w, h = box # detect box is centered x,y
            if w>MAX_X or h>MAX_Y : # don't process boxes too big
             continue
            #print('%s %s %s %s'%(x,y,w,h))
            PXE = 10 # border expansion to capture more context
            x1 = max(0,x-w/2-PXE)
            y1 = max(0,y-h/2-PXE)
            x2 = min(self.xdim,x+w/2+PXE)
            y2 = min(self.ydim,y+h/2+PXE)
            bound = [y1,y2,x1,x2]
            yield bound, subimage(img,bound)
        
    def draw_keypoints(self, img, b, kp_list):
        for ix,kp in enumerate(kp_list):
            kpx = b[2]+int(round(kp[0]))
            kpy = b[0]+int(round(kp[1]))
            conf = kp[2]
            kpcolor = (0,255,0)
            if ix<5: # nose leye reye lear rear
                kpcolor = (0,0,255)
            elif 5<=ix<9: # lrshoulder lrelbow
                kpcolor = (255,0,255)
            
            cv2.rectangle(img, (kpx-1,kpy-1), (kpx+1,kpy+1), color=kpcolor, thickness=1)


    def threshold_edge(self, img):
        # yellow outline
        lowseg = (20,90,100) # hsv 180,255,255
        uppseg = (45,256,256)
        mask1 = cv2.inRange(img, lowseg, uppseg)
        # super bright
        lowseg = (10,0,200) # hsv 180,255,255
        uppseg = (90,70,256)
        mask2 = cv2.inRange(img, lowseg, uppseg)
        return (mask1|mask2)#.astype(np.float32)/254. # 0 or 1 only
        
    def threshold_legs(self, img):
        lowseg = (0,0,0) # hsv 180,255,255
        uppseg = (180,60,130)
        mask3 = cv2.inRange(img, lowseg, uppseg)
        return (mask3)#.astype(np.float32)/254. # 0 or 1 only
    
def topleft(bound):
    y,y2,x,x2 = bound
    return (int(x),int(y))

def botright(bound):
    y,y2,x,x2 = bound
    return (int(x2),int(y2))
    
def subimage(img, bound):
    y,y2,x,x2 = bound
    return img[int(y):int(y2),int(x):int(x2)]
    
_ = PoseAnalyzer()
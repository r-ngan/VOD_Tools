import math
import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from ImgProc import ImgTask

POSE_MODEL_PATH='assets/valbots640-pose.pt'
POSE_MODEL_IMGSZ=(640)

# find location of bots and presents them for analyzers to use
# Theory of operation:
# With custom trained model, pure pose estimator time is comparable to doing a two-pass search
#
# After first pass and hits have been identified, use optical flow to track updates
# lighter lift by only pose estimating on the estimated target location rather than full screen
#
# Sometimes pose fitting will fail. Allow dead reckoning for a while until get a good lock
# Each time pose fit succeeds, use head position to improve tracking
class BotFind(ImgTask.ImgTask):
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.finemodel = YOLO(POSE_MODEL_PATH)
        self.finedim = POSE_MODEL_IMGSZ
        self.filtcls = []
        for ix, cls in self.finemodel.names.items():
            if cls in ['person']:
                self.filtcls.append(ix)
            
    def requires(self):
        return [ImgTask.IMG_BASE]
        
    def outputs(self):
        return ['botlist']

    # proc_frame signature must match requires / outputs
    def proc_frame(self, img):
        bots = self.findbot(img)
        return bots
        
    def findbot(self, img):
        results = []
        tst = time.time_ns()
        res = self.finemodel.predict(img,
                        conf=0.5,
                        imgsz=self.finedim, 
                        classes=self.filtcls,
                        stream=False, verbose=False)[0]
                        
        ten = time.time_ns()
        #print ('findbot= %3.3fms'%((ten-tst)/1e6))
        
        #print (res.boxes.conf)
        for box, kp_list in zip(res.boxes.data, res.keypoints.data):
            det = Bot(self.xdim, self.ydim)
            det.last_bound = box.cpu()
            det.last_pose = kp_list.cpu()
            det.last_head = det.pin_head()
            results.append(det)

        return results

# helper class to package bot into a data structure
class Bot:
    def __init__(self, xdim, ydim, ts=0, **kwargs):
        self.last_bound = torch.tensor([0,0,0,0])
        self.last_head = torch.tensor([0,0,0])
        self.last_pose = None
        self.last_ts = ts
        self.xdim = xdim
        self.ydim = ydim
        
    def __repr__(self):
        return str(vars(self))
        
    def sim_score(self, bot2): # get similarity between two detections
        a = self.last_head
        b = bot2.last_head
        val = (a-b)[:2]
        dist = torch.norm(val)
        return 20./(20.+dist)
        
    def get_bound_xywh(self):
        x1,y1,x2,y2 = self.last_bound.cpu().numpy()[:4]
        w = (x2-x1)/self.xdim
        h = (y2-y1)/self.ydim
        x = (x2+x1)/2/self.xdim
        y = (y2+y1)/2/self.ydim
        return x,y,w,h
        
    def get_head_xywh(self):
        x1,y1,size = self.last_head
        w = (2*size)/self.xdim
        h = (2*size)/self.ydim
        x = x1/self.xdim
        y = y1/self.ydim
        return x,y,w,h
        
    def get_pose_data(self):
        kp_list = self.last_pose.cpu().numpy()
        res = []
        for x1,y1,conf in kp_list:
            x = x1/self.xdim
            y = y1/self.ydim
            conf = 1. if conf>0.8 else 0. # threshold for ground truth
            res.extend([x,y,conf])
        return res
        
    def update_track(self, motion):
        dx = motion[0]
        dy = motion[1]
        delta = torch.tensor([dx,dy,0], device=self.last_pose.device)
        self.last_bound[0::2] += dx
        self.last_bound[1::2] += dy
        self.last_head = self.last_head + delta
        self.last_pose = self.last_pose + delta
        
    def pin_head(self, pose=None): # pure pose estimator
        if pose is None:
            pose = self.last_pose
        # use shoulders for head size
        base_vector = pose[5]-pose[6] # shoulder keypoints
        headsize = torch.norm(base_vector[:-1]) *0.45 # head is proportion of shoulder
        
        headxy = pose[0].clone().detach()
        headxy[-1] = headsize
        return headxy
                
    def refine(self, bot):
        new_pose = bot.last_pose
        new_bound = bot.last_bound
        new_headxy = bot.last_head
        
        ALPHA = 0.15
        cconf = self.get_chest_conf(new_pose)
        robust = cconf>0.9
        if robust:
            self.last_bound = self.last_bound*(1-ALPHA) + new_bound*(ALPHA)
            self.last_head = self.last_head*(1-ALPHA) + new_headxy*(ALPHA)
            last_c = (self.last_pose[:,-1:]**6)*(1-ALPHA)
            new_c = (new_pose[:,-1:]**6)*(ALPHA)
            sum_c = last_c+new_c
            
            self.last_pose = self.last_pose*last_c/sum_c + \
                            new_pose*new_c/sum_c
            
    # use a small region around the head for flow tracking
    def get_headbox(self):
        PXE = self.last_head[-1] # radius of head
        PXE_DOWN = PXE*2
        x = self.last_head[0]
        y = self.last_head[1]
        res = torch.tensor([x-PXE,y-PXE,x+PXE,y+PXE_DOWN], device=self.last_head.device)
        res[0::2] = torch.clip(res[0::2], min=0, max=self.xdim)
        res[1::2] = torch.clip(res[1::2], min=0, max=self.ydim)
        return res
        
    # Based on keypoints, check how confident a full body is present
    def get_body_conf(self, kp_list):
        # items are [x,y,confidence]
        sub_ix = [0,5,6,11,12]
        sub_list = [x for ix, x in enumerate(kp_list) if ix in sub_ix]
        IX_X = 0
        IX_Y = 1
        IX_CONF = 2
        if len(kp_list)<1: # no key points
            return 0
        return sum([it[IX_CONF] for it in sub_list]) / len(sub_list)
        
    # Based on keypoints, check how confident face+chest is good
    def get_chest_conf(self, kp_list):
        chest_ix = [0,5,6,11,12]
        chest = kp_list[chest_ix]
        # pretend chest is axis-aligned bounding box (need to account rotate/warp later)
        chest_min = chest.min(axis=0).values
        chest_max = chest.max(axis=0).values
        
        return chest_min[-1]
    
def pixel(xy):
    x,y = xy
    return (int(x),int(y))
    
def topleft(bound):
    x1,y1,x2,y2 = bound.cpu().numpy()[:4]
    return (int(x1),int(y1))

def botright(bound):
    x1,y1,x2,y2 = bound.cpu().numpy()[:4]
    return (int(x2),int(y2))
    
def subimage(img, bound):
    x1,y1,x2,y2 = bound.cpu().numpy()[:4]
    return img[int(y1):int(y2),int(x1):int(x2)]
    
instance = BotFind()
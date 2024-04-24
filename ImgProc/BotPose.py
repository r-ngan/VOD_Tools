import math
import time
from pubsub import pub
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from ImgProc import ImgEvents, Preprocessor
from ImgProc import Delta, OpticFlow # dependency for generation

DEBUG=True
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
BOT_TIMEOUT = 10
class BotPose(Preprocessor.Preprocessor):
    
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
        self.bots = []

    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['flow', 'abs_delta']):
            return False
        delta = aux_imgs['abs_delta']
        
        poses = []
        dbg_img = np.array(img) if DEBUG else None
        bots = self.findbot(img, debug=dbg_img)
        self.map_new_bots(bots, timestamp, aux_imgs['flow'])
        self.draw_bots(dbg_img)
        
        if len(self.bots)>0:
            mid = np.array([self.midx, self.midy])
            poses.extend([(x.last_head[:2]-mid).cpu().numpy() for x in self.bots])
            
        pub.sendMessage(ImgEvents.APPEND, key='poses', imgdata=poses)
        if DEBUG:
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=dbg_img)
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=aux_imgs['last'])
        return True
        
    def findbot(self, img, debug=None):
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
        
    # compare the new hits against existing bots. use precision update to reduce noise on tracking bots
    def map_new_bots(self, new_bots, timestamp, flow):
        SIM_THRES = 0.1
        M = len(self.bots)
        N = len(new_bots)
        sim_map = np.zeros([M,N])
        itaken = []
        jtaken = []
        for ix, bot in enumerate(self.bots):
            head_bound = bot.get_headbox()
            flimg = subimage(flow, head_bound)
            motion, mvar = OpticFlow.get_avg_flow(flimg)
            # flow centroid variance threshold (too high will be inaccurate)
            motion_reliable = mvar<0.01
            if motion_reliable: # only track if the motion is stable
                bot.update_track(motion)
                
            for jx, x in enumerate(new_bots):
                similarity = bot.sim_score(x)
                sim_map[ix,jx] = similarity
        
        flatord = np.argsort(sim_map, axis=None)[::-1] # sort best matches
        x,y = np.unravel_index(flatord, sim_map.shape)
        for ix,jx in zip(x,y):
            if (ix in itaken) or (jx in jtaken) or sim_map[ix,jx]<SIM_THRES:
                continue
            #print ('map %s -> %s'%(jx, ix))
            self.bots[ix].refine(new_bots[jx])
            self.bots[ix].last_ts = timestamp
            itaken.append(ix)
            jtaken.append(jx)
        self.bots[:] = [x for x in self.bots if timestamp - x.last_ts <= BOT_TIMEOUT] # remove stale bots
        for jx in range(N):
            if not jx in jtaken:
                new_bots[jx].last_ts = timestamp
                self.bots.append(new_bots[jx])
                #print ('new %s'%(jx))
        
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
        

   
def get_bot_pose(aux_imgs={}):
    res = [float('nan'),float('nan')]
    if 'poses' in aux_imgs:
        poses = aux_imgs['poses']
        if len(poses)>0:
            res = poses[0]
    return res
    
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
    
instance = BotPose()
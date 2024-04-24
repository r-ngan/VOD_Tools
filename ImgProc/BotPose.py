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
POSE_MODEL_PATH='assets/valbotm320-pose.pt'
POSE_MODEL_IMGSZ=(320)

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
            poses.extend([x.last_head[:2]-mid for x in self.bots])
            
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
        
        for b, kp_list in zip(res.boxes.data, res.keypoints.data):
            box = b.cpu().numpy()
            det = Bot(self.xdim, self.ydim)
            det.last_bound = np.array([box[1],box[3],box[0],box[2]])
            det.last_pose = kp_list
            headxy = det.pin_head()
            det.last_head = headxy
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
            circ = (kpy,kpy,kpx,kpx)
            cv2.circle(debug, topleft(circ), int(headsize), color=(0,0,255), thickness=1)
        
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
        self.bots[:] = [x for x in self.bots if timestamp - x.last_ts < BOT_TIMEOUT] # remove stale bots
        for jx in range(N):
            if not jx in jtaken:
                new_bots[jx].last_ts = timestamp
                self.bots.append(new_bots[jx])
                #print ('new %s'%(jx))
            
        #print (sim_map)
        #print (self.bots)
        
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
        val = np.array(a-b)[:2]
        dist = np.linalg.norm(val, axis=-1)
        return 20./(20.+dist)
        
        
    def get_bound_xywh(self):
        y1,y2,x1,x2 = self.last_bound
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
        self.last_bound[2:] += dx
        self.last_bound[:2] += dy
        self.last_head[0] += dx
        self.last_head[1] += dy
        pose_delta = torch.tensor([dx,dy,0], device=self.last_pose.device)
        self.last_pose = self.last_pose + pose_delta
        
    def pin_head(self, pose=None): # pure pose estimator
        if pose is None:
            pose = self.last_pose
        # use shoulders for head size
        shoulders = pose[5:7].cpu()
        base = self.get_weighted_avg(shoulders)
        base_dist = math.dist(shoulders[0][:-1], shoulders[1][:-1])
        headsize = base_dist*0.4 # head is proportion of shoulder
        
        head = pose[0].cpu()
        headxy = np.array([head[0], head[1], headsize])
        return headxy
                
    def refine(self, bot):
        new_pose = bot.last_pose
        new_bound = bot.last_bound
        new_headxy = bot.last_head
        
        ALPHA = 0.2
        cconf, cbox = self.get_chest_conf(new_pose)
        robust = cconf>0.9
        if robust:
            self.last_bound = self.last_bound*(1-ALPHA) + new_bound*(ALPHA)
            self.last_head = self.last_head*(1-ALPHA) + new_headxy*(ALPHA)
            self.last_pose = self.last_pose*(1-ALPHA) + new_pose*(ALPHA)
            
    # use a small region around the head for flow tracking
    def get_headbox(self):
        PXE = 15
        PXE_DOWN = 30
        x1 = max(0,self.last_head[0]-PXE)
        y1 = max(0,self.last_head[1]-PXE)
        x2 = min(self.xdim,self.last_head[0]+PXE)
        y2 = min(self.ydim,self.last_head[1]+PXE_DOWN)
        return [y1,y2,x1,x2]        
        
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
        
    # Based on keypoints, check how confident chest is good
    def get_chest_conf(self, kp_list):
        chest_ix = [5,6,11,12]
        chest = kp_list[chest_ix].cpu()
        # pretend chest is axis-aligned bounding box (need to account rotate/warp later)
        chest_min = chest.min(axis=0).values
        chest_max = chest.max(axis=0).values
        y1 = chest_min[1]
        y2 = chest_max[1]
        x1 = chest_min[0]
        x2 = chest_max[0]
        bound = np.array([y1,y2,x1,x2])
        
        chest_conf = chest_min[-1]
        return chest_conf, bound
        
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
    y,y2,x,x2 = bound
    return (int(x),int(y))

def botright(bound):
    y,y2,x,x2 = bound
    return (int(x2),int(y2))
    
def subimage(img, bound):
    y,y2,x,x2 = bound
    return img[int(y):int(y2),int(x):int(x2)]
    
instance = BotPose()
import math
import time
from pubsub import pub
import numpy as np
import cv2
from ultralytics import YOLO

from ImgProc import ImgEvents, Preprocessor
from ImgProc import Delta, OpticFlow # dependency for generation

DEBUG=True

# find location of bots and presents them for analyzers to use
# Theory of operation:
# Use low res YOLO classifier to identify people onscreen (quick)
# -> however there are a lot of false negatives from hands, portraits, etc
# Confirm a hit by using a higher detailed YOLO body pose estimator
# We should be able to see head and body at least
# Use pose data to identify true head position
#
# After first pass and hits have been identified, use optical flow to track updates
# lighter lift by only pose estimating on the estimated target location rather than full screen
# -> far away targets are hard to correctly id. Scaling up image shows some improvement
#
# Sometimes pose fitting will fail. Allow dead reckoning for a while until get a good lock
# Each time pose fit succeeds, use head position to improve tracking
BOT_TIMEOUT = 20
class BotPose(Preprocessor.Preprocessor):
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.qmodel = YOLO('assets/yolov8m.pt')
        YOLO_KERNEL=32
        DOWNSCALE=2.
        self.qdim = (((self.ydim/DOWNSCALE)//YOLO_KERNEL) * YOLO_KERNEL,
                    ((self.xdim/DOWNSCALE)//YOLO_KERNEL) * YOLO_KERNEL,)
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
        self.last_bound[:2] += dy
        self.last_bound[2:] += dx
        
    def pin_head(self, pose):
        # don't rely on face kps, use shoulders and extrapolate a triangle is more stable
        shoulders = pose[5:7]
        base = self.get_weighted_avg(shoulders)
        base_dist = math.dist(shoulders[0][:-1], shoulders[1][:-1])
        headxy = np.array([base[0], base[1]-base_dist*0.6])
        return headxy
        
    def refine_pose(self, pose):
        ALPHA = 0.1
        cconf, cbox = self.get_chest_conf(pose)
        robust = cconf>0.9
        if robust:
            h = cbox[1]-cbox[0]
            w = cbox[3]-cbox[2]
            #print ('%s : %s'%(w, h))
            cbox[0] = max(0,cbox[0]-1.2*h)
            cbox[1] = min(self.ydim,cbox[1]+2.7*h)
            cbox[2] = max(0,cbox[2]-1.5*w)
            cbox[3] = min(self.xdim,cbox[3]+1.5*w)
            self.last_bound = self.last_bound*(1-ALPHA) + cbox*(ALPHA)
        headxy = self.pin_head(pose)
        self.last_head = self.last_head*(1-ALPHA) + headxy*(ALPHA)
        
    # use a small region around the head for flow tracking
    def get_headbox(self):
        PXE = 15
        PXE_DOWN = 30
        x1 = max(0,self.last_head[0]-PXE)
        y1 = max(0,self.last_head[1]-PXE)
        x2 = min(self.xdim,self.last_head[0]+PXE)
        y2 = min(self.ydim,self.last_head[1]+PXE_DOWN)
        return [y1,y2,x1,x2]

    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['base', 'last', 'flow']):
            return False
            
        poses = []
        add_pose = False
        dbg_img = aux_imgs['debug'] if DEBUG else None
        if self.bot_track: # light update based on flow, pose check
            head_bound = self.get_headbox()
            flimg = subimage(aux_imgs['flow'], head_bound)
            motion, mvar = OpticFlow.get_avg_flow(flimg)
            # flow centroid variance threshold (too high will be inaccurate)
            motion_reliable = mvar<0.01
            #if not motion_reliable:
            #    print ('%s:%s'%(timestamp,mvar))
            
            add_pose = True
            if timestamp - self.last_bot_ts > BOT_TIMEOUT: # bot is probably gone
                self.bot_track = False
                add_pose = False
            elif motion_reliable: # only track if the motion is stable
                self.update_track(motion)
                self.last_head += motion
                
                newsubimg = subimage(img, self.last_bound) # bound after update
                kp_list, kp_conf = self.pose_findbot(newsubimg, self.last_bound)
                #print ('conf=%s'%(kp_conf))
                if kp_list is not None: # bot still present
                    self.refine_pose(kp_list)
                    self.last_bot_ts = timestamp
            
            if dbg_img is not None:
                cv2.rectangle(dbg_img, topleft(self.last_bound), botright(self.last_bound), color=(0,255,0), thickness=1)
                kpx, kpy = self.last_head
                box = (kpy-2,kpy+2,kpx-2,kpx+2)
                cv2.rectangle(dbg_img, topleft(box), botright(box), color=(0,255,0), thickness=1)
        
        else: # full screen search
            bots = self.findbot(img, debug=dbg_img)
            if len(bots)>0:
                self.last_bot_ts = timestamp
                self.last_head = bots[0]['head']
                self.last_bound = bots[0]['bound']
                self.bot_track = True
                add_pose = True
        if add_pose:
            mid = np.array([self.midx, self.midy])
            poses.append(self.last_head-mid)
        
        pub.sendMessage(ImgEvents.APPEND, key='poses', imgdata=poses)
        return True
        
    # Based on keypoints, check how confident a full body is present
    def get_body_conf(self, kp_list):
        # items are [x,y,confidence]
        IX_X = 0
        IX_Y = 1
        IX_CONF = 2
        if len(kp_list)<1: # no key points
            return 0
        return sum([it[IX_CONF] for it in kp_list]) / len(kp_list)
        
    # Based on keypoints, check how confident chest is good
    def get_chest_conf(self, kp_list):
        chest_ix = [5,6,11,12]
        chest = kp_list[chest_ix]
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
        
    # full search procedure. Start with coarse classifier and confirm hits with pose estimator
    def findbot(self, img, debug=None):
        results = []
        for b, subimg in self.coarse_findbot(img):
            if debug is not None:
                cv2.rectangle(debug, topleft(b), botright(b), color=(0,0,255), thickness=1)
            kp_list, kp_conf = self.pose_findbot(subimg, b)
            if kp_list is None:
                continue
                
            headxy = self.pin_head(kp_list)
            kpx, kpy = headxy
            if debug is not None:
                self.draw_keypoints(debug, kp_list)
                box = (kpy-2,kpy+2,kpx-2,kpx+2)
                cv2.rectangle(debug, topleft(box), botright(box), color=(255,0,0), thickness=1)
                
            results.append({
                    'head' :headxy,
                    'bound':b, })
        return results
        
    # function with (x=50 y=2.5), (x=80 y=2), (x=140 y=1.5)
    def pose_scale(self, img):
        h = (img.shape[0]-100)*0.03
        scale = 1.1+1.4/(1.+math.exp(h))
        scale = round(scale*2)/2. # allow half steps only
        return scale
        
    def pose_findbot(self, img, img_offset=[0,0,0,0]):
        SCALE = self.pose_scale(img) # zoom in if far away
        #print (img.shape)
        #print (SCALE)
        enlarge = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
        imgsz = np.array(enlarge.shape[:2])
        YOLO_STRIDE = 32
        imgsz = ((imgsz+YOLO_STRIDE-1)//YOLO_STRIDE)*YOLO_STRIDE # YOLO needs multiple of 32
        pose = self.finemodel.predict(enlarge,
                        conf=0.22,
                        imgsz=tuple(imgsz), 
                        classes=self.filtcls,
                        stream=False, verbose=False)[0]
        if pose.keypoints is None:
            return None, None
        kp_list = pose.keypoints.data[0]/SCALE # beware that confidence is SCALED also
        kp_list[:, 0] += img_offset[2]
        kp_list[:, 1] += img_offset[0]
        kp_list[:,-1] *= SCALE
        if self.get_body_conf(kp_list)<0.4: # weed out non-bots
            return None, None
        return kp_list, pose.boxes.conf
        
    def coarse_findbot(self, img):
        SCALE = 1
        MAX_X = self.xdim/5
        MAX_Y = self.ydim/3
        MIN_ASPECT = 1.2 # H should be larger than W
        TRIM_Y1 = 90
        TRIM_Y2 = 90
        enlarge = cv2.resize(img[TRIM_Y1:-TRIM_Y2,...], (0, 0), fx=SCALE, fy=SCALE)
        res = self.qmodel.predict(enlarge,
                        conf=0.02, iou=0.3,
                        imgsz=self.qdim, 
                        classes=self.filtcls,
                        stream=False, verbose=False)[0]
                        
        boxes = res.boxes.xywh/SCALE
        for box in boxes:
            x, y, w, h = box # detect box is centered x,y
            if w>MAX_X or h>MAX_Y or h/w<MIN_ASPECT: # don't process boxes too big
             continue
            PXE = 10 # border expansion to capture more context
            x1 = max(0,x-w/2-PXE)
            y1 = max(0,y-h/2-PXE+TRIM_Y1)
            x2 = min(self.xdim,x+w/2+PXE)
            y2 = min(self.ydim,y+h/2+PXE+TRIM_Y1)
            bound = np.array([y1,y2,x1,x2])
            yield bound, subimage(img,bound)
        
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
    
def get_bot_pose(aux_imgs={}):
    res = (float('nan'),float('nan'))
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
    
_ = BotPose()
import math
import time
import numpy as np
import cv2

from ImgProc import ImgTask
from ImgProc import OpticFlow # dependency for generation
from ImgProc.BotFind import pixel, topleft, botright, subimage

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
class BotTrack(ImgTask.ImgTask):
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.bots = []
            
    def requires(self):
        return [ImgTask.VAL_FRAMENUM, ImgTask.IMG_FLOW, 'botlist']
        
    def outputs(self):
        return ['botposes']

    # proc_frame signature must match requires / outputs
    def proc_frame(self, frame_num, flow, botlist):
        poses = []
        tst = time.time_ns()
        self.map_new_bots(botlist, frame_num, flow)
        ten = time.time_ns()
        #print ('map= %3.3fms'%((ten-tst)/1e6))
        
        if len(self.bots)>0:
            poses.extend(self.center_head(self.bots))

        return poses
        
    def center_head(self, heads):
        mid = np.array([self.midx, self.midy, 0.])
        return [(x.last_head.cpu().numpy()-mid) for x in self.bots]
        
        
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
            motion, mvar = get_avg_flow(flimg)
            #motion, mvar = OpticFlow.get_avg_flow(flimg) # clustering flow is 10x more accurate, but slower
            # flow centroid variance threshold (too high will be inaccurate)
            motion_reliable = mvar<0.02
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

def get_avg_flow(flow):
    flist = flow.reshape(-1,2)
    cent = np.mean(flist,0)
    mvar = np.var(flist-cent)
    return cent, mvar
    
instance = BotTrack()
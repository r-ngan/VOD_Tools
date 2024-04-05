import math
import time
from pubsub import pub
import numpy as np
import cv2
from sklearn.cluster import BisectingKMeans

from ImgProc import ImgEvents, Preprocessor
from ImgProc import Delta # dependency for generation

DEBUG=False

VIS_SIZE = 400
VIS_SCALE = VIS_SIZE/ 80. # 80 px movement is full span
KERN_SIZE = 15
CLUSTERS = 6
DOWNSCALE = 2
class OpticFlow(Preprocessor.Preprocessor):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.STRENGTH = 1e6/(self.xdim*self.ydim*self.depth)
        
        XSTEP = 20
        YSTEP = 20
        self.y,self.x = np.mgrid[YSTEP/2:self.ydim:YSTEP, XSTEP/2:self.xdim:XSTEP]. \
                        reshape(2,-1).astype(int)
        
        fu = self.x.astype(np.float32) - self.midx
        fv = self.y.astype(np.float32) - self.midy
        mag, ang = cv2.cartToPolar(fu, fv)
        hsv = np.ones([self.ydim, self.xdim, 3], dtype=np.uint8)*255
        hsv[self.y, self.x, 0] = ang[...,0]*180/np.pi/2
        self.hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)/255.
        
        #self.regress = AgglomerativeClustering(n_clusters=None, distance_threshold=90, linkage='ward')
        self.regress = BisectingKMeans(n_clusters=CLUSTERS, bisecting_strategy='biggest_inertia')
        
    def flowviz(self, flow):
        
        vis = np.zeros([VIS_SIZE,VIS_SIZE,3], dtype=np.float32)
        flow_mag = np.linalg.norm(flow[self.y,self.x], axis=-1)
        moving = flow_mag>0.5 # filter down to reduce work on non-movement
        fy,fx = self.y[moving], self.x[moving]
        uvs = flow[fy,fx].T *VIS_SCALE
        locs = (np.clip(uvs, -VIS_SIZE/2, VIS_SIZE/2-1)+VIS_SIZE/2).astype(int)
        px = (locs[1],locs[0])
        
        # sum up flow heatmap to xy grid
        np.add.at(vis, px, self.STRENGTH*self.hue[fy,fx]) # at allows repeated indices
        vis = cv2.GaussianBlur(vis, (KERN_SIZE, KERN_SIZE), 1.) # smooth the heatmap
        
        # cluster flow to find representative motion
        if moving.sum()>100:
            flist = flow[fy,fx].reshape(-1,2)
            cent = get_avg_flow(flow[fy,fx])
            pos = (VIS_SIZE/2+cent*VIS_SCALE).astype(int)
            cv2.circle(vis, pos, 5, (0,0,255),1)
            
            km = self.regress.fit(flist)
            #KMN = km.n_clusters_
            KMN = CLUSTERS
            count = [(km.labels_==n).sum() for n in range(KMN)]
            maxcount = max(count)
            modes = [i for i in range(KMN) if count[i]==maxcount]
            
            for ix in modes:
                cent = np.mean(flist[km.labels_==ix],0)
                pos = (VIS_SIZE/2+cent*VIS_SCALE).astype(int)
                kmcolor = (0,0,255)
                if maxcount == count[ix]:
                    kmcolor = (0,255,0)
                cv2.circle(vis, pos, 5, kmcolor,1)
        return (np.clip(vis,0,1.)*255).astype(np.uint8)
    
    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['base', 'last', 'abs_delta']):
            return
        lframe = aux_imgs['last']
        frame = aux_imgs['base']
        abs_delta = aux_imgs['abs_delta'] # abs_delta to check moving
        
        # movement threshold
        moving = False
        VAL_THRES = 40
        COUNT_THRES = int(0.003*self.xdim*self.ydim*self.depth)
        if (abs_delta > VAL_THRES).sum() > COUNT_THRES:
            moving = True
            pub.sendMessage(ImgEvents.APPEND, key='moving', imgdata=[True])
        
        if moving: # only calculate optic flow if sufficient movement
            fprev = cv2.resize(cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY),
                                (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE).astype(np.int16)
            fnext = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE).astype(np.int16)
            CLIP_E = 50 # remove border pixels
            flow = cv2.calcOpticalFlowFarneback(fprev, fnext, None, 0.5, 6, 35, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = cv2.resize(flow*DOWNSCALE, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # hsv = np.zeros_like(frame)
            # hsv[..., 0] = ang*180/np.pi/2
            # hsv[..., 1] = 255
            # hsv[..., 2] = np.clip(np.sqrt(mag)*20,0,255)
            #flowviz = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        else:
            flow = np.zeros_like(frame)[...,:2]
    
        if DEBUG:
            heatmap = self.flowviz(flow)
            flowviz = np.array(frame)
            flowviz[0:400,0:400,:] = heatmap
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=flowviz)
        
        pub.sendMessage(ImgEvents.APPEND, key='flow', imgdata=flow)

def get_avg_flow(flow):
    flist = flow.reshape(-1,2)
    cent = np.mean(flist,0)
    
    regress = BisectingKMeans(n_clusters=CLUSTERS, bisecting_strategy='biggest_inertia')
    km = regress.fit(flist)
    #KMN = km.n_clusters_
    KMN = CLUSTERS
    count = [(km.labels_==n).sum() for n in range(KMN)]
    maxcount = max(count)
    modes = [i for i in range(KMN) if count[i]==maxcount]
    
    for ix in modes:
        cent = np.mean(flist[km.labels_==ix],0)
        return cent

_ = OpticFlow()
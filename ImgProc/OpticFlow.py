import math
import time
from pubsub import pub
import numpy as np
import cv2
from sklearn.cluster import BisectingKMeans

from ImgProc import ImgEvents, Preprocessor
from ImgProc import Delta # dependency for generation

VIS_SIZE = 400
VIS_SCALE = VIS_SIZE/ 80. # 80 px movement is full span
KERN_SIZE = 15
CLUSTERS = 5
DOWNSCALE = 2
class OpticFlow(Preprocessor.Preprocessor):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.last_fprev = np.array([])
        self.last_fprev_hash = 0
        
        self.STRENGTH = 1e6/(self.xdim*self.ydim*self.depth)
                
        XSTEP = 20
        YSTEP = 10
        self.y,self.x = np.mgrid[YSTEP/2:self.ydim:YSTEP, XSTEP/2:self.xdim:XSTEP]. \
                        reshape(2,-1).astype(int)
        fu = self.x.astype(np.float32) - self.midx
        fv = self.y.astype(np.float32) - self.midy
        mag, ang = cv2.cartToPolar(fu, fv)
        hsv = np.ones([self.ydim, self.xdim, 3], dtype=np.uint8)*255
        hsv[self.y, self.x, 0] = ang[...,0]*180/np.pi/2
        self.hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)/255.
        
        
    def get_lastgray(self, img):
        if hash(img.data.tobytes()) == self.last_fprev_hash:
            return self.last_fprev
        else:
            return None
        
    def cache_lastgray(self, img, fnext):
        self.last_fprev = fnext
        self.last_fprev_hash = hash(img.data.tobytes())
        
    def vizflow(self, flow, str_mod=1):
        vis = np.zeros([VIS_SIZE,VIS_SIZE,3], dtype=np.float32)
        flow_mag = np.linalg.norm(flow[self.y,self.x], axis=-1)
        moving = flow_mag>0.25 # filter down to reduce work on non-movement
        fy,fx = self.y[moving], self.x[moving]
        #fy,fx = self.y, self.x
        if moving.sum()>100:
            flist = flow[fy,fx].reshape(-1,2)
            uvs = flow[fy,fx].T *VIS_SCALE
            locs = (np.clip(uvs, -VIS_SIZE/2, VIS_SIZE/2-1)+VIS_SIZE/2).astype(int)
            px = (locs[1],locs[0])
            
            # sum up flow heatmap to xy grid
            np.add.at(vis, px, self.STRENGTH*str_mod*self.hue[fy,fx]) # at allows repeated indices
            vis = cv2.GaussianBlur(vis, (KERN_SIZE, KERN_SIZE), 1.) # smooth the heatmap
        return (np.clip(vis,0,1.)*255).astype(np.uint8)
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['base', 'last', 'abs_delta']):
            return False
        lframe = aux_imgs['last']
        frame = aux_imgs['base']
        abs_delta = aux_imgs['abs_delta'] # abs_delta to check moving
        
        # movement threshold
        moving = False
        VAL_THRES = 20
        COUNT_THRES = int(0.003*self.xdim*self.ydim*self.depth)
        if (abs_delta > VAL_THRES).sum() > COUNT_THRES:
            moving = True
            pub.sendMessage(ImgEvents.APPEND, key='moving', imgdata=[True])
        
        if moving: # only calculate optic flow if sufficient movement
            fnext = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE).astype(np.int16)
            fprev = self.get_lastgray(lframe)
            if fprev is None:
                fprev = cv2.resize(cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY),
                                    (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE).astype(np.int16)
            self.cache_lastgray(frame, fnext)
            flow_reduce = cv2.calcOpticalFlowFarneback(fprev, fnext, None, 0.5, 6, 35, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = cv2.resize(flow_reduce*DOWNSCALE, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        else:
            flow = np.zeros_like(frame)[...,:2]
        
        pub.sendMessage(ImgEvents.APPEND, key='flow', imgdata=flow)
        return True

def get_avg_flow(flow):
    flist = flow.reshape(-1,2)
    #cent = np.mean(flist,0) # clustering center is more accurate than mean
    
    regress = BisectingKMeans(n_clusters=CLUSTERS, bisecting_strategy='biggest_inertia')
    km = regress.fit(flist)
    #KMN = km.n_clusters_
    KMN = CLUSTERS
    count = [(km.labels_==n).sum() for n in range(KMN)]
    maxcount = max(count)
    modes = [i for i in range(KMN) if count[i]==maxcount]
    
    for ix in modes:
        cent = np.mean(flist[km.labels_==ix],0)
        vals = regress.transform(flist[km.labels_==ix])[:,ix]
        vari = 0
        if len(vals)>3:
            vari = np.var(vals)
        return cent, vari

instance = OpticFlow()
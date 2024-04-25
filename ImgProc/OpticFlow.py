import math
import time
import numpy as np
import cv2
from sklearn.cluster import BisectingKMeans

from ImgProc import ImgTask
#from ImgProc import Delta # dependency for generation

CLUSTERS = 5
DOWNSCALE = 3
class OpticFlow(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.last_fprev = np.array([])
        self.last_fprev_hash = 0
        
        self.flow = np.zeros([int(self.ydim//DOWNSCALE),
                            int(self.xdim//DOWNSCALE),2], dtype=np.float32)
        
    def get_lastgray(self, img):
        if hash(img.data.tobytes()) == self.last_fprev_hash:
            return self.last_fprev
        else:
            return None
        
    def cache_lastgray(self, img, fnext):
        self.last_fprev = fnext
        self.last_fprev_hash = hash(img.data.tobytes())
        
    def requires(self):
        return [ImgTask.IMG_BASE, ImgTask.IMG_LAST, ImgTask.IMG_ABSD]
        
    def outputs(self):
        return [ImgTask.IMG_FLOW]
        
    def proc_frame(self, frame, lframe, abs_delta):        
        # movement threshold
        moving = True
        VAL_THRES = 20
        COUNT_THRES = int(0.003*self.xdim*self.ydim*self.depth)
        if (abs_delta > VAL_THRES).sum() > COUNT_THRES:
            moving = True
        
        if moving: # only calculate optic flow if sufficient movement
            tst = time.time_ns()
            fnext = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE)
            fprev = self.get_lastgray(lframe)
            if fprev is None:
                fprev = cv2.resize(cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY),
                                    (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE)
            self.cache_lastgray(frame, fnext)
            self.flow = cv2.calcOpticalFlowFarneback(fprev, fnext, self.flow, 
                        pyr_scale=0.5, levels=6, winsize=25, 
                        iterations=3, poly_n=3, poly_sigma=1.1, 
                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW )
            
            flow = cv2.resize(self.flow*DOWNSCALE, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            ten = time.time_ns()
            #print ('flow= %3.3fms'%((ten-tst)/1e6))
        else:
            flow = np.zeros_like(frame, dtype=float)[...,:2]
        
        return flow

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
import math
import time
import numpy as np
import cv2

from ImgProc import ImgTask

DEBUG=True # module is only used for debugging

DOWNSCALE = 2
class VizPredict(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.y,self.x = np.mgrid[0:self.ydim:DOWNSCALE, 0:self.xdim:DOWNSCALE].reshape(2,-1).astype(np.uint16)
        self.xy = np.zeros([self.ydim*self.xdim//(DOWNSCALE**2), 2], dtype=np.int16)
        self.xy[...,0] = self.x
        self.xy[...,1] = self.y
            
    def requires(self):
        return [ImgTask.IMG_BASE, ImgTask.IMG_LAST, ImgTask.IMG_FLOW, ImgTask.IMG_ABSD]
        
    def outputs(self):
        return [ImgTask.IMG_DEBUG]
    
    def proc_frame(self, img, lframe, flow, imgdelta):
        #self.get_changed_regions(imgdelta)
        
        no_move = np.sum(imgdelta, axis=-1)<15
        static = no_move[self.y,self.x]
        
        flint = (flow[self.y,self.x]).round().astype(np.int16)
        flint[static] = 0
        # turn flow into prediction image
        uvmap = (self.xy-flint)
        np.clip(uvmap[:,0], 0, self.xdim-1, out=uvmap[:,0])
        np.clip(uvmap[:,1], 0, self.ydim-1, out=uvmap[:,1])
        viz = lframe[uvmap[:,1], uvmap[:,0]]
        
        #delta = (img[self.y,self.x] - viz).reshape(self.ydim//DOWNSCALE,self.xdim//DOWNSCALE,3)
        #delta_norm = np.linalg.norm(delta, axis=-1)
        #delta_thres = (delta_norm>110).astype(np.uint8)
        viz = viz.reshape(self.ydim//DOWNSCALE,self.xdim//DOWNSCALE,3)
        
        return viz #, no_move.astype(np.uint8)*255

_ = VizPredict()
import math
import time
import numpy as np
import cv2

from ImgProc import ImgTask

DEBUG=True # module is only used for debugging

class VizHeat:
    def __init__(self, size=400, span=80, kern_size=15): # 80 px movement is full span
        self.VIS_SIZE = size
        self.VIS_SCALE = self.VIS_SIZE/span
        self.KERN_SIZE = kern_size
        
    def set_span(self, span):
        self.VIS_SCALE = self.VIS_SIZE/span
        
    def vizflow(self, flow, hues=None, str_mod=1):
        if hues is None:
            hues = np.array([1,1,1], dtype=np.float32)
        box = np.zeros([self.VIS_SIZE,self.VIS_SIZE,3], dtype=np.float32)
        uvs = flow.T *self.VIS_SCALE
        locs = (np.clip(uvs, -self.VIS_SIZE/2, self.VIS_SIZE/2-1)+self.VIS_SIZE/2).astype(int)
        px = (locs[1],locs[0])
        
        # sum up flow heatmap to xy grid
        np.add.at(box, px, str_mod*hues) # at allows repeated indices
        box = cv2.GaussianBlur(box, (self.KERN_SIZE, self.KERN_SIZE), 1.) # smooth the heatmap
        return (np.clip(box,0,1.)*255).astype(np.uint8)

    def proc_frame(self, viz, data, hues=None, str_mod=1):
        heatmap = self.vizflow(data, hues, str_mod)
        viz[:heatmap.shape[0],:heatmap.shape[1]] = heatmap
        return viz

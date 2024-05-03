import math
import time
import numpy as np
import cv2

from ImgProc import ImgTask
from ImgProc.VizHeat import VizHeat

DEBUG=True # module is only used for debugging

SPAN = 50
class VizMoveHist(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
                
        self.STRENGTH = 5
        self.hue = np.array([1,1,1], dtype=np.float32) * self.STRENGTH
        self.vizheat = VizHeat(span=SPAN)
            
    def requires(self):
        return [ImgTask.IMG_BASE, 'move_hist']
        
    def outputs(self):
        return [ImgTask.IMG_DEBUG]
        
    def proc_frame(self, frame, move_hist):
        LINE_GRAPH = True
        viz = np.array(frame)
        hues = np.repeat(self.hue[None,:], move_hist.shape[0], axis=0)
        mag = np.linalg.norm(move_hist, axis=-1)
        if LINE_GRAPH and mag.shape[0]>2: # temporal graph
            dx1 = mag[1:] - mag[:-1]
            dx2 = dx1[1:] - dx1[:-1]
            move_coord = np.repeat(SPAN/2-mag[:,None], 2, axis=-1)
            move_coord[:,0] = np.linspace(-SPAN/2, SPAN/2, move_hist.shape[0])
            dx1_coord = np.repeat(-dx1[:,None], 2, axis=-1)
            dx1_coord[:,0] = np.linspace(-SPAN/2, SPAN/2, dx1.shape[0])
            dx2_coord = np.repeat(-dx2[:,None], 2, axis=-1)
            dx2_coord[:,0] = np.linspace(-SPAN/2, SPAN/2, dx2.shape[0])
            dx1_hues = np.repeat([[0,0,1]], dx1_coord.shape[0], axis=0)*self.STRENGTH
            dx2_hues = np.repeat([[0,1,0]], dx2_coord.shape[0], axis=0)*self.STRENGTH
            
            coords = np.concatenate((move_coord, dx1_coord, dx2_coord), axis=0)
            allhues = np.concatenate((hues, dx1_hues, dx2_hues), axis=0)
            #print (move_coord)
            #self.vizheat.proc_frame(viz, coords, allhues)
            self.vizheat.proc_frame(viz, move_coord, hues)
        else: # polar graph
            bright_scale = (np.arange(move_hist.shape[0])**2/(move_hist.shape[0]**2))[:,None]
            hues *= bright_scale
            hues[-1,:] = np.array([0,0,1], dtype=np.float32) * self.STRENGTH
            mag_adjust = np.clip(5*np.log(mag+1e-6),0,None) * 1/(mag+1e-6)
            #mag_adjust = 20./(mag+1e-6)
            self.vizheat.proc_frame(viz, move_hist*mag_adjust[:,None], hues)
        
        return viz

instance = VizMoveHist()
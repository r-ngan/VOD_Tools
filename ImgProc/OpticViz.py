import math
import time
from pubsub import pub
import numpy as np
import cv2

from ImgProc import ImgEvents, Preprocessor
from ImgProc import OpticFlow # dependency for generation

DOWNSCALE = 2
class OpticViz(Preprocessor.Preprocessor):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.y,self.x = np.mgrid[0:self.ydim:DOWNSCALE, 0:self.xdim:DOWNSCALE].reshape(2,-1).astype(np.uint16)
        self.xy = np.zeros([self.ydim*self.xdim//(DOWNSCALE**2), 2], dtype=np.int16)
        self.xy[...,0] = self.x
        self.xy[...,1] = self.y
        
    def get_changed_regions(self, delta):
        KERN_SIZE = 35
        motion = np.sum(delta, axis=-1)>55
        frame = np.zeros_like(delta, dtype=float)
        frame[motion] = 1
        frame = cv2.GaussianBlur(frame, (KERN_SIZE, KERN_SIZE), 1.)
        
        pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=frame)
        return None
    
    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['flow', 'abs_delta']):
            return False
        lframe = aux_imgs['last']
        flow = aux_imgs['flow']
        imgdelta = aux_imgs['abs_delta']
        
        
        tst = time.time_ns()
        self.get_changed_regions(imgdelta)
        
        no_move = np.sum(imgdelta, axis=-1)<15
        static = no_move[self.y,self.x]
        
        flint = (flow[self.y,self.x]).round().astype(np.int16)
        flint[static] = 0
        # turn flow into prediction image
        uvmap = (self.xy-flint)
        np.clip(uvmap[:,0], 0, self.xdim-1, out=uvmap[:,0])
        np.clip(uvmap[:,1], 0, self.ydim-1, out=uvmap[:,1])
        viz = lframe[uvmap[:,1], uvmap[:,0]].astype(np.int16)
        
        delta = (img[self.y,self.x] - viz).reshape(self.ydim//DOWNSCALE,self.xdim//DOWNSCALE,3)
        delta_norm = np.linalg.norm(delta, axis=-1)
        delta_thres = (delta_norm>110).astype(np.uint8)
        ten = time.time_ns()
        print ('optviz= %3.3fms'%((ten-tst)/1e6))
        viz = viz.reshape(self.ydim//DOWNSCALE,self.xdim//DOWNSCALE,3)
        
        
        pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=imgdelta)
        pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=no_move.astype(np.uint8)*255)
        
        if True:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(img)
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(np.sqrt(mag)*40,0,255)
            flowviz = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=flowviz)
        return True

_ = OpticViz()
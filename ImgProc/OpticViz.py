import math
import time
from pubsub import pub
import numpy as np
import cv2

from ImgProc import ImgEvents, Preprocessor
from ImgProc import OpticFlow # dependency for generation

class OpticViz(Preprocessor.Preprocessor):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.y,self.x = np.mgrid[0:self.ydim, 0:self.xdim].reshape(2,-1).astype(int)
        self.xy = np.zeros([self.ydim*self.xdim, 2], dtype=int)
        self.xy[...,0] = self.x
        self.xy[...,1] = self.y
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['flow', 'delta']):
            return False
        lframe = aux_imgs['last']
        flow = aux_imgs['flow']
        
        viz = np.zeros_like(img)
        flint = np.array(flow.reshape(-1,2).round(), dtype=int)
        # turn flow into prediction image
        uvmap = (self.xy-flint)
        np.clip(uvmap[:,0], 0, self.xdim-1, out=uvmap[:,0])
        np.clip(uvmap[:,1], 0, self.ydim-1, out=uvmap[:,1])
        viz[self.y,self.x] = lframe[uvmap[:,1], uvmap[:,0]]
        
        pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=viz)
        
        delta = img - viz.astype(np.int16) # allow negative range, >128 delta
        pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=delta*np.abs(delta//2))
        pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=aux_imgs['delta']*120)
        
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
import math
import time
import numpy as np
import cv2

from ImgProc import ImgTask

DEBUG=True # module is only used for debugging
            
VIS_SIZE = 400
VIS_SCALE = VIS_SIZE/ 80. # 80 px movement is full span
KERN_SIZE = 15
class VizFlow(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
                
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
            
    def requires(self):
        return ['flow']
        
    def outputs(self):
        return ['debug']
        
    def proc_frame(self, flow):
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros([self.ydim, self.xdim, 3], dtype=np.uint8)
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(np.sqrt(mag)*40,0,255)
        viz1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
        # show flow arrows on debug
        YGRID = 30
        XGRID = 30
        y,x = np.mgrid[YGRID/2:self.ydim:YGRID, XGRID/2:self.xdim:XGRID].reshape(2,-1).astype(int)
        u,v = (x+flow[y,x,0]).round().astype(int), (y+flow[y,x,1]).round().astype(int)
        for x1,y1,x2,y2 in [(a,b,c,d) for a,b,c,d in zip (x,y, u,v) if a!=c or b!=d]:
            cv2.line(viz1, (x1,y1), (x2,y2), (0,255,0), 1)
            
        heatmap = self.vizflow(flow, str_mod=0.65)
        viz1[:heatmap.shape[0],:heatmap.shape[1]] = heatmap
        
        return viz1

instance = VizFlow()
import math
import time
import numpy as np
import cv2
import torch

from ImgProc import ImgTask
from ImgProc import VizFlow, MotionAnalyzer # dependency for generation

DEBUG=True # module is only used for debugging

class VizMotion(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.PTDEV = torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        #    self.PTDEV = torch.device('cuda')
                
        self.xy_map = np.mgrid[0:self.ydim, 0:self.xdim].astype(float)
        self.allfy, self.allfx = self.xy_map.reshape(2,-1).astype(int)
        self.xy_map[0,:] -= self.midy
        self.xy_map[1,:] -= self.midx
        self.xy_map[0,:] /= self.ydim # normalize to 0.5 = half height
        self.xy_map[1,:] /= self.xdim
        self.tensxy = torch.stack([torch.tensor(self.xy_map[1], device=self.PTDEV),
                                torch.tensor(self.xy_map[0], device=self.PTDEV)],dim=-1)
                            
        self.rand = np.random.rand(self.ydim, self.xdim)*40+15 # "fixed" random Z-data
        self.Z = torch.tensor(self.rand[self.allfy,self.allfx], device=self.PTDEV)
        
    def requires(self):
        return ['pred_cam']
        
    def outputs(self):
        return ['debug']
        
    def proc_frame(self, pred_cam):
        TX = np.zeros(3)
        
        if pred_cam is None:
            viz = np.zeros([self.ydim, self.xdim, 3], dtype=np.uint8)
        else:
            bang = torch.tensor([0, pred_cam[0]], device=self.PTDEV)
            dang = torch.tensor([pred_cam[1], pred_cam[2]], device=self.PTDEV)
            Z = self.Z
            XY = self.tensxy[self.allfy,self.allfx]
            flow2 = MotionAnalyzer.instance.sim_motion(Z, XY, bang, dang, TX).reshape(self.ydim, self.xdim, 2)
            flow2 = flow2.data.cpu().numpy()
            
            flow2[self.midy+200:,::2] = 0 # crop some symmetry away for better visualization
            mag, ang = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
            hsv = np.zeros([self.ydim, self.xdim, 3], dtype=np.uint8)
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(np.sqrt(mag)*40,0,255)
            viz = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # show flow arrows on debug
            YGRID = 30
            XGRID = 30
            y,x = np.mgrid[YGRID/2:self.ydim:YGRID, XGRID/2:self.xdim:XGRID].reshape(2,-1).astype(int)
            u,v = (x+flow2[y,x,0]).round().astype(int), (y+flow2[y,x,1]).round().astype(int)
            for x1,y1,x2,y2 in [(a,b,c,d) for a,b,c,d in zip (x,y, u,v) if a!=c or b!=d]:
                cv2.line(viz, (x1,y1), (x2,y2), (0,255,0), 1)
            heatmap = VizFlow.instance.vizflow(flow2, str_mod=0.65)
            viz[:heatmap.shape[0],:heatmap.shape[1]] = heatmap
        
        return viz

instance = VizMotion()
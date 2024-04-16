import math
import time
from pubsub import pub
import numpy as np
import cv2
import torch
from scipy import optimize
from scipy.special import huber

import VideoAnalysis
import VODEvents
import VODState

from ImgProc import ImgEvents
from ImgProc import OpticFlow # dependency for generation

DEBUG=False
            
XFACTOR = 0.
YFACTOR = 0.
ZFACTOR = 0.
HFOV = 103.*(np.pi/180)
VFOV = 71.*(np.pi/180)
ASPECT = 1080./1920

# uses flow information to estimate movement
class FlowAnalyzer(VideoAnalysis.Analyzer):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
                
        # subsample to increase speed
        XSTEP = 45
        YSTEP = 25
        TRIM_Y1 = 90
        TRIM_Y2 = 90
        self.y,self.x = np.mgrid[TRIM_Y1+YSTEP/2:self.ydim-TRIM_Y2:YSTEP, XSTEP/2:self.xdim:XSTEP]. \
                        reshape(2,-1).astype(int)
        self.xy_map = np.mgrid[0:self.ydim, 0:self.xdim].astype(float)
        self.allfy, self.allfx = self.xy_map.reshape(2,-1).astype(int)
        self.xy_map[0,:] -= self.midy
        self.xy_map[1,:] -= self.midx
        self.xy_map /= self.ydim # normalize to 0.5 = half height
        self.xy_map[1,:] *= ASPECT
        self.tensxy = torch.stack([torch.tensor(self.xy_map[1]), torch.tensor(self.xy_map[0])],dim=-1)
                            
        self.base_axis = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=float).T
        # projection scalars
        tanfovx = 2*np.tan(HFOV/2)
        tanfovy = 2*np.tan(VFOV/2)
        self.fov = torch.tensor([tanfovx,tanfovy,1])
        self.rand = np.random.rand(self.ydim, self.xdim)*40+15 # "fixed" random Z-data
        
        self.base_ang = np.array([0,0],dtype=np.float32)
        self.delta_ang = np.array([0,0],dtype=np.float32)
        self.mouse_xy = np.array([0,0],dtype=np.float32)
        self.moving = False
        
    def get_rot_mat(self, yaw, pitch): # roll is not used in fps
        t0 = torch.tensor(0., dtype=float)
        t1 = torch.tensor(1., dtype=float)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        ymat = torch.stack([torch.stack([cy,t0,sy]),
                        torch.stack([t0,t1,t0]),
                        torch.stack([-sy,t0,cy])])
        pmat = torch.stack([torch.stack([t1,t0,t0]),
                        torch.stack([t0,cp,-sp]),
                        torch.stack([t0,sp,cp]),])
        return ymat @ pmat
        
    # simulate the motion we would see with known angles / translations
    # Use this as basis for optimizer
    def sim_motion(self, Z, XY, base_ang=(torch.tensor(0.),torch.tensor(0.)), rot_ang=(torch.tensor(0.),torch.tensor(0.)), translate=[0,0,0]):
        
        # delta yaw is affected by base pitch due to gimbal lock issues, so need to guess the base pitch
        baserot = self.get_rot_mat(0*base_ang[0], -base_ang[1]) # base rotation always yaw=0
        axis = baserot @ self.base_axis # align axis for later projection
        deltarot = self.get_rot_mat(-rot_ang[0], rot_ang[1])
        TX = torch.tensor(translate)
        
        # create screen to world map
        pad = torch.ones(XY.shape[0], 1) 
        pts = torch.cat((XY,pad),1) # X = x*Z
        pts = pts * Z[:, None] * self.fov
        pts = baserot @ (pts.T)

        new_pts = (deltarot @ pts).T + TX # turn back to (...,3) for inner product
        proj = torch.inner(new_pts, axis.T)
        proj = proj/proj[..., 2:] / self.fov # UVW/W
        
        # delta flow in screen space
        # new x - orig x
        scr_scale = torch.tensor([self.xdim, self.ydim])
        flow = (proj[:,:2]-XY)*scr_scale
        
        return flow
        
    def flow_loss(self, x, *args):
        x = torch.tensor(x, requires_grad=True)
        bpitch, dyaw, dpitch = x
        byaw = torch.tensor(0.)
        flow, Z, XY = args
        
        simflow = self.sim_motion(Z, XY, (byaw,bpitch), (dyaw,dpitch))
        dx = (simflow-flow)
        flow_mag = (torch.norm(dx, dim=-1))**0.65 # squared error biases towards outliers, flatten a bit
        result = flow_mag.sum()
        result.backward()
        return result.detach(), x.grad
        
    def get_cam_params(self, flow, flow_est=None):
        solvers = ['Nelder-Mead',
            'Powell',
            'CG',
            'BFGS',
            'Newton-CG',
            'L-BFGS-B',
            'TNC',
            'COBYLA',
            'SLSQP',
            'trust-constr',
            'dogleg',
            'trust-ncg',
            'trust-exact',
            'trust-krylov',]
        flow_mag = np.linalg.norm(flow[self.y,self.x], axis=-1)
        moving = flow_mag>0.25 # filter down to reduce work on non-movement
        if moving.sum()>1000:
            fy = self.y[moving]
            fx = self.x[moving]
            fy[-1] = self.midy
            fx[-1] = self.midx
            
            # x0 = base pitch, delta yaw, delta pitch,
            bounds = [(-np.pi/2, np.pi/2),(-1, 1),(-1, 1),]
                    #(-1, 1),(-1, 1),(-1, 1),]
            guess = np.zeros(len(bounds),dtype=float)
            guess[0] = self.base_ang[1]
            guess[1] = self.delta_ang[0]
            guess[2] = self.delta_ang[1]
            
            inflow = torch.tensor(flow[fy,fx])
            Z = torch.tensor(self.rand[fy,fx])
            XY = self.tensxy[fy,fx]
            fit = optimize.minimize(self.flow_loss, method=solvers[5], 
                                    x0=guess, args=(inflow, Z, XY),
                                    jac=True, bounds=bounds)
            
            base_ang = (0, fit.x[0])
            delta_ang = (fit.x[1], fit.x[2])
            self.base_ang += delta_ang # dead reckoning for base angle (TODO: weigh in predicted base angle)
            
            return fit, np.array([0,0])
        else:
            return None, None
            
    def proc_frame(self, timestamp, img, aux_imgs={}):
        super().proc_frame(timestamp=timestamp, img=img, aux_imgs=aux_imgs)
        frame = img
        flow = aux_imgs['flow']
        abs_delta = aux_imgs['abs_delta'] # abs_delta to check moving
        # movement threshold
        self.moving = False
        VAL_THRES = 20
        COUNT_THRES = int(0.003*self.xdim*self.ydim*self.depth)
        if (abs_delta > VAL_THRES).sum() > COUNT_THRES:
            self.moving = True
        flow2 = np.zeros_like(flow)
        base_ang = np.zeros(2)
        delta_ang = np.zeros(2)
        mouse_xy = np.zeros(2)
        TX = np.zeros(3)
        fit = None
        if DEBUG:
            viz1 = np.zeros_like(frame)
            viz2 = np.zeros_like(frame)
        if self.moving: # only calculate optic flow if sufficient movement
            tst = time.time_ns()
            fit, mxy = self.get_cam_params(flow, flow2)
            ten = time.time_ns()
            #print ('cam_params= %3.3fms'%((ten-tst)/1e6))

            if fit is not None:
                mouse_xy = mxy
                base_ang = (0, fit.x[0])
                delta_ang = np.array([fit.x[1], fit.x[2]])
                cam_params = -delta_ang
                pub.sendMessage(ImgEvents.APPEND, key='pred_cam', imgdata=cam_params)
            
            if DEBUG:
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(frame)
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 1] = 255
                hsv[..., 2] = np.clip(np.sqrt(mag)*40,0,255)
                viz1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                bang = torch.tensor(base_ang)
                dang = torch.tensor(delta_ang)
                Z = torch.tensor(self.rand[self.allfy,self.allfx])
                XY = self.tensxy[self.allfy,self.allfx]
                flow2 = self.sim_motion(Z, XY, bang, dang, TX).reshape(self.ydim, self.xdim, 2)
                flow2 = flow2.data.cpu().numpy()
                
                flow2[self.midy+200:,::2] = 0 # crop some symmetry away for better visualization
                mag, ang = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
                hsv = np.zeros_like(frame)
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 1] = 255
                hsv[..., 2] = np.clip(np.sqrt(mag)*40,0,255)
                viz2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
        self.delta_ang = delta_ang
        self.mouse_xy = mouse_xy
        if DEBUG:
            # show flow arrows on debug
            YGRID = 30
            XGRID = 30
            y,x = np.mgrid[YGRID/2:self.ydim:YGRID, XGRID/2:self.xdim:XGRID].reshape(2,-1).astype(int)
            u,v = (x+flow[y,x,0]).round().astype(int), (y+flow[y,x,1]).round().astype(int)
            for x1,y1,x2,y2 in [(a,b,c,d) for a,b,c,d in zip (x,y, u,v) if a!=c or b!=d]:
                cv2.line(viz1, (x1,y1), (x2,y2), (0,255,0), 1)
                
            heatmap = OpticFlow.instance.vizflow(flow, str_mod=0.65)
            viz1[:heatmap.shape[0],:heatmap.shape[1]] = heatmap
            
            u,v = (x+flow2[y,x,0]).round().astype(int), (y+flow2[y,x,1]).round().astype(int)
            for x1,y1,x2,y2 in [(a,b,c,d) for a,b,c,d in zip (x,y, u,v) if a!=c or b!=d]:
                cv2.line(viz2, (x1,y1), (x2,y2), (0,255,0), 1)
            heatmap = OpticFlow.instance.vizflow(flow2, str_mod=0.65)
            viz2[:heatmap.shape[0],:heatmap.shape[1]] = heatmap
            
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=viz1)
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=viz2)
            
        flow2[...,:] = delta_ang*180/np.pi # show delta in degrees
        pub.sendMessage(ImgEvents.APPEND, key='flow2', imgdata=flow2)

# does not publish any events, used as a superclass
#instance = FlowAnalyzer()
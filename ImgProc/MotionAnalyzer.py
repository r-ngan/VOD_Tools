import math
import time
from pubsub import pub
import numpy as np
import cv2
import torch
from scipy import optimize
from scipy.special import huber

from ImgProc import ImgTask
            
XFACTOR = 0.
YFACTOR = 0.
ZFACTOR = 0.
HFOV = 103.*(np.pi/180)
VFOV = 71.*(np.pi/180)
N_CAM_PARAMS = 3
# uses flow information to estimate movement
class MotionAnalyzer(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.PTDEV = torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        #    self.PTDEV = PTDEV = torch.device('cuda')
        # optimizer is not working well with GPU-CPU thrashing
                
        # subsample to increase speed
        XSTEP = 45
        YSTEP = 25
        TRIM_Y1 = 90
        TRIM_Y2 = 90
        self.y,self.x = np.mgrid[TRIM_Y1+YSTEP/2:self.ydim-TRIM_Y2:YSTEP, XSTEP/2:self.xdim:XSTEP]. \
                        reshape(2,-1).astype(int)
        xy_map = np.mgrid[0:self.ydim, 0:self.xdim].astype(float)
        xy_map[0,:] -= self.midy
        xy_map[1,:] -= self.midx
        xy_map[0,:] /= self.ydim # normalize to 0.5 = half height
        xy_map[1,:] /= self.xdim
        self.tensxy = torch.stack([torch.tensor(xy_map[1], device=self.PTDEV), 
                                    torch.tensor(xy_map[0], device=self.PTDEV)],dim=-1)
                            
        self.base_axis = torch.eye(3, dtype=float, device=self.PTDEV)
        # projection scalars
        tanfovx = 2*np.tan(HFOV/2)
        tanfovy = 2*np.tan(VFOV/2)
        self.fov = torch.tensor([tanfovx,tanfovy,1], device=self.PTDEV)
        self.rand = np.random.rand(self.ydim, self.xdim)*40+15 # "fixed" random Z-data
        
        self.base_ang = np.array([0,0],dtype=np.float32)
        self.delta_ang = np.array([0,0],dtype=np.float32)
        self.mouse_xy = np.array([0,0],dtype=np.float32)
        self.moving = False
        
    def get_rot_mat(self, yaw, pitch): # roll is not used in fps
        t0 = torch.tensor(0., dtype=float, device=self.PTDEV)
        t1 = torch.tensor(1., dtype=float, device=self.PTDEV)
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
    def sim_motion(self, Z, XY, base_ang=None, rot_ang=None, translate=[0,0,0]):
        if base_ang is None:
            base_ang=(torch.tensor(0., device=self.PTDEV),
                        torch.tensor(0., device=self.PTDEV))
        if rot_ang is None:
            rot_ang=(torch.tensor(0., device=self.PTDEV),
                    torch.tensor(0., device=self.PTDEV))
        # delta yaw is affected by base pitch due to gimbal lock issues, so need to guess the base pitch
        baserot = self.get_rot_mat(0*base_ang[0], -base_ang[1]) # base rotation always yaw=0
        axis = baserot @ self.base_axis # align axis for later projection
        deltarot = self.get_rot_mat(-rot_ang[0], rot_ang[1])
        TX = torch.tensor(translate, device=self.PTDEV)
        
        # create screen to world map
        pad = torch.ones(XY.shape[0], 1, device=self.PTDEV) 
        pts = torch.cat((XY,pad),1) # X = x*Z
        pts = pts * Z[:, None] * self.fov
        pts = baserot @ (pts.T)

        new_pts = (deltarot @ pts).T + TX # turn back to (...,3) for inner product
        proj = torch.inner(new_pts, axis.T)
        proj = proj/proj[..., 2:] / self.fov # UVW/W
        
        # delta flow in screen space
        # new x - orig x
        scr_scale = torch.tensor([self.xdim, self.ydim], device=self.PTDEV)
        flow = (proj[:,:2]-XY)*scr_scale
        
        return flow
        
    def flow_loss(self, x, *args):
        x = torch.tensor(x, device=self.PTDEV, requires_grad=True)
        bpitch, dyaw, dpitch = x
        byaw = torch.tensor(0., device=self.PTDEV)
        flow, Z, XY = args
        
        simflow = self.sim_motion(Z, XY, (byaw,bpitch), (dyaw,dpitch))
        dx = (simflow-flow)
        flow_mag = (torch.norm(dx, dim=-1))**0.65 # squared error biases towards outliers, flatten a bit
        result = flow_mag.sum()
        result.backward()
        return result.detach().cpu(), x.grad.cpu()
        
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
            guess = np.zeros(N_CAM_PARAMS, dtype=float)
            guess[0] = self.base_ang[1]
            guess[1] = self.delta_ang[0]
            guess[2] = self.delta_ang[1]
            
            inflow = torch.tensor(flow[fy,fx], device=self.PTDEV)
            Z = torch.tensor(self.rand[fy,fx], device=self.PTDEV)
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
            
    def requires(self):
        return [ImgTask.IMG_FLOW, ImgTask.IMG_ABSD]
        
    def outputs(self):
        return ['pred_cam']
        
    def proc_frame(self, flow, abs_delta):
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
        cam_params = np.zeros(N_CAM_PARAMS)
        if self.moving: # only calculate optic flow if sufficient movement
            tst = time.time_ns()
            fit, mxy = self.get_cam_params(flow, flow2)
            ten = time.time_ns()
            #print ('cam_params= %3.3fms'%((ten-tst)/1e6))

            if fit is not None:
                cam_params[:] = fit.x
                mouse_xy = mxy
                base_ang = (0, fit.x[0])
                delta_ang = np.array([fit.x[1], fit.x[2]])
                
        self.delta_ang = delta_ang
        self.mouse_xy = mouse_xy
            
        return cam_params

instance = MotionAnalyzer()
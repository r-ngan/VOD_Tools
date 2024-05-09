import math
import time
from pubsub import pub
import numpy as np
import cv2
import torch

from ImgProc import ImgTask

HFOV = 103.*(np.pi/180)
VFOV = 71.*(np.pi/180)
# Motion discriminator. Use assumptions that cam / translation are generally separate
# fit assuming one is zero. If the error is high, then try multi-fit
class MotionDisc(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        if ImgTask.CUDA and torch.cuda.is_available():
            torch.set_default_device('cuda')
        else:
            torch.set_default_device('cpu')
        # optimizer is not working well with GPU-CPU thrashing
                
        # subsample to increase speed
        self.XSTEP = XSTEP = self.xdim//40
        self.YSTEP = YSTEP = self.ydim//40
        TRIM_Y1 = self.ydim//10 # 108px at 1080p
        TRIM_Y2 = self.ydim//10
        yx = np.mgrid[TRIM_Y1+YSTEP/2:self.ydim-TRIM_Y2:YSTEP, XSTEP/2:self.xdim:XSTEP]
        self.y,self.x = yx.reshape(2,-1).astype(int)
        
        expand_yx = cv2.resize(np.moveaxis(yx,0,-1), (0, 0), fx=XSTEP, fy=YSTEP, interpolation=cv2.INTER_NEAREST)
        self.exp_y, self.exp_x = np.moveaxis(expand_yx,-1,0).reshape(2,-1).astype(int)
        
        self.trim_y,self.trim_x = np.mgrid[TRIM_Y1:self.ydim-TRIM_Y2, 0:self.xdim].reshape(2,-1).astype(int)
        
        xy_map = np.mgrid[0:self.ydim, 0:self.xdim].astype(float)
        xy_map[0,:] -= self.midy
        xy_map[1,:] -= self.midx
        xy_map[0,:] /= self.ydim # normalize to 0.5 = half height
        xy_map[1,:] /= self.xdim
        self.tensxy = torch.stack([torch.tensor(xy_map[1], dtype=torch.float64), 
                                    torch.tensor(xy_map[0], dtype=torch.float64)],dim=-1)
                            
        self.base_axis = torch.eye(3, dtype=float)
        # projection scalars
        tanfovx = 2*np.tan(HFOV/2)
        tanfovy = 2*np.tan(VFOV/2)
        self.fov = torch.tensor([tanfovx,tanfovy,1])
        self.rand = np.random.rand(self.ydim, self.xdim)*0+0.001 # "fixed" random Z-data
        
        self.base_ang = np.array([0,0],dtype=np.float32)
        self.delta_ang = np.array([0,0],dtype=np.float32)
        self.mouse_xy = np.array([0,0],dtype=np.float32)
        self.moving = False
        
        self.last_cam = torch.zeros(3, dtype=torch.float64)
        self.last_TX = torch.zeros(3, dtype=torch.float64)
        self.last_Z = 0.001*torch.ones([self.ydim, self.xdim], dtype=torch.float64)
        
        self.move_thres = int(0.4*self.y.shape[0]) # analyze if more than 40% moving
        
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
        
    def get_2drot_mat(self, angle):
        t0 = torch.tensor(0., dtype=float)
        t1 = torch.tensor(1., dtype=float)
        cy = torch.cos(angle)
        sy = torch.sin(angle)
        rmat = torch.stack([torch.stack([cy,sy]),
                        torch.stack([-sy,cy])])
        return rmat
        
    # simulate the motion we would see with known angles / translations
    # Use this as basis for optimizer
    def sim_motion(self, Z, XY, base_ang=None, rot_ang=None, TX=None):
        if base_ang is None:
            base_ang=torch.zeros(2)
        if rot_ang is None:
            rot_ang=torch.zeros(2)
        if TX is None:
            TX=torch.zeros(3)
            
        # delta yaw is affected by base pitch due to gimbal lock issues, so need to guess the base pitch
        baserot = self.get_rot_mat(0*base_ang[0], -base_ang[1]) # base rotation always yaw=0
        axis = baserot @ self.base_axis # align axis for later projection
        deltarot = self.get_rot_mat(-rot_ang[0], rot_ang[1])
        
        # create screen to world map
        pad = torch.ones(XY.shape[0], 1) 
        pts = torch.cat((XY,pad),1) # X = x*Z
        pts = (pts / Z[:, None]) * self.fov
        pts = baserot @ (pts.T)

        new_pts = (deltarot @ pts).T + TX # turn back to (...,3) for inner product
        proj = torch.inner(new_pts, axis.T)
        proj = proj/proj[..., 2:] / self.fov # UVW/W
        
        # delta flow in screen space
        # new x - orig x
        scr_scale = torch.tensor([self.xdim, self.ydim])
        flow = (proj[:,:2]-XY)*scr_scale
        
        return flow
        
    def flow_loss(self, flow, cam, Z, TX, XY, *args):
        Z_TARG = 0.1
        bpitch, dyaw, dpitch = cam
        byaw = torch.tensor(0.)
        
        simflow = self.sim_motion(Z, XY, [byaw,bpitch], [dyaw,dpitch], TX)
        dx = (simflow-flow)
        flow_mag = (torch.norm(dx, dim=-1))**0.65 # squared error biases towards outliers, flatten a bit
        ang_dx = 1e1*(bpitch-self.last_cam[0]).abs()**2 # avoid shifting base angle outside of dead reckoning
        result = flow_mag.sum()/flow.shape[0] + ang_dx
        result.backward()
        return result
        
    def line_loss(self, flow, TX):
        r = self.get_2drot_mat(100*TX[0])
        pts = (r @ flow.T).T
        #print (pts)
        #dist = (flow[...,0]*st - flow[...,1]*ct)**2
        direction = pts[...,0].mean()
        result = (pts[...,1].abs()**0.75).sum()/flow.shape[0] # normalize
        result.backward()
        return result, direction.detach()
        
        
    def get_cam_params(self, flow):
        flow_mag = np.linalg.norm(flow[self.y,self.x], axis=-1)
        moving = flow_mag>0.25 # filter down to reduce work on non-movement
        if moving.sum()>self.move_thres:
            fy = self.y[moving]
            fx = self.x[moving]
            #fy[-1] = self.midy
            #fx[-1] = self.midx
            
            inflow = torch.tensor(flow[fy,fx], dtype=torch.float64)
            XY = self.tensxy[fy,fx]
            
            self.last_cam[0] += self.last_cam[2] # dead reckoning of base pitch
            cam = self.last_cam.clone().detach().requires_grad_(True)
            Z = self.last_Z[fy,fx].clone().detach().requires_grad_(True)
            TX = self.last_TX.clone().detach().requires_grad_(True)
            TXang = torch.zeros(1, dtype=torch.float64).requires_grad_(True)
            
            nocam = torch.zeros(3, dtype=torch.float64)
            notx = torch.zeros(3, dtype=torch.float64)
            
            tx_lr = 1e-3
            cam_lr = 5e-3
            Z_lr = 1e-4
            tx_solver = torch.optim.Adam([{'params': TXang, 'lr': tx_lr}])
            Z_solver = torch.optim.Adam([{'params': Z, 'lr': Z_lr}])
            cam_solver = torch.optim.Adam([{'params': cam, 'lr': cam_lr}])
            all_solver = torch.optim.Adam([
                                        {'params': cam, 'lr': cam_lr},
                                        {'params': TX, 'lr': tx_lr},
                                        {'params': Z, 'lr': Z_lr},
                                        ])

            loss = 0.
            mag = torch.zeros(1, dtype=torch.float64)
            for i in range(20):
                tx_solver.zero_grad()
                loss, mag = self.line_loss(inflow, TXang)
                tx_solver.step()
                if mag<-1e-3:
                    with torch.no_grad():
                        TXang += math.pi/100.
                #print (TXang.detach().cpu().numpy()*180/math.pi)
            txloss = loss.detach().cpu().numpy()*2.3
            #print ('tx loss = %s'%(txloss))
            
            for i in range(20):
                cam_solver.zero_grad()
                loss = self.flow_loss(inflow, cam, Z, notx, XY)
                cam_solver.step()
            camloss = loss.detach().cpu().numpy()
            #print ('cam loss = %s'%(camloss))
            
            if txloss<camloss:
                ang = 100*(TXang.detach())
                with torch.no_grad():
                    TX[0] = torch.cos(ang[0]) * mag
                    TX[1] = torch.sin(ang[0]) * mag
                    TX[2] = 0.
                    cam[1:] = 0.
                camloss = 0.
                for i in range(5): # get preliminary Z data
                    Z_solver.zero_grad()
                    loss = self.flow_loss(inflow, cam.detach(), Z, TX.detach(), XY)
                    Z_solver.step()
                #print ('tx only')
            elif camloss<txloss:
                with torch.no_grad():
                    TX[:] = 0.
                txloss = 0.
                #print ('cam only')
                
            if txloss>=1.0 or camloss>=1.2:
                #print ('full solve')
                for i in range(10):
                    all_solver.zero_grad()
                    loss = self.flow_loss(inflow, cam, Z, TX, XY)
                    all_solver.step()
            
            self.last_loss = loss.detach().cpu().numpy()
            self.last_cam[:] = cam.detach()
            self.last_Z[fy,fx] = Z.detach()
            self.last_TX[:] = TX.detach()
        else:
            self.last_loss = 0.
            self.last_cam[1:] = 0.
            self.last_TX[:] = 0.
        return self.last_cam.cpu().numpy()
            
    def expand_Z(self):
        self.last_Z[self.trim_y, self.trim_x] = self.last_Z[self.exp_y,self.exp_x]
            
    def requires(self):
        return [ImgTask.IMG_FLOW, ImgTask.IMG_ABSD, ImgTask.VAL_FRAMENUM]
        
    def outputs(self):
        return ['pred_cam']
        
    def proc_frame(self, flow, abs_delta, frame_num):
        if ImgTask.CUDA and torch.cuda.is_available():
            torch.set_default_device('cuda')
        else:
            torch.set_default_device('cpu')
        # movement threshold
        self.moving = False
        VAL_THRES = 20
        COUNT_THRES = int(0.003*self.xdim*self.ydim*self.depth)
        if (abs_delta > VAL_THRES).sum() > COUNT_THRES:
            self.moving = True
        #dbg_img = np.zeros([self.ydim, self.xdim, 1], dtype=float)
        tst = time.time_ns()
        self.get_cam_params(flow)
        ten = time.time_ns()
        #print ('cam_params= %3.3fms'%((ten-tst)/1e6))
        self.expand_Z()
        #print ('%d  %0.2f %s %s'%(frame_num, self.last_loss, self.last_cam.cpu().numpy(), self.last_TX.cpu().numpy()))
        #print ('%d  %0.2f %s %s'%(frame_num, self.last_loss, self.last_cam.cpu().numpy(), self.last_TX.cpu().numpy()), file=self.log_target)
        cam_params = self.last_cam.cpu().numpy()
        #dbg_img[...,0] = (self.last_Z.cpu().numpy())*100
        return cam_params

def add_log_target(stream):
    instance.log_target = stream
    
instance = MotionDisc()
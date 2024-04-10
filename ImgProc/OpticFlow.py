import math
import time
from pubsub import pub
import numpy as np
import cv2
from sklearn.cluster import BisectingKMeans
from sklearn.mixture import BayesianGaussianMixture

from ImgProc import ImgEvents, Preprocessor
from ImgProc import Delta # dependency for generation

DEBUG=True

VIS_SIZE = 400
VIS_SCALE = VIS_SIZE/ 80. # 80 px movement is full span
KERN_SIZE = 15
CLUSTERS = 5
DOWNSCALE = 2
COL_MAP = np.array([(0,255,0),
            (0,0,255),
            (0,255,255),
            (255,255,0),
            (255,0,255),
            (255,0,0),
            ])
            
XFACTOR = 1
YFACTOR = 1
HFOV = 103./2
VFOV = 71./2
ASPECT = 1080./1920
HWARP_RATIO = 1.#/(ASPECT*np.tan(HFOV*(np.pi/180)))
VWARP_RATIO = HWARP_RATIO#/(np.tan(HFOV*(np.pi/180)))
class OpticFlow(Preprocessor.Preprocessor):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.last_fprev = np.array([])
        self.last_fprev_hash = 0
        
        
        self.STRENGTH = 1e6/(self.xdim*self.ydim*self.depth)
                
        XSTEP = 20
        YSTEP = 10
        self.y,self.x = np.mgrid[YSTEP/2:self.ydim:YSTEP, XSTEP/2:self.xdim:XSTEP]. \
                        reshape(2,-1).astype(int)
        self.xy_map = np.mgrid[0:self.ydim, 0:self.xdim].astype(np.float32)
        self.xy_map[0,:] -= self.midy
        self.xy_map[1,:] -= self.midx
        self.xy_map /= self.ydim # normalize to 0.5 = half height
        self.cos2_map = np.cos(self.xy_map*2*VFOV*(np.pi/180))
        self.cos2_map[1,:] = np.cos(self.xy_map[1,:]*2*ASPECT*HFOV*(np.pi/180))
        self.cos2_map **= 2
        self.sin_map = np.sin(self.xy_map*2*VFOV*(np.pi/180))
        self.sin_map[1,:] = np.sin(self.xy_map[1,:]*2*ASPECT*HFOV*(np.pi/180))
        fu = self.x.astype(np.float32) - self.midx
        fv = self.y.astype(np.float32) - self.midy
        mag, ang = cv2.cartToPolar(fu, fv)
        hsv = np.ones([self.ydim, self.xdim, 3], dtype=np.uint8)*255
        hsv[self.y, self.x, 0] = ang[...,0]*180/np.pi/2
        self.hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)/255.
        
        #self.regress = BisectingKMeans(n_clusters=CLUSTERS, bisecting_strategy='biggest_inertia')
        self.regress = BayesianGaussianMixture(n_components=CLUSTERS, n_init=3, 
                            init_params='k-means++', covariance_type='full', warm_start=False)
        
        
    def point_on_line(b, p):
        ap = p
        ab = b
        t = np.dot(ap, ab) / np.dot(ab, ab)
        # if you need the the closest point belonging to the segment
        t = max(0, min(1, t))
        result = t * ab
        return result
    def get_lastgray(self, img):
        if hash(img.data.tobytes()) == self.last_fprev_hash:
            return self.last_fprev
        else:
            return None
        
    def cache_lastgray(self, img, fnext):
        self.last_fprev = fnext
        self.last_fprev_hash = hash(img.data.tobytes())
        
    def vizflow(self, flow, flow_mag, str_mod=1):
        vis = np.zeros([VIS_SIZE,VIS_SIZE,3], dtype=np.float32)
        moving = flow_mag>0.25 # filter down to reduce work on non-movement
        fy,fx = self.y[moving], self.x[moving]
        #fy,fx = self.y, self.x
        if moving.sum()>100:
            flist = flow[fy,fx].reshape(-1,2)
            #labels = self.regress.fit_predict(flist)
            #KMN = CLUSTERS
            #count = [(labels==n).sum() for n in range(KMN)]
            #min_label = np.argmin(count)
            
            uvs = flow[fy,fx].T *VIS_SCALE
            locs = (np.clip(uvs, -VIS_SIZE/2, VIS_SIZE/2-1)+VIS_SIZE/2).astype(int)
            px = (locs[1],locs[0])
            
            # sum up flow heatmap to xy grid
            np.add.at(vis, px, self.STRENGTH*str_mod*self.hue[fy,fx]) # at allows repeated indices
            
            #sort_map = np.argsort(count)[::-1]
            #rev_map = np.arange(CLUSTERS)
            #rev_map[sort_map] = np.arange(CLUSTERS)
            #lcol = np.take(rev_map, np.array(labels))
            #hues = COL_MAP[lcol]/255.
            #np.add.at(vis, px, self.STRENGTH*hues) # at allows repeated indices
            vis = cv2.GaussianBlur(vis, (KERN_SIZE, KERN_SIZE), 1.) # smooth the heatmap
        
        # cluster flow to find representative motion
        if False and moving.sum()>100:
            flist = flow[fy,fx].reshape(-1,2)
            cent, cvar = get_avg_flow(flow[fy,fx])
            pos = (VIS_SIZE/2+cent*VIS_SCALE).astype(int)
            cv2.circle(vis, pos, 5, (0,0,255),1)
            
            km = self.regress
            labels = km.predict(flist)
            KMN = CLUSTERS
            count = [(labels==n).sum() for n in range(KMN)]
            n_labels = len(labels)
            #print ('c=%s'%(count))
            #print ('m=%s'%(km.means_))
            for ix, cent in enumerate(km.means_):
                v, w = np.linalg.eigh(km.covariances_[ix])
                #print ('v=%s w=%s'%(v,w))
                v = np.sqrt(2.0) * np.sqrt(v)
                u = w[0] / np.linalg.norm(w[0])
                up = u[::-1]
                up[0] *= -1
                
                pos = (VIS_SIZE/2+cent*VIS_SCALE).astype(int)
                kmcolor = (1,0,1)
                #cv2.circle(vis, pos, 5, kmcolor,1)
                pts = [u*v[0],
                        up*v[1],
                        -u*v[0],
                        -up*v[1],
                        u*v[0],]
                mag = (2.*count[ix])/n_labels
                kmcolor = (mag,0,mag)
                for pt1, pt2 in zip(pts[:],pts[1:]):
                    x1 = int(pos[0]+pt1[0]*VIS_SCALE)
                    y1 = int(pos[1]+pt1[1]*VIS_SCALE)
                    x2 = int(pos[0]+pt2[0]*VIS_SCALE)
                    y2 = int(pos[1]+pt2[1]*VIS_SCALE)
                    #cv2.line(vis, (x1,y1), (x2,y2), kmcolor, 1)
        return (np.clip(vis,0,1.)*255).astype(np.uint8)
        
    def vizgroup(self, viz, flow, flow_mag):
        XSTEP = 10
        YSTEP = 10
        fy,fx = np.mgrid[YSTEP/2:self.ydim:YSTEP, XSTEP/2:self.xdim:XSTEP].reshape(2,-1).astype(int)
        moving = flow_mag>0.75 # filter down to reduce work on non-movement
        #fy,fx = gy[moving], gx[moving]
        moving_conf = moving.sum()/(self.y.shape[0])
        if (moving_conf>=0.01):
            centroid = np.median(flow[fy,fx], axis=0)
            #print ('%0.4f, %s'%(moving_conf, centroid))
            
        if moving.sum()>100:
            flist = flow[fy,fx].reshape(-1,2)
            
            km = self.regress
            labels = km.predict(flist)
            #print (km.weights_)
            #print (km.means_)
            KMN = CLUSTERS
            count = [(labels==n).sum() for n in range(KMN)]
            sort_map = np.argsort(count)[::-1]
            rev_map = np.arange(CLUSTERS)
            rev_map[sort_map] = np.arange(CLUSTERS)
            lcol = np.take(rev_map, np.array(labels))
            for y,x,label in zip(fy,fx,lcol):
                viz[y:y+YSTEP,x:x+XSTEP] = COL_MAP[label]
        return viz
        
    def get_fit(self, flow, fy,fx):
        cos2 = self.cos2_map[:,fy,fx]
        sin = self.sin_map[:,fy,fx]
        xy = self.xy_map[:,fy,fx]
        dSxda = (HWARP_RATIO/cos2[1])
        dSxdb = (HWARP_RATIO*xy[1]*sin[0]/cos2[0])
        dSyda = (VWARP_RATIO*xy[0]*sin[1]/cos2[1])
        dSydb = (VWARP_RATIO/cos2[0])
        
        dSxdt = flow[fy,fx,0]
        dSydt = flow[fy,fx,1]
        
        all_da = np.stack((dSxda.flatten(),dSyda.flatten())).reshape(-1)
        all_db = np.stack((dSxdb.flatten(),dSydb.flatten())).reshape(-1)
        observations = np.stack((dSxdt.flatten(),dSydt.flatten()))
        mat_flowA = np.stack((all_da, all_db), axis=-1)
        fit = np.linalg.lstsq(mat_flowA, observations.reshape(-1), rcond=None)
        estX = fit[0]
        estB = np.matmul(mat_flowA, estX).reshape(2,-1)
        delta = observations-estB
        return fit, delta
    
        
    def get_cam_params(self, flow):
        XSTEP = 25
        YSTEP = 20
        fy,fx = np.mgrid[YSTEP/2:self.ydim:YSTEP, XSTEP/2:self.xdim:XSTEP].reshape(2,-1).astype(int)
        flow_mag = np.linalg.norm(flow[fy,fx], axis=-1)
        moving = flow_mag>0.25 # filter down to reduce work on non-movement
        if moving.sum()>1000:
            move_count = moving.sum()
            crop = moving
            for _ in range(3): # remove outliers and try fit again
                fy,fx = fy[crop], fx[crop]
                fit, delta = self.get_fit(flow, fy, fx)
                error = np.linalg.norm(delta, axis=0)
                if error.max() < 1. : # quit early if converged
                    break
                #print ('fit=%s err=%s'%(fit[0], error.max()))
                crop = error<np.median(error)
            
            return fit
        else:
            return np.linalg.lstsq([[0,0],[0,0]], [0,0], rcond=None)
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['base', 'last', 'abs_delta']):
            return False
        lframe = aux_imgs['last']
        frame = aux_imgs['base']
        abs_delta = aux_imgs['abs_delta'] # abs_delta to check moving
        
        # movement threshold
        moving = False
        VAL_THRES = 40
        COUNT_THRES = int(0.003*self.xdim*self.ydim*self.depth)
        if (abs_delta > VAL_THRES).sum() > COUNT_THRES:
            moving = True
            pub.sendMessage(ImgEvents.APPEND, key='moving', imgdata=[True])
        
        viz1 = np.zeros_like(frame)
        viz2 = np.zeros_like(frame)
        if True or moving: # only calculate optic flow if sufficient movement
            fnext = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE).astype(np.int16)
            fprev = self.get_lastgray(lframe)
            if fprev is None:
                fprev = cv2.resize(cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY),
                                    (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE).astype(np.int16)
            self.cache_lastgray(frame, fnext)
            flow_reduce = cv2.calcOpticalFlowFarneback(fprev, fnext, None, 0.5, 6, 35, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = cv2.resize(flow_reduce*DOWNSCALE, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            flow2 = np.zeros_like(flow)
            
            fit = self.get_cam_params(flow)
            cam_params = fit[0]
            cam_var = fit[1]
            #print ('%6.3f, %6.3f'%(cam_params[0],cam_params[1]))
            pub.sendMessage(ImgEvents.APPEND, key='pred_cam', imgdata=cam_params)
            
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #mag, ang = cv2.cartToPolar(self.warp_map[0], self.warp_map[1])
            hsv = np.zeros_like(frame)
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(np.sqrt(mag)*40,0,255)
            viz1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            YAW = cam_params[0]
            PITCH = cam_params[1]
            flow2[...,0] = YAW*HWARP_RATIO/self.cos2_map[1,:] +\
                        PITCH*HWARP_RATIO*self.xy_map[1,...]*self.sin_map[0,:]/self.cos2_map[0,:]
            flow2[...,1] = PITCH*VWARP_RATIO/self.cos2_map[0,:] +\
                        YAW*VWARP_RATIO*self.xy_map[0,...]*self.sin_map[1,:]/self.cos2_map[1,:]
            pub.sendMessage(ImgEvents.APPEND, key='flow2', imgdata=flow2)
            
            flow2[self.midy+200:,:] = 0
            mag, ang = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
            hsv = np.zeros_like(frame)
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(np.sqrt(mag)*40,0,255)
            viz2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        else:
            flow = np.zeros_like(frame)[...,:2]
            flow2 = np.zeros_like(flow)
            
    
        if DEBUG:
            # show flow arrows on debug
            YGRID = 30
            XGRID = 30
            y,x = np.mgrid[YGRID/2:self.ydim:YGRID, XGRID/2:self.xdim:XGRID].reshape(2,-1).astype(int)
            u,v = (x+flow[y,x,0]).round().astype(int), (y+flow[y,x,1]).round().astype(int)
            for x1,y1,x2,y2 in [(a,b,c,d) for a,b,c,d in zip (x,y, u,v) if a!=c or b!=d]:
                cv2.line(viz1, (x1,y1), (x2,y2), (0,255,0), 1)
                
            flow_mag = np.linalg.norm(flow[self.y,self.x], axis=-1)
            heatmap = self.vizflow(flow, flow_mag, str_mod=0.65)
            viz1[:VIS_SIZE,:VIS_SIZE,:] = heatmap
            
            u,v = (x+flow2[y,x,0]).round().astype(int), (y+flow2[y,x,1]).round().astype(int)
            for x1,y1,x2,y2 in [(a,b,c,d) for a,b,c,d in zip (x,y, u,v) if a!=c or b!=d]:
                cv2.line(viz2, (x1,y1), (x2,y2), (0,255,0), 1)
            flow_mag = np.linalg.norm(flow2[self.y,self.x], axis=-1)
            heatmap = self.vizflow(flow2, flow_mag, str_mod=0.65)
            viz2[:VIS_SIZE,:VIS_SIZE,:] = heatmap

            #self.vizgroup(viz2, flow, flow_mag)
            
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=viz1)
            pub.sendMessage(ImgEvents.APPEND, key='debug', imgdata=viz2)
        
        pub.sendMessage(ImgEvents.APPEND, key='flow', imgdata=flow)
        return True

def get_avg_flow(flow):
    flist = flow.reshape(-1,2)
    #cent = np.mean(flist,0) # clustering center is more accurate than mean
    
    regress = BisectingKMeans(n_clusters=CLUSTERS, bisecting_strategy='biggest_inertia')
    km = regress.fit(flist)
    #KMN = km.n_clusters_
    KMN = CLUSTERS
    count = [(km.labels_==n).sum() for n in range(KMN)]
    maxcount = max(count)
    modes = [i for i in range(KMN) if count[i]==maxcount]
    
    for ix in modes:
        cent = np.mean(flist[km.labels_==ix],0)
        vals = regress.transform(flist[km.labels_==ix])[:,ix]
        vari = 0
        if len(vals)>3:
            vari = np.var(vals)
        return cent, vari

_ = OpticFlow()
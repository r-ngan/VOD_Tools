import math
import time
import numpy as np
import cv2

from ImgProc import ImgTask
import VODEvents

# Simplify optical flow into 4 stages of motion:
# 1) Reaction time
# 2) Coarse aim
# 3) Fine aim / microadjust
# 4) Target confirmation
START_THRES = 2.4
FINE_THRES = 2.3
CONF_THRES = 1.1
class MoveStage(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
                
        # subsample to increase speed
        self.XSTEP = XSTEP = self.xdim//40
        self.YSTEP = YSTEP = self.ydim//40
        TRIM_Y1 = self.ydim//10 # 108px at 1080p
        TRIM_Y2 = self.ydim//10
        yx = np.mgrid[TRIM_Y1+YSTEP/2:self.ydim-TRIM_Y2:YSTEP, XSTEP/2:self.xdim:XSTEP]
        self.y,self.x = yx.reshape(2,-1).astype(int)
        
        self.ts = 0
        self.move_thres = int(0.4*self.y.shape[0]) # analyze if more than 40% moving
        self.flow_hist = [[0,0]]
        self.fsm = MoveFSM()
        self.data = {}
        self.latent_max = 0 # latent movement prior to reaction
        
        self.fsm.add_enter_callback(self.enter)
        self.fsm.add_exit_callback(self.exit)
        
    def get_flow_params(self, flow):
        flow_mag = np.linalg.norm(flow[self.y,self.x], axis=-1)
        moving = flow_mag>0.25 # filter down to reduce work on non-movement
        flow_mean = flow_mag.mean()
        flow_avg = np.zeros(2)
        if moving.sum()>self.move_thres:
            fy = self.y[moving]
            fx = self.x[moving]
            flow_avg = np.median(flow[fy,fx],axis=0)
            
        return flow_avg
            
    def requires(self):
        return [ImgTask.IMG_FLOW, ImgTask.VAL_TS, VODEvents.EVENT_NODE]
        
    def outputs(self):
        return ['move_hist', 'fsm_move']
        
    def proc_frame(self, flow, timestamp, eventdict):
        last_state = self.fsm.state
        
        events = self.unroll(eventdict)
        for ev in events:
            if ev['topic'] == VODEvents.BOT_APPEAR:
                self.fsm.transit(MoveFSM.BOT_APPEAR, ts=timestamp)
                self.latent_max = np.inf
            if ev['topic'] in [VODEvents.MOUSE_LMB_DOWN, VODEvents.BOT_NONE]:
                self.fsm.transit(MoveFSM.COOLDOWN, ts=timestamp)
            if ev['topic'] == VODEvents.BOT_NONE:
                self.fsm.transit(MoveFSM.IDLE, ts=timestamp)
                
        # analyze type of movement
        move = self.get_flow_params(flow)
        mag = np.linalg.norm(move)
        if (self.fsm.state==MoveFSM.BOT_APPEAR):
            self.latent_max = min(self.latent_max, mag)
            
        if (self.fsm.state==MoveFSM.BOT_APPEAR) and mag>START_THRES+self.latent_max:
            self.fsm.transit(MoveFSM.ADJ_COARSE, ts=timestamp)
        elif (self.fsm.state==MoveFSM.ADJ_COARSE) and mag<FINE_THRES:
            self.fsm.transit(MoveFSM.ADJ_FINE, ts=timestamp)
        elif (self.fsm.state==MoveFSM.ADJ_FINE) and mag<CONF_THRES:
            self.fsm.transit(MoveFSM.CONFIRM, ts=timestamp)
        elif (self.fsm.state==MoveFSM.CONFIRM) and mag>CONF_THRES:
            self.fsm.transit(MoveFSM.ADJ_FINE, ts=timestamp)
        
        if self.fsm.state not in [MoveFSM.IDLE, MoveFSM.COOLDOWN]: # don't store post firing
            self.flow_hist.append(move)
            if len(self.flow_hist)>80:
                self.flow_hist = self.flow_hist[-80:]
        
        state_map = [last_state, self.fsm.state] # state data to convey change / current state
        return np.array(self.flow_hist), state_map
        
    def enter(self, state, ts=None):
        #print ('enter state %s'%(state))
        if state == MoveFSM.BOT_APPEAR:
            self.data.clear()
            self.flow_hist.clear()
            self.data['react0'] = ts
        elif state == MoveFSM.ADJ_COARSE:
            self.data['coarse0'] = ts
        elif state == MoveFSM.ADJ_FINE:
            if 'fine0' not in self.data: # latch
                self.data['fine0'] = ts
        elif state == MoveFSM.CONFIRM:
            self.data['conf0'] = ts
        elif state == MoveFSM.COOLDOWN:
            pass

    def exit(self, state, ts=None): # called with old state
        if state == MoveFSM.BOT_APPEAR:
            self.data['react1'] = ts
        elif state == MoveFSM.ADJ_COARSE:
            self.data['coarse1'] = ts
        elif state == MoveFSM.ADJ_FINE:
            self.data['fine1'] = ts
        elif state == MoveFSM.CONFIRM:
            self.data['conf1'] = ts
    
# State diagram:
class MoveFSM():
    IDLE = 0
    BOT_APPEAR = 1
    ADJ_COARSE = 2
    ADJ_FINE = 3
    CONFIRM = 4
    COOLDOWN = 5
    
    def __init__(self):
        self.state = MoveFSM.IDLE
        self.enter = []
        self.exit = []
        
    def add_enter_callback(self, callable):
        self.enter.append(callable)
        
    def add_exit_callback(self, callable):
        self.exit.append(callable)
        
    def legal_trans(self, newstate):
        trans = [(MoveFSM.IDLE, MoveFSM.BOT_APPEAR),
                (MoveFSM.BOT_APPEAR, MoveFSM.ADJ_COARSE),
                (MoveFSM.ADJ_COARSE, MoveFSM.ADJ_FINE),
                (MoveFSM.ADJ_FINE, MoveFSM.CONFIRM),
                (MoveFSM.CONFIRM, MoveFSM.ADJ_FINE),
                (MoveFSM.CONFIRM, MoveFSM.COOLDOWN),
                (MoveFSM.BOT_APPEAR, MoveFSM.COOLDOWN), # disappear without user action
                (MoveFSM.ADJ_FINE, MoveFSM.COOLDOWN),
                (MoveFSM.ADJ_COARSE, MoveFSM.COOLDOWN),
                (MoveFSM.COOLDOWN, MoveFSM.IDLE), ]
        if (self.state, newstate) in trans:
            return True
        else:
            return False
        
    def transit(self, newstate, **kwargs):
        if self.legal_trans(newstate):
            for cb in self.exit:
                cb(self.state, **kwargs)
            self.state = newstate
            for cb in self.enter:
                cb(self.state, **kwargs)
            return True
        else:
            return False

instance = MoveStage()
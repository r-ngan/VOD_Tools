import math
import sys
import traceback
import json
from pubsub import pub
import numpy as np

import VODEvents
import Capture


class RangeStats(Capture.Capture):
    
    def __init__(self):
        super().__init__()
    
    def initialize(self, topic=pub.AUTO_TOPIC, **data):
        super().initialize(topic=topic, data=data)
        self.trial_count = 0
        
    def reset(self):
        super().reset()
        self.B1 = -1
        self.B2 = -1
        self.M1 = -1
        self.M2 = -1
        self.K1 = -1
        self.K2 = -1
        self.Bpos1 = (-999,-999)
        self.Bpos2 = (-999,-999)
        
    def event(self, timestamp, topic=pub.AUTO_TOPIC, **data):
        if not super().event(timestamp=timestamp, topic=topic, data=data):
            return False
        if (topic.getName() == Capture.BOT_END):
            self.B2 = timestamp
            print ('%s'%(vars(self)))
            self.stats_out()
            self.reset()
        elif (topic.getName() == Capture.BOT_START):
            self.B1 = timestamp
            self.Bpos1 = (data['x'],data['y'])
        elif (topic.getName() == Capture.MOUSE_START):
            self.M1 = timestamp
        elif (topic.getName() == Capture.MOUSE_END):
            self.M2 = timestamp
            self.Bpos2 = (data['x'],data['y'])
        elif (topic.getName() == Capture.KEY_START):
            self.K1 = timestamp
        elif (topic.getName() == Capture.KEY_END):
            self.K2 = timestamp
            
        return True
    
    def stats_out(self):
        STOP_TIME = 90 #ms (6 fr at 60fps)
        HEAD_RAD = 8 #px, TODO convert into degrees
        t_react = (min(self.M1,self.K1)-self.B1) * self.ms_frame_rate
        t_keyb = (self.K1-self.B1) * self.ms_frame_rate
        t_mouse = (self.M1-self.B1) * self.ms_frame_rate
        t_static = (abs(self.M2-self.K2)) * self.ms_frame_rate
        t_kill = (self.M2-self.B1) * self.ms_frame_rate
        d_bot = math.sqrt(self.Bpos2[0]**2+self.Bpos2[1]**2)
        
        t_stop = (self.M2 - self.K2) * self.ms_frame_rate
        kill_good = False
        if t_stop>=STOP_TIME and d_bot<=HEAD_RAD:
            kill_good = True
            
        self.trial_count += 1
        print (' react = %5.0fms'%(t_react))
        print (' keyb = %6.0fms'%(t_keyb))
        print (' mouse = %5.0fms'%(t_mouse))
        print (' static = %4.0fms'%(t_static))
        print (' kill = %6.0fms'%(t_kill))
        print (' offset = %4.2fpx'%(d_bot))
        print (' Good? = %s'%(kill_good))
        
        
_ = RangeStats()
import math
import sys
import traceback
import json
from pubsub import pub
import numpy as np

import VODEvents
import Capture

VERBOSE = False
class RangeStats(Capture.Capture):
    datastore = {}
    trial_count = 0
    formatting = {
        't_react' : {'desc':'Reaction time', 'unit':'ms', 'prec':'%5.1f', 'func':np.mean},
        't_keyb'  : {'desc':'Time to key  ', 'unit':'ms', 'prec':'%5.1f', 'func':np.mean},
        't_mouse' : {'desc':'Time to mouse', 'unit':'ms', 'prec':'%5.1f', 'func':np.mean},
        't_static': {'desc':'Time idling  ', 'unit':'ms', 'prec':'%5.1f', 'func':np.mean},
        't_shoot' : {'desc':'Time to shoot', 'unit':'ms', 'prec':'%5.1f', 'func':np.mean},
        'd_target': {'desc':'Head offset  ', 'unit':'px', 'prec':'%5.2f', 'func':np.mean},
        'hit'     : {'desc':'Hits         ', 'unit':'%','prec':'%5d', 'func':lambda x: np.mean(x)*100},}
    
    def initialize(self, **data):
        super().initialize(**data)
        
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
        if not super().event(timestamp=timestamp, topic=topic, **data):
            return False
        if (topic.getName() == Capture.BOT_END):
            if self.B2<0: # latch trigger
                self.B2 = timestamp
            self.stats_out()
            self.reset()
        elif (topic.getName() == Capture.BOT_START):
            if self.B1<0: # latch trigger
                self.reset() # clear all triggers
                self.B1 = timestamp
                self.Bpos1 = (data['x'],data['y'])
        elif (topic.getName() == Capture.MOUSE_START):
            if self.M1<0: # latch trigger
                self.M1 = timestamp
        elif (topic.getName() == Capture.MOUSE_END):
            if self.M2<0: # latch trigger
                self.M2 = timestamp
                self.Bpos2 = (data['x'],data['y'])
        elif (topic.getName() == Capture.KEY_START):
            if self.K1<0: # latch trigger
                self.K1 = timestamp
        elif (topic.getName() == Capture.KEY_END):
            if self.K2<0: # latch trigger
                self.K2 = timestamp
            
        return True
    
    def stats_out(self):
        STOP_TIME = 90 #ms (6 fr at 60fps)
        HEAD_RAD = 8 #px, TODO convert into degrees
        RangeStats.trial_count += 1
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
            
        add_store(RangeStats.datastore, 't_react', t_react)
        add_store(RangeStats.datastore, 't_keyb', t_keyb)
        add_store(RangeStats.datastore, 't_mouse', t_mouse)
        add_store(RangeStats.datastore, 't_static', t_static)
        add_store(RangeStats.datastore, 't_shoot', t_kill)
        add_store(RangeStats.datastore, 'd_target', d_bot)
        add_store(RangeStats.datastore, 'hit', kill_good)
        pub.sendMessage('LOG', text='%s'%(vars(self)) )
        print ('trial #%s done'%(RangeStats.trial_count))
        if VERBOSE:
            print ('%s'%(vars(self)))
            print (' react = %5.0fms'%(t_react))
            print (' keyb = %6.0fms'%(t_keyb))
            print (' mouse = %5.0fms'%(t_mouse))
            print (' static = %4.0fms'%(t_static))
            print (' shoot = %6.0fms'%(t_kill))
            print (' offset = %4.2fpx'%(d_bot))
            print (' Good? = %s'%(kill_good))
            
    def summarize():
        for key, values in RangeStats.datastore.items():
            meta = RangeStats.formatting[key]
            stat = meta['func'](values)
            fstring = '%s = ' + meta['prec'] +'%s'
            print(fstring%(meta['desc'], stat, meta['unit']))
            
        
def add_store(store, key, value):
    if not key in store:
        store[key] = []
    store[key].append(value)
_ = RangeStats()
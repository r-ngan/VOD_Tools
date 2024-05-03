import math
import sys
import traceback
import json
from pubsub import pub
import pandas as pd

import VODEvents
from ImgProc import ImgTask

VERBOSE = False
NODE = 'RangeStats'
class RangeStats(ImgTask.ImgTask):
    trial_count = 0
    formatting = {
        't_react' : {'desc':'Reaction time', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_keyb'  : {'desc':'Time to key  ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_mouse' : {'desc':'Time to mouse', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_static': {'desc':'Delta of kb/mouse', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_shoot' : {'desc':'Time to shoot', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        'd_target': {'desc':'Head offset  ', 'unit':'px', 'prec':'%5.2f', 'func':lambda x: x.mean()},
        'hit'     : {'desc':'Hits         ', 'unit':'%','prec':'%5d', 'func':lambda x: x.mean()*100},}
        
    def __init__ (self, **kwargs):
        super().__init__(**kwargs)
        self.df = pd.DataFrame([], columns=[
                        'B1','B2','M1','M2','K1','K2','Bpos1','Bpos2',])
        self.last_ix = 0
        self.log_target = [sys.stdout]
        
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.wait_for_bot = True
        
    def close(self):
        self.summarize()
        
    def store(self, column, value, row=None):
        if row is None:
            row = self.last_ix
        self.df.at[row, column] = value
        
    def is_empty(self, column, row=None):
        if row is None:
            row = self.last_ix
        if row not in self.df.index:
            return True
        return pd.isna(self.df.at[row, column])
        
    def start_new_row(self):
        #self.reset() # clear all triggers
        if self.last_ix in self.df.index: # start new index
            self.last_ix += 1
            
    def requires(self):
        # use last bothead since there are cases where input overlay shows recoil + LMB on same frame
        return [VODEvents.EVENT_NODE, ImgTask.VAL_TS, 'bothead_last']
        
    def outputs(self):
        return [NODE]

    # proc_frame signature must match requires / outputs
    def proc_frame(self, eventdict, ts, bothead):
        events = self.unroll(eventdict)
        for e in events:
            pub.sendMessage(e['topic'], timestamp=ts, bot=bothead, **e)
            self.event(timestamp=ts, bot=bothead, **e)
        return True
        
    def event(self, timestamp, bot, topic, **data):
        botxy = tuple(bot)
        if (topic == VODEvents.BOT_NONE):
            if self.is_empty('B2'): # latch trigger
                self.store('B2', timestamp)
            self.stats_out()
            self.wait_for_bot = True
        elif (topic == VODEvents.BOT_APPEAR):
            #if self.is_empty('B1'): # latch trigger
            if self.wait_for_bot:
                self.wait_for_bot = False
                self.start_new_row()
                self.store('B1', timestamp)
                self.store('Bpos1', botxy)
        elif (topic == VODEvents.MOUSE_MOVE_START):
            if self.is_empty('M1'): # latch trigger
                self.store('M1', timestamp)
        elif (topic in [VODEvents.MOUSE_LMB_DOWN]):
            if self.is_empty('M2'): # latch trigger
                self.store('M2', timestamp)
                self.store('Bpos2', botxy)
        elif (topic == VODEvents.KEY_ANY_DOWN):
            if self.is_empty('K1'): # latch trigger
                self.store('K1', timestamp)
        elif (topic == VODEvents.KEY_ANY_UP):
            if self.is_empty('K2'): # latch trigger
                self.store('K2', timestamp)
    
    def stats_out(self):
        pub.sendMessage('LOG', text='%s'%(self.df.loc[self.last_ix].to_dict()) )
        RangeStats.trial_count += 1
            
        print ('trial #%s done'%(RangeStats.trial_count))
            
    def summarize(self):
        STOP_TIME = 90 #ms (6 fr at 60fps)
        HEAD_RAD = 8 #px, TODO convert into degrees
        self.df['t_react'] = (self.df[['M1','K1']].min(axis=1) - self.df['B1'])
        self.df['t_keyb'] = (self.df['K1'] - self.df['B1'])
        self.df['t_mouse'] = (self.df['M1'] - self.df['B1'])
        self.df['t_static'] = (self.df['K1']-self.df['M1'])
        self.df['t_shoot'] = (self.df['M2'] - self.df['B1'])
        self.df['d_target'] = self.df['Bpos2'].apply(lambda x: math.sqrt(x[0]**2+x[1]**2) if pd.notna(x) else math.nan)
        self.df['t_stop'] = (self.df['M2'] - self.df['K2'])
        
        self.df['hit'] = (self.df['t_stop']>=STOP_TIME) & (self.df['d_target']<=self.df['Bpos2'].apply(lambda x: x[2]))
        
        for file in self.log_target:
            print(self.df.to_string(), file=file)
        for key in RangeStats.formatting.keys():
            values = self.df[key]
            meta = RangeStats.formatting[key]
            stat = meta['func'](values)
            fstring = '%s = ' + meta['prec'] +'%s'
            for file in self.log_target:
                print(fstring%(meta['desc'], stat, meta['unit']), file=file)
            
        
def add_log_target(stream):
    instance.log_target.append(stream)
    
instance = RangeStats()
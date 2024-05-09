import math
import sys
import traceback
import json
from pubsub import pub
import pandas as pd

import VODEvents
from ImgProc import ImgTask
from ImgProc.MoveStage import MoveFSM

VERBOSE = False
NODE = 'RangeStats'

#column enum
BOT_APPEAR = 'B1'
BOT_GONE = 'B2'
KB_START = 'K1'
KB_STOP = 'K2'
MOUSE_COARSE = 'M1'
MOUSE_FINE = 'M2'
MOUSE_CONFIRM = 'M3'
MOUSE_LMB = 'M4'
BOTPOS_START = 'Bpos1'
BOTPOS_OVERAIM = 'Bpos2'
BOTPOS_LMB = 'Bpos3'
class RangeStats(ImgTask.ImgTask):
    trial_count = 0
    formatting = {
        't_react' : {'desc':'Reaction time     ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_keyb'  : {'desc':'Reaction (kb)     ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_mouse' : {'desc':'Reaction (mouse)  ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_kbm'   : {'desc':'Delta of kb/mouse ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        
        't_flick' : {'desc':'Time to flick past', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        'd_over'  : {'desc':'Overaim offset    ', 'unit':'px', 'prec':'%5.2f', 'func':lambda x: x.mean()},
        
        't_step'  : {'desc':'KB move time      ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        't_conf'  : {'desc':'Confirm time      ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        
        't_shoot' : {'desc':'Time to shoot     ', 'unit':'ms', 'prec':'%5.1f', 'func':lambda x: x.mean()},
        'd_target': {'desc':'Head offset       ', 'unit':'px', 'prec':'%5.2f', 'func':lambda x: x.mean()},
        'hit'     : {'desc':'Hits              ', 'unit':'%','prec':'%5d', 'func':lambda x: x.mean()*100},}
        
    def __init__ (self, **kwargs):
        super().__init__(**kwargs)
        self.df = pd.DataFrame([], columns=[
                        BOT_APPEAR,BOT_GONE,
                        MOUSE_COARSE,MOUSE_FINE,MOUSE_CONFIRM,MOUSE_LMB,
                        KB_START,KB_STOP,
                        BOTPOS_START,BOTPOS_OVERAIM,BOTPOS_LMB,])
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
        return [VODEvents.EVENT_NODE, ImgTask.VAL_FRAMENUM, ImgTask.VAL_TS, 'bothead_last', 'fsm_move']
        
    def outputs(self):
        return [NODE]

    # proc_frame signature must match requires / outputs
    def proc_frame(self, eventdict, frame_num, ts, bothead, fsm_move):
        events = self.unroll(eventdict)
        for e in events:
            pub.sendMessage(e['topic'], timestamp=frame_num, bot=bothead, **e)
            self.event(timestamp=ts, bot=bothead, **e)
        self.capture_move(timestamp=ts, bot=bothead, fsm_move=fsm_move)
        return True
        
    def event(self, timestamp, bot, topic, **data):
        botxy = tuple(bot)
        if (topic == VODEvents.BOT_NONE):
            if self.is_empty(BOT_GONE): # latch trigger
                self.store(BOT_GONE, timestamp)
            self.stats_out()
            self.wait_for_bot = True
        elif (topic == VODEvents.BOT_APPEAR):
            #if self.is_empty(BOT_APPEAR): # latch trigger
            if self.wait_for_bot:
                self.wait_for_bot = False
                self.start_new_row()
                self.store(BOT_APPEAR, timestamp)
                self.store(BOTPOS_START, botxy)
        elif (topic == VODEvents.MOUSE_MOVE_START):
            if self.is_empty(MOUSE_COARSE): # latch trigger
                self.store(MOUSE_COARSE, timestamp)
        elif (topic in [VODEvents.MOUSE_LMB_DOWN]):
            if self.is_empty(MOUSE_LMB): # latch trigger
                self.store(MOUSE_LMB, timestamp)
                self.store(BOTPOS_LMB, botxy)
        elif (topic == VODEvents.KEY_ANY_DOWN):
            if self.is_empty(KB_START): # latch trigger
                self.store(KB_START, timestamp)
        elif (topic == VODEvents.KEY_ANY_UP):
            if self.is_empty(KB_STOP): # latch trigger
                self.store(KB_STOP, timestamp)
                
    def capture_move(self, timestamp, bot, fsm_move):
        botxy = tuple(bot)
        last, curr = fsm_move[:2]
        if curr==MoveFSM.ADJ_COARSE and last!=curr: # transition in
            if self.is_empty(MOUSE_COARSE): # latch trigger
                self.store(MOUSE_COARSE, timestamp)
        if curr==MoveFSM.ADJ_FINE and last!=curr: # transition in
            if self.is_empty(MOUSE_FINE): # latch trigger
                self.store(MOUSE_FINE, timestamp)
                self.store(BOTPOS_OVERAIM, botxy)
        if curr==MoveFSM.CONFIRM and last!=curr: # store start of target confirm. it may happen multiple times if adjusted
            self.store(MOUSE_CONFIRM, timestamp)
        if curr==MoveFSM.COOLDOWN and last!=curr and last!=MoveFSM.CONFIRM: # transition without confirm
            self.store(MOUSE_CONFIRM, timestamp)
    
    def stats_out(self):
        pub.sendMessage('LOG', text='%s'%(self.df.loc[self.last_ix].to_dict()) )
        RangeStats.trial_count += 1
            
        print ('trial #%s done'%(RangeStats.trial_count))
            
    def summarize(self):
        STOP_TIME = 90 #ms (6 fr at 60fps)
        BOT_TIME = 730 #ms (44 fr at 60fps medium bots)
        self.df['t_keyb'] = (self.df[KB_START] - self.df[BOT_APPEAR])
        self.df['t_mouse'] = (self.df[MOUSE_COARSE] - self.df[BOT_APPEAR])
        self.df['t_react'] = self.df[['t_keyb','t_mouse']].min(axis=1)
        self.df['t_kbm'] = (self.df[MOUSE_COARSE] - self.df[KB_START])
        
        self.df['t_flick'] = (self.df[MOUSE_FINE] - self.df[MOUSE_COARSE])
        self.df['d_over'] = self.df[BOTPOS_OVERAIM].apply(lambda x: math.sqrt(x[0]**2+x[1]**2) if pd.notna(x) else math.nan)
        
        
        self.df['t_step'] = (self.df[KB_STOP] - self.df[KB_START])
        self.df['t_conf'] = (self.df[MOUSE_LMB] - self.df[MOUSE_CONFIRM])
        
        self.df['t_shoot'] = (self.df[MOUSE_LMB] - self.df[BOT_APPEAR])
        self.df['d_target'] = self.df[BOTPOS_LMB].apply(lambda x: math.sqrt(x[0]**2+x[1]**2) if pd.notna(x) else math.nan)

        STOP_MOTION = (self.df[MOUSE_LMB] - self.df[KB_STOP]) # are you still moving when firing
        HEAD_RAD = self.df[BOTPOS_LMB].apply(lambda x: x[2] if pd.notna(x) else 0.)
        self.df['hit'] = (STOP_MOTION>=STOP_TIME) & \
                        (self.df['t_shoot']<=BOT_TIME) & \
                        (self.df['d_target']<=HEAD_RAD)
        
        pd.options.display.float_format = "{:.2f}".format
        printdf = self.df.to_string(formatters={BOTPOS_START: lambda x: '({:,.1f}, {:,.1f}, {:,.1f})'.format(*x),
                                BOTPOS_OVERAIM: lambda x: '({:,.1f}, {:,.1f}, {:,.1f})'.format(*x),
                                BOTPOS_LMB: lambda x: '({:,.1f}, {:,.1f}, {:,.1f})'.format(*x),
                                })
        for file in self.log_target:
            print(printdf, file=file)
        for key in RangeStats.formatting.keys():
            values = self.df[key]
            meta = RangeStats.formatting[key]
            stat = meta['func'](values)
            fstring = '%s = ' + meta['prec'] +' %s'
            for file in self.log_target:
                print(fstring%(meta['desc'], stat, meta['unit']), file=file)
            
        
def add_log_target(stream):
    instance.log_target.append(stream)
    
instance = RangeStats()
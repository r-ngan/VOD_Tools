import math
import time
import numpy as np

from ImgProc import ImgTask

class BestBot(ImgTask.ImgTask):
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.last_bot = [float('nan'),float('nan'),float('nan')]
            
    def requires(self):
        return ['botposes']
        
    def outputs(self):
        return ['bothead_closest', 'bothead_last']

    # proc_frame signature must match requires / outputs
    def proc_frame(self, botposes):
        res = [float('nan'),float('nan'),float('nan')]
        if (botposes is not None) and (len(botposes)>0):
            res = botposes[0]
        res_last = res if math.isnan(self.last_bot[0]) else self.last_bot
        self.last_bot = res
        return res, res_last

instance = BestBot()
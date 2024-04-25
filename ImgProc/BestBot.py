import math
import time
import numpy as np

from ImgProc import ImgTask

class BestBot(ImgTask.ImgTask):
            
    def requires(self):
        return ['botposes']
        
    def outputs(self):
        return ['bothead_closest']

    # proc_frame signature must match requires / outputs
    def proc_frame(self, botposes):
        res = [float('nan'),float('nan')]
        if (botposes is not None) and (len(botposes)>0):
            res = botposes[0]
        return res

instance = BestBot()
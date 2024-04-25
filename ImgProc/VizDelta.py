import math
import sys
import traceback
import json
import numpy as np
import cv2

from ImgProc import ImgTask

DEBUG=True # module is only used for debugging

class VizDelta(ImgTask.ImgTask):
            
    def requires(self):
        return [ImgTask.IMG_DELTA]
        
    def outputs(self):
        outs = [ImgTask.IMG_DEBUG]

    def proc_frame(self, frame):
        return frame

_ = VizDelta()
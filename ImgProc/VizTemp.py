import math
import sys
import traceback
import json
import numpy as np
import cv2

from ImgProc import ImgTask

DEBUG=True # module is only used for debugging

class VizTemp(ImgTask.ImgTask):
            
    def requires(self):
        return ['depth']
        
    def outputs(self):
        return [ImgTask.IMG_DEBUG]

    def proc_frame(self, frame):
        return frame

_ = VizTemp()
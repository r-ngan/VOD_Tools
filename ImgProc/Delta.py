import math
import sys
import traceback
import json
import numpy as np
import cv2

from ImgProc import ImgTask

class Delta(ImgTask.ImgTask):
            
    def requires(self):
        return [ImgTask.IMG_BASE, ImgTask.IMG_LAST]
        
    def outputs(self):
        return [ImgTask.IMG_DELTA, ImgTask.IMG_ABSD]

    def proc_frame(self, frame, lframe):
        delta = frame.astype(np.int16) - lframe # allow negative range, >128 delta
        abs_delta = (np.abs(delta)//2).astype(np.uint8)
        return delta, abs_delta

_ = Delta()
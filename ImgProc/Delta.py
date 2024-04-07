import math
import sys
import traceback
import json
from pubsub import pub
import numpy as np
import cv2

from ImgProc import ImgEvents, Preprocessor

class Delta(Preprocessor.Preprocessor):
    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, ['base', 'last']):
            return False
        frame = img
        lframe = aux_imgs['last']
        delta = frame.astype(np.int16) - lframe # allow negative range, >128 delta
        abs_delta = (np.abs(delta)//2).astype(np.uint8)
        pub.sendMessage(ImgEvents.APPEND, key='delta', imgdata=delta)
        pub.sendMessage(ImgEvents.APPEND, key='abs_delta', imgdata=abs_delta)
        return True

_ = Delta()
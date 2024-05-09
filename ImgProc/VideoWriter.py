import math
import sys
import traceback
import json
import numpy as np
import cv2

from ImgProc import ImgTask

WRITER_NODE = 'VODWRITE'

class VideoWriter(ImgTask.ImgTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = 'dump.mp4'

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        frate = kwargs[ImgTask._FR_KEY]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.sink = cv2.VideoWriter(self.path, fourcc, frate, (self.xdim, self.ydim))
        
    def close(self):
        if self.sink is not None:
            self.sink.release()
            
    def requires(self):
        return [ImgTask.VAL_FRAMENUM, ImgTask.VAL_TS, ImgTask.IMG_DEBUG]
        
    def outputs(self):
        return [WRITER_NODE]

    def proc_frame(self, frame_num, ts, frame):
        img = self.filter_source(frame)
        draw_text(img, '%d'%(frame_num), self.xdim-120,45)
        draw_text(img, '%d'%(ts), self.xdim-120,75)
        self.sink.write(img)
        return 0
        
    def filter_source(self, frame):
        if isinstance(frame, dict):
            frame = list(frame.values())[0]
        elif isinstance(frame, list):
            frame = frame[0]
        return np.array(frame)

def draw_text(frame, text, x, y):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x,y)
    fontScale              = 1
    fontColor              = (65535,65535,65535)
    thickness              = 1
    lineType               = 2

    cv2.putText(frame,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        
_ = VideoWriter()
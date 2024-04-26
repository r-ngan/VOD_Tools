import math
import sys
import traceback
import json
import numpy as np
import cv2

from ImgProc import ImgTask

class VideoSource(ImgTask.ImgTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cap = None
        self.frame_num = 0
        self.frame = None
        self.lframe = None
        self.pause = False

    def open(self, path): # video source has to be set up before Initialize
        self.cap = cv2.VideoCapture(path)
        if not self.cap_ok():
            print ('error opening %s'%(path))
            return None
        return self.cap
        
    def close(self):
        if self.cap is not None:
            self.cap.release()
        
    def cap_ok(self):
        if (self.cap is None) or (not self.cap.isOpened()):
            return False
        return True
            
    def skip_to_frame(self, frame_num):
        if not self.cap_ok():
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
        self.frame_num = frame_num-2 # jog will set it to -1
        return self.jog_frame()
            
    def jog_frame(self):
        if not self.cap_ok():
            return False
        self.lframe = self.frame
        ret, self.frame = self.cap.read()
        self.frame_num += 1
        return ret
        
    def get_video_params(self):
        if not self.cap_ok():
            return None
        xdim = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ydim = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frate = self.cap.get(cv2.CAP_PROP_FPS)
        frames_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return xdim, ydim, frate, frames_total
        
    def get_video_ts(self):
        if not self.cap_ok():
            return None
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)
            
    def requires(self):
        return None
        
    def outputs(self):
        return [ImgTask.VAL_FRAMENUM, ImgTask.VAL_TS,
                ImgTask.IMG_BASE, ImgTask.IMG_LAST]

    def proc_frame(self):
        if not self.pause:
            if not self.jog_frame(): # jog one frame
                raise VideoException()
        frame_num = self.frame_num
        timestamp = self.get_video_ts()
        frame = self.frame
        lframe = self.lframe
        return frame_num, timestamp, frame, lframe
        

class VideoException(Exception): # error reading frame, need to abort
    pass

instance = VideoSource()
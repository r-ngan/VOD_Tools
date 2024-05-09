import math
import sys
import traceback
import json
import numpy as np
import cv2

from ImgProc import ImgTask

# Abstract interface for all video sources
# do not instantiate directly
# only one implementation class should be instantiated and set VideoSource.instance
class VideoSource(ImgTask.ImgTask):

    def open(self, path): # video source has to be set up before Initialize
        pass
        
    def close(self):
        pass
        
    def is_running(self):
        pass
            
    def skip_to_frame(self, frame_num):
        return False # jogging not supported
            
    def jog_frame(self):
        return False # jogging not supported
        
    def get_video_params(self):
        frames_total = 0
        return self.xdim, self.ydim, 1000./self.ms_fr, frames_total
        
    def get_video_ts(self):
        return 0
        

class VideoException(Exception): # error reading frame, need to abort
    pass

instance = None
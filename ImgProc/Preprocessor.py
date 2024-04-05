from pubsub import pub
import ImgProc.ImgEvents as ImgEvents

_WIDTH_KEY = 'width'
_HEIGHT_KEY= 'height'
_DEPTH_KEY = 'depth'
_FR_KEY    = 'frame_rate'

class Preprocessor():
    def __init__(self, xdim=0, ydim=0, depth=0):
        self.frame_done = False
        self.xdim = 0
        self.ydim = 0
        self.midx = self.xdim//2
        self.midy = self.ydim//2
        self.depth = 0
        self.ms_fr = 1
        pub.subscribe(self.initialize, ImgEvents.INIT)
        pub.subscribe(self.proc_frame, ImgEvents.PREPROCESS)
        pub.subscribe(self.reset, ImgEvents.DONE)
        
    def initialize(self, topic=pub.AUTO_TOPIC, **data):
        self.frame_done = False
        if _WIDTH_KEY in data:
            self.xdim = data[_WIDTH_KEY]
            self.midx = data[_WIDTH_KEY]//2
        if _HEIGHT_KEY in data:
            self.ydim = data[_HEIGHT_KEY]
            self.midy = data[_HEIGHT_KEY]//2
        if _DEPTH_KEY in data:
            self.depth = data[_DEPTH_KEY]
        if _FR_KEY in data:
            self.ms_fr = 1000./data[_FR_KEY]
            
    def reset (self):
        if not self.frame_done:
            print ('Warning: missed processing on %s'%(self.__class__))
        self.frame_done = False
            
    def check_requirements(self, frame_db, keylist):
        if self.frame_done: # already processed this loop
            return False
        for x in keylist:
            if x not in frame_db: # missing requirement, stall
                pub.sendMessage(ImgEvents.DELAY, id=str(self.__class__))
                return False
        self.frame_done = True
        return True
        
    def proc_frame(self, timestamp, img, aux_imgs={}):
        if not self.check_requirements(aux_imgs, []):
            return
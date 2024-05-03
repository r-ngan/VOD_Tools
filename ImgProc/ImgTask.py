from pubsub import pub
import schedula as sh
import ImgProc.ImgEvents as ImgEvents

_WIDTH_KEY = 'width'
_HEIGHT_KEY= 'height'
_DEPTH_KEY = 'depth'
_FR_KEY    = 'frame_rate'

# common defines for node names
VAL_FRAMENUM = 'frame_num'
VAL_TS = 'timestamp'
IMG_BASE = 'base'
IMG_LAST = 'last'
IMG_DELTA = 'delta'
IMG_ABSD = 'abs_delta'
IMG_FLOW = 'flow'
IMG_DEBUG = 'debug'

CUDA = False
class ImgTask():
    def __init__(self, xdim=0, ydim=0, depth=0, **kwargs):
        self.xdim = 0
        self.ydim = 0
        self.midx = self.xdim//2
        self.midy = self.ydim//2
        self.depth = 0
        self.ms_fr = 1
        pub.subscribe(self.initialize, ImgEvents.INIT)
        pub.subscribe(self.close, ImgEvents.DONE)
        
    def initialize(self, pipe, topic=pub.AUTO_TOPIC, **data):
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
        pipe.register(self.proc_frame, name=str(self.__class__),
                    reqs=self.requires(), outs=self.outputs())
            
    def close(self):
        pass
        
    def requires(self):
        return [VAL_FRAMENUM, IMG_BASE]
        
    def outputs(self):
        return []
            
    # proc_frame signature must match requires / outputs
    def proc_frame(self, timestamp, img):
        return None
        
    def unroll(self, eventdict): # convert dict format into list
        res = []
        for k,v in eventdict.items():
            if v is not None:
                res.extend(v)
        return res
        
class ImgPipe():
    def __init__(self, name='ImgPipe'):
        self.dsp = sh.Dispatcher(name=name, raises=True)
        self.dsp.add_data(data_id=IMG_BASE)
        self.dsp.add_data(data_id=IMG_LAST)
        self.dsp.add_data(data_id=VAL_FRAMENUM)
        self.dsp.add_data(data_id=VAL_TS)
        
    def add_capture(self, topic):
        self.dsp.add_data(data_id=topic, wait_inputs=True, function=lambda x:x)
        
    def register(self, callable, reqs, outs, name=None):
        self.dsp.add_function(function=callable, inputs=reqs, outputs=outs, function_id=name)
        
    def run_pipe(self, ins, outs):
        try:
            sol = self.dsp.dispatch(inputs=ins, outputs=outs)#, executor='async')
            self.sol = sol
            res = sol.result() # resolve async 
        except sh.utils.exc.DispatcherError as e: # out of frames
            raise e.ex # flag true exception, not wrapper
        
        res = {k:sol[k] for k in outs if k in sol.keys()}
        return res


# make mock listeners to establish pubsub MDS
def initialize(pipe, width=0, height=0, depth=0, frame_rate=0):
    pass
pub.subscribe(initialize, ImgEvents.INIT)

pipe = ImgPipe()
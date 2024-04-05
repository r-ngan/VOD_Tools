from pubsub import pub

VOD_START = 'VSTART' # shared with ImgEvent.INIT
VOD_FRAME = 'VFRAME'
VOD_END = 'VEND'
BOT_APPEAR = 'BSTART'
BOT_NONE = 'BEND'
KEY_ANY_DOWN = 'KEYSTART'
KEY_ANY_UP = 'KEYEND'
MOUSE_MOVE_START = 'MMSTART'
MOUSE_MOVE_END = 'MMEND'
MOUSE_LMB_DOWN = 'MLMBSTART'
MOUSE_LMB_UP = 'MLMBEND'
MOUSE_RMB_DOWN = 'MRMBSTART'
MOUSE_RMB_UP = 'MRMBEND'


# make mock listeners to establish pubsub MDS
def initialize(width=0, height=0, depth=0, frame_rate=0):
    pass
pub.subscribe(initialize, VOD_START)
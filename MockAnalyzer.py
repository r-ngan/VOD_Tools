import math
import sys
import traceback
import json
from pubsub import pub

import VideoAnalysis
import VODEvents

class MockAnalyzer(VideoAnalysis.Analyzer):
    def proc_frame(self, timestamp, img, img_delta):
        super().proc_frame(timestamp=timestamp, img=img, img_delta=img_delta)
        for line in mock_json:
            if line['timestamp'] == timestamp:
                botx, boty = line['x']-self.midx, line['y']-self.midy
                pub.sendMessage(line['event'], timestamp=timestamp,x=botx,y=boty,)
_ = MockAnalyzer()

mock_json = [
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 193, 'x': 791, 'y': 539},
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 210, 'x': 791, 'y': 539},
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 211, 'x': 794, 'y': 541},
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 223, 'x': 953, 'y': 542} ,
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 237, 'x': 974, 'y': 533} ,
    {'event': VODEvents.BOT_NONE        , 'timestamp': 239, 'x': 0, 'y': 0} ,
    
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 283, 'x': 1136, 'y': 543},
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 298, 'x': 1133, 'y': 546},
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 299, 'x': 1128, 'y': 544},
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 311, 'x': 952, 'y': 539} ,
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 321, 'x': 962, 'y': 535} ,
    {'event': VODEvents.BOT_NONE        , 'timestamp': 324, 'x': 961, 'y': 585} ,
    
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 369, 'x': 907, 'y': 540} ,
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 386, 'x': 910, 'y': 547} ,
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 386, 'x': 911, 'y': 545} ,
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 398, 'x': 966, 'y': 542} ,
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 405, 'x': 955, 'y': 542} ,
    {'event': VODEvents.BOT_NONE        , 'timestamp': 408, 'x': 960, 'y': 541} ,
    
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 454, 'x': 926, 'y': 544},
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 469, 'x': 929, 'y': 545},
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 470, 'x': 933, 'y': 544},
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 481, 'x': 971, 'y': 539},
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 488, 'x': 958, 'y': 538},
    {'event': VODEvents.BOT_NONE        , 'timestamp': 491, 'x': 959, 'y': 540},
    
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 537, 'x': 950, 'y': 544},
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 552, 'x': 953, 'y': 542},
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 553, 'x': 958, 'y': 544},
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 564, 'x': 975, 'y': 541},
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 572, 'x': 960, 'y': 541},
    {'event': VODEvents.BOT_NONE        , 'timestamp': 575, 'x': 961, 'y': 541},
    
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 620, 'x': 692, 'y': 537} ,
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 636, 'x': 694, 'y': 542} ,
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 636, 'x': 695, 'y': 541} ,
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 648, 'x': 1004, 'y': 540},
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 661, 'x': 976, 'y': 541} ,
    {'event': VODEvents.BOT_NONE        , 'timestamp': 666, 'x': 961, 'y': 540} ,
    
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 710, 'x': 997, 'y': 539} ,
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 724, 'x': 996, 'y': 544} ,
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 725, 'x': 994, 'y': 542} ,
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 737, 'x': 937, 'y': 533} ,
    {'event': VODEvents.MOUSE_MOVE_END  , 'timestamp': 745, 'x': 960, 'y': 538} ,
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 747, 'x': 960, 'y': 538} ,
    {'event': VODEvents.BOT_NONE        , 'timestamp': 756, 'x': 959, 'y': 578} ,
    
    {'event': VODEvents.BOT_APPEAR      , 'timestamp': 795, 'x': 1010, 'y': 537},
    {'event': VODEvents.MOUSE_MOVE_START, 'timestamp': 810, 'x': 1006, 'y': 540},
    {'event': VODEvents.KEY_ANY_DOWN    , 'timestamp': 810, 'x': 1006, 'y': 540},
    {'event': VODEvents.KEY_ANY_UP      , 'timestamp': 824, 'x': 943, 'y': 538} ,
    {'event': VODEvents.MOUSE_LMB_DOWN  , 'timestamp': 832, 'x': 966, 'y': 538} ,
    {'event': VODEvents.BOT_NONE        , 'timestamp': 835, 'x': 960, 'y': 539} ,

]
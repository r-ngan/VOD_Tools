import math
import sys
import traceback
import json

from ImgProc import ImgTask
import VODEvents
import VODState

# use bot poses to mark trial start/end
class PoseAnalyzer(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.game_state = VODState.VOD_IDLE
            
    def requires(self):
        return ['frame_num', 'botposes']
        
    def outputs(self):
        return [VODEvents.EVENT_NODE]
        
    def proc_frame(self, timestamp, poses):
        events = []
        if self.game_state == VODState.VOD_BOT_ONSCREEN:
            if len(poses) < 1:
                self.game_state = VODState.VOD_IDLE
                events.append({'topic':VODEvents.BOT_NONE, 
                                'timestamp':timestamp})
            
        if self.game_state == VODState.VOD_IDLE: # full screen search
            if len(poses) > 0:
                bx,by = poses[0]
                self.game_state = VODState.VOD_BOT_ONSCREEN
                events.append({'topic':VODEvents.BOT_APPEAR,
                                'timestamp':timestamp,
                                'x':bx,'y':by,})
        return events
    
_ = PoseAnalyzer()
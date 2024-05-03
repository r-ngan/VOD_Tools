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
        return ['botposes']
        
    def outputs(self):
        return [VODEvents.EVENT_NODE]
        
    def proc_frame(self, poses):
        events = []
        if self.game_state == VODState.VOD_BOT_ONSCREEN:
            if len(poses) < 1:
                self.game_state = VODState.VOD_IDLE
                events.append({'topic':VODEvents.BOT_NONE})
            
        if self.game_state == VODState.VOD_IDLE: # full screen search
            if len(poses) > 0:
                bx,by = poses[0][:2]
                self.game_state = VODState.VOD_BOT_ONSCREEN
                events.append({'topic':VODEvents.BOT_APPEAR,
                                'x':bx,'y':by,})
        return events
    
_ = PoseAnalyzer()
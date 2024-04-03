from pubsub import pub

import VODEvents
import Capture

# a default router that sends game events to the statistics capture module

def proc_reset(timestamp, x=0, y=0, topic=pub.AUTO_TOPIC):
    pub.sendMessage(Capture.BOT_END, timestamp= timestamp)

def proc_game_event(timestamp, x, y, topic=pub.AUTO_TOPIC):
    capt_map = {VODEvents.BOT_APPEAR: Capture.BOT_START,
                VODEvents.KEY_ANY_DOWN: Capture.KEY_START,
                VODEvents.KEY_ANY_UP: Capture.KEY_END,
                VODEvents.MOUSE_MOVE_START: Capture.MOUSE_START,
                VODEvents.MOUSE_MOVE_END: Capture.MOUSE_END,
                VODEvents.MOUSE_LMB_DOWN: Capture.MOUSE_END,
                }
    pub.sendMessage(capt_map[topic.getName()],
                timestamp= timestamp,
                x= x,
                y= y,)

pub.subscribe(proc_reset, VODEvents.BOT_NONE)
pub.subscribe(proc_game_event, VODEvents.BOT_APPEAR)
pub.subscribe(proc_game_event, VODEvents.KEY_ANY_DOWN)
pub.subscribe(proc_game_event, VODEvents.KEY_ANY_UP)
pub.subscribe(proc_game_event, VODEvents.MOUSE_MOVE_START)
pub.subscribe(proc_game_event, VODEvents.MOUSE_MOVE_END)
pub.subscribe(proc_game_event, VODEvents.MOUSE_LMB_DOWN)
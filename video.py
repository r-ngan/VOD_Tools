import math
import argparse
import sys
import time
import traceback
import json
from pubsub import pub
import schedula as sh
import numpy as np
import cv2

import VODEvents
import Capture
from ImgProc import ImgTask, ImgEvents

#import automatically activates the module
import ImgProc.Delta
import ImgProc.OpticFlow
import ImgProc.BotFind
import ImgProc.BotTrack
import ImgProc.BestBot
import ImgProc.MotionAnalyzer
#import ImgProc.VizFlow
#import ImgProc.VizMotion
#import ImgProc.VizPredict
import ImgProc.VizBots
import ImgProc.MouseTrack
import InputAnalyzer
import PoseAnalyzer
import RangeStats
from ImgProc import VideoSource

BATCH = False # batch mode don't print on screen

mousex = 0
mousey = 0
mouse_text = ''
frame_num = 0
frame_data = np.array([])
logstream = sys.stdout

def get_mouse(event, x, y, flags, param):
    global mouse_text, frame_data
    if event == cv2.EVENT_LBUTTONDOWN:
        mousex = x
        mousey = y
        value = frame_data[y,x,:]
        pub.sendMessage(Capture.MOUSE_START,
                    timestamp= value,
                    x= mousex,
                    y= mousey,)
    #mouse_text = '(%d,%d)'%(x, y)
    #flow = frame_db['flow'][y,x]
    #flow2 = frame_db['flow2'][y,x]
    #value = '%7.3f %7.3f %7.3f\n%7.3f %7.3f %7.3f'%\
    #        (np.linalg.norm(flow), flow[0], flow[1],
    #        np.linalg.norm(flow2), flow2[0], flow2[1])
    value = '%s'%(frame_data[y,x])
    mouse_text = '%s'%(value)

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

def dbg_event(topic=pub.AUTO_TOPIC, **kwargs):
    global logstream
    if (not topic.getName().startswith('capture.')):
        print ('event %s / %s'%(topic.getName(), kwargs), file=logstream)
    
def unroll(eventdict): # convert dict format into list
    res = []
    for k,v in eventdict.items():
        if v is not None:
            res.append(v)
    return res
    
def fill_missing(eventdict, keys):
    for k in keys: # pad defaults for event and debug list
        if k not in eventdict.keys():
            eventdict[k]={}
    return eventdict
    
def print_runtime():
    total=0
    for k,v in ImgTask.pipe.dsp.solution.workflow.nodes.items():
        if 'duration' in v:
            dur = v['duration']*1000
            total += dur
            print ('  %s : %5.2fms'%(k, dur))
    print ('seq time= %s'%(total))

def main(args):
    global logstream, frame_data, frame_num, frame_wait, waiters
    
    argp = argparse.ArgumentParser(description='VOD review tool')
    argp.add_argument('source', nargs='?', default='test.mkv', 
                    help='video source path')
    argp.add_argument('--skip', type=int, nargs='?', default=180,
                    help='start at offset frames into video')
    argp.add_argument('-m', '--manual', action='store_false',
                    help='turn off autoplay')
    argp.add_argument('-d', '--dump', nargs='?', const='dump.mp4', default=None,
                    help='dump out debug frame to video file')
    params = argp.parse_args(args)
    
    src = VideoSource.instance
    if not src.open(params.source):
        print ('FATAL ERROR: cannot open video')
        return
        
    DUMP = False
    if params.dump is not None:
        DUMP = True
        from ImgProc import VideoWriter
        VideoWriter.path = params.dump
        print('Video writer imported')
        
    src.skip_to_frame(params.skip) # skip VOD preamble
    # 23-09-15 tricky bot appear: 1440, 2590
    # 23-12-24 tricky bot appear: 811, 1433
    xdim, ydim, frate, frames_total = src.get_video_params()
    depth = src.frame.shape[-1]
    print ('Video = %s @ %s fps. %s frames'%(src.frame.shape, frate, frames_total))
    
    show_img = 0
    autoplay = params.manual
    if not BATCH:
        def breakpoint(timestamp=0, x=0, y=0):
            nonlocal autoplay
            autoplay = False
        #pub.subscribe(breakpoint, VODEvents.BOT_APPEAR) # pause at certain events
        #pub.subscribe(breakpoint, VODEvents.KEY_ANY_DOWN) # pause at certain events
    else:
        autoplay = True
    
    with open('zlog.txt', 'w', buffering=1) as logfile: # line buffered
        logstream = logfile
        pub.subscribe(dbg_event, pub.ALL_TOPICS)
        pub.sendMessage(ImgEvents.INIT, 
                        pipe= ImgTask.pipe,
                        width= xdim,
                        height= ydim,
                        depth= depth,
                        frame_rate= frate,) # initialize all modules
        RangeStats.add_log_target(logstream)
        # configure reduce nodes
        ImgTask.pipe.add_capture(ImgTask.IMG_DEBUG)
        ImgTask.pipe.add_capture(VODEvents.EVENT_NODE)
        
        if not BATCH:
            cv2.namedWindow('VODTool')
            cv2.setMouseCallback('VODTool', get_mouse)
        
        vidst = time.time_ns()
        while(src.cap_ok()):
            frame_num = src.frame_num
            if (frame_num%frate)==0:
                perc = frame_num / frames_total * 100.
                print ('%5.1f%% done. timestamp= %4.1f s'%(perc, src.get_video_ts()/1000))
                print_runtime()
            
            tst = time.time_ns()
            outs = [RangeStats.NODE]
            outs.append(ImgTask.IMG_DEBUG)
            if DUMP:
                outs.append(VideoWriter.WRITER_NODE)
            try:
                sol = ImgTask.pipe.run_pipe(ins=None, outs=outs)
            except VideoSource.VideoException: # out of frames
                break
            ten = time.time_ns()
            #print ('pipe= %3.3fms'%((ten-tst)/1e6))
            
            if BATCH:
                continue # skip user interface
            
                
            sol = fill_missing(sol, outs)
            frame_dbg = unroll(sol[ImgTask.IMG_DEBUG])
            while (1): # show frame and wait for user input
                if show_img>0:
                    output = np.array(frame_dbg[show_img-1])
                else:
                    output = np.array(src.frame)
                frame_data = output
                    
                #MoveAnalyzer.instance.draw_hist(output)
                draw_text(output, mouse_text, 50,50)
                
                draw_text(output, '%d'%(frame_num), 1800,50)
                cv2.imshow('VODTool',output)
                key = cv2.pollKey()
                
                STEP_SIZE = 0.1
                if key==-1:
                    if autoplay:
                        break
                    else:
                        time.sleep(0.060)
                        continue
                if key==ord('r'): # flip display
                    show_img = (show_img-1)%(len(frame_dbg)+1)
                    #print ('showing %s'%(show_img))
                    continue
                if key==ord('f'): # flip display
                    show_img = (show_img+1)%(len(frame_dbg)+1)
                    #print ('showing %s'%(show_img))
                    continue
                if key==ord('k'): # play/pause
                    autoplay = not autoplay
                    continue
                    
                '''
                PAUSE = True
                if key==ord('w'): # replay frame
                    FlowAnalyzer.YFACTOR += STEP_SIZE
                elif key==ord('s'): # replay frame
                    FlowAnalyzer.YFACTOR -= STEP_SIZE
                elif key==ord('a'): # replay frame
                    FlowAnalyzer.XFACTOR -= STEP_SIZE
                elif key==ord('d'): # replay frame
                    FlowAnalyzer.XFACTOR += STEP_SIZE
                elif key==ord('z'): # replay frame
                    FlowAnalyzer.ZFACTOR -= STEP_SIZE
                elif key==ord('x'): # replay frame
                    FlowAnalyzer.ZFACTOR += STEP_SIZE
                else:
                    PAUSE = False
                if PAUSE:
                    frame_num -= 1
                    print ('factor=%s, %s, %s'%(FlowAnalyzer.XFACTOR,
                                            FlowAnalyzer.YFACTOR,
                                            FlowAnalyzer.ZFACTOR))
                    break #'''
                break # any other key is go next frame
            
            if key & 0xFF == ord('q'): # exit condition
                break
                
        viden = time.time_ns()
        avg_time = (viden-vidst)/(frame_num-params.skip)
        print ('time elapsed = %3.3fs'%((viden-vidst)/1e9))
        print ('avg per frame = %3.3fms'%(avg_time/1e6))
        pub.sendMessage(ImgEvents.DONE) # clean up modules
        
    #print ('total trials = %d'%(RangeStats.trial_count))
    
    cv2.destroyAllWindows()

    print ('ok finished')
    


'''
==Framework==

event hooks (bgr_frame, frame_delta, timestamp, event_type)
event_type to reset trigger
on reset, hit analysis hook (how to pass variables from events to analysis?)
tally up statistics across trials

iterate through frames:
    detect bot first appear, mark timestamp of first appearance (event) A1, Bpos1
    detect first keyboard action, mark timestamp K1
    detect first mouse move, mark timestamp M1
    detect end of keyboard move, mark timestamp, bot location K2, Bpos2
    detect mouse fire event, mark timestamp, bot location M2, Bpos3
    detect no bots on screen (reset all triggers)

per trial, calculate:
    reaction time = min(M1,K1) - A1
    keyboard react = K1 - A1
    mouse react = M1 - A1
    static time = M2 - K2
    time to kill = M2 - A1
    
    bot difficulty = Bpos1
    micro-adjust error = Bpos2
    micro-adjust improvement = Bpos3 - Bpos2
    true error = Bpos3

'''


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
import math
import sys
import time
import traceback
import json
from pubsub import pub
import numpy as np
import cv2

import VODEvents
import Capture
import VideoAnalysis
from ImgProc import ImgEvents

#import automatically activates the module
#import ImgProc.Delta
from ImgProc import OpticFlow
#import ImgProc.BotPose
import InputAnalyzer
import MoveAnalyzer
import PoseAnalyzer
import DefaultEvRouter
from RangeStats import RangeStats

mousex = 0
mousey = 0
mouse_text = ''
frame_num = 0
frame_data = np.array([])
frame_db = {}
frame_wait = False
waiters = []
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
    flow = frame_db['flow'][y,x]
    flow2 = frame_db['flow2'][y,x]
    value = '%7.3f %7.3f %7.3f\n%7.3f %7.3f %7.3f'%\
            (np.linalg.norm(flow), flow[0], flow[1],
            np.linalg.norm(flow2), flow2[0], flow2[1])
    mouse_text = '%s'%(value)

def draw_text(frame, text, x, y):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x,y)
    fontScale              = 1
    fontColor              = (255,255,255)
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
    if (not topic.getName() == VODEvents.VOD_FRAME) and \
        (not topic.getName() == ImgEvents.PREPROCESS) and \
        (not topic.getName() == ImgEvents.APPEND) and \
        (not topic.getName() == ImgEvents.DONE) and \
        (not topic.getName().startswith('capture.')):
        print ('event %s / %s'%(topic.getName(), kwargs), file=logstream)

def frame_append(key, imgdata, topic=pub.AUTO_TOPIC, **kwargs):
    global frame_db
    if key=='debug':
        frame_db[key].append(imgdata)
    else:
        if key in frame_db:
            print ('!warning: %s already in frame db'%(key))
        frame_db[key] = imgdata
    
def frame_delay(id, topic=pub.AUTO_TOPIC, **kwargs): # at least one module requesting wait
    global frame_wait, waiters
    frame_wait = True
    waiters.append(id)

def main(args):
    global logstream, frame_data, frame_num, frame_db, frame_wait, waiters
    cap = cv2.VideoCapture('move_test3.mp4')
                            #'test2023-09-15.mkv')
                            #'look_test.mp4')
    if not cap.isOpened():
        print ('error opening')
        return
    
    SKIP_FRAMES = 480 # skip VOD preamble
    cap.set(cv2.CAP_PROP_POS_FRAMES, SKIP_FRAMES-1)
    frame_num = SKIP_FRAMES-1
    ret, frame = cap.read()
    ydim, xdim, depth = frame.shape
    midx = xdim//2
    midy = ydim//2
    frate = cap.get(cv2.CAP_PROP_FPS)
    frames_total = int(cap.get( cv2.CAP_PROP_FRAME_COUNT))
    print ('Video = %s @ %s fps. %s frames'%(frame.shape,frate, frames_total))
    
    
    show_img = 0
    autoplay = True
    def breakpoint(timestamp=0, x=0, y=0):
        nonlocal autoplay
        autoplay = False
    #pub.subscribe(breakpoint, VODEvents.BOT_APPEAR) # pause at certain events
    #pub.subscribe(breakpoint, VODEvents.KEY_ANY_DOWN) # pause at certain events
    
    path_hist = []
    def del_path(timestamp=0, x=0, y=0):
        nonlocal path_hist
        path_hist.clear()
    pub.subscribe(del_path, VODEvents.BOT_APPEAR) # clear path at event
    
    with open('zlog.txt', 'w', buffering=1) as logfile: # line buffered
        logstream = logfile
        pub.subscribe(dbg_event, pub.ALL_TOPICS)
        pub.subscribe(frame_append, ImgEvents.APPEND)
        pub.subscribe(frame_delay, ImgEvents.DELAY)
        pub.sendMessage(VODEvents.VOD_START, 
                        width= xdim,
                        height= ydim,
                        depth= depth,
                        frame_rate= frate,) # initialize all modules
        
        cv2.namedWindow('VODTool')
        cv2.setMouseCallback('VODTool', get_mouse)
        PAUSE = False
        while(cap.isOpened()):
            if not PAUSE:
                lframe = frame
                ret, frame = cap.read()
            else:
                PAUSE = False
            if not ret: # out of frames, VOD done
                break
            frame_num += 1
            if frame_num < SKIP_FRAMES:
                continue
                
            # reset frame db
            frame_db.clear()
            frame_db['base'] = frame
            frame_db['last'] = lframe
            frame_db['debug'] = []
            
            frame_wait = True
            while (frame_wait): # preprocessors can request wait if out of order due to pubsub, forms a DAG of processing
                frame_wait = False
                waiters.clear()
                db_size = len(frame_db)
                pub.sendMessage(ImgEvents.PREPROCESS,
                            timestamp= frame_num,
                            img= frame,
                            aux_imgs= frame_db,)
                if frame_wait: # wait requested
                    curr_db_size = len(frame_db)
                    if curr_db_size<=db_size: # no changes since last cycle, stuck
                        print ('preprocess done, some modules failed: %s'%(waiters))
                        break
            pub.sendMessage(ImgEvents.DONE) # preprocessing is done, clean up
            
            # all preprocessors done, go to analysis
            pub.sendMessage(VODEvents.VOD_FRAME,
                        timestamp= frame_num,
                        img= frame,
                        aux_imgs= frame_db,)
            # continue # skip user interface
            
            if 'pred_cam' in frame_db:
                cam_params = frame_db['pred_cam']
                path_hist.append(cam_params)
            frame_data = frame
            ''' turn flow into prediction image
            uvmap = np.indices((ydim, xdim)).astype(np.int16)
            uvmap[1] -= flint[...,0]
            uvmap[1,uvmap[1,...]<0] = 0
            uvmap[1,uvmap[1,...]>=xdim] = 0
            uvmap[0] -= flint[...,1]
            uvmap[0,uvmap[0,...]<0] = 0
            uvmap[0,uvmap[0,...]>=ydim] = 0
            frame16 = fprev.astype(np.int16)
            for y in range(0,ydim):
                for x in range(0,xdim):
                    v,u = uvmap[0, y,x], uvmap[1, y,x]
                    dout[y,x] = frame16[v,u]#-lframe[y,x] #'''
                
            while (1): # show frame and wait for user input
                if show_img>0:
                    output = np.array(frame_db['debug'][show_img-1])
                else:
                    output = np.array(frame)
                    
                loc = np.array([midx,midy])
                for xy in path_hist[::-1]:
                    mag = np.linalg.norm(xy)
                    MAG_LIM = 20
                    mag = min(MAG_LIM,mag) # cap to limit
                    color = [0,128*mag/MAG_LIM,200*mag/MAG_LIM]
                    color = tuple(int(x) for x in color)
                    newloc = loc+xy
                    x1 = int(loc[0])
                    y1 = int(loc[1])
                    x2 = int(newloc[0])
                    y2 = int(newloc[1])
                    cv2.line(output, (x1,y1), (x2,y2), color, int(1+2.5*mag/MAG_LIM))
                    loc = newloc
                #cv2.rectangle(output,(midx-1,midy-1),(midx+1,midy+1),(255,255,255),1)
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
                    show_img = (show_img-1)%(len(frame_db['debug'])+1)
                    #print ('showing %s'%(show_img))
                    continue
                if key==ord('f'): # flip display
                    show_img = (show_img+1)%(len(frame_db['debug'])+1)
                    #print ('showing %s'%(show_img))
                    continue
                if key==ord('k'): # play/pause
                    autoplay = not autoplay
                    continue
                    
                PAUSE = True
                if key==ord('w'): # replay frame
                    OpticFlow.YFACTOR += STEP_SIZE
                elif key==ord('s'): # replay frame
                    OpticFlow.YFACTOR -= STEP_SIZE
                elif key==ord('a'): # replay frame
                    OpticFlow.XFACTOR -= STEP_SIZE
                elif key==ord('d'): # replay frame
                    OpticFlow.XFACTOR += STEP_SIZE
                else:
                    PAUSE = False
                if PAUSE:
                    frame_num -= 1
                    print ('factor=%s, %s'%(OpticFlow.XFACTOR,OpticFlow.YFACTOR))
                    break
                break # any other key is go next frame
            
            if key & 0xFF == ord('q'): # exit condition
                break
        print ('%s'%(RangeStats.datastore), file=logstream)
    print ('total trials = %d'%(RangeStats.trial_count))
    RangeStats.summarize()
    #with open('zz2.csv', 'w') as dumpfile:
    #    for x in InputAnalyzer.key_data:
    #        dumpfile.write('%s\n'%(x))
    cap.release()
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
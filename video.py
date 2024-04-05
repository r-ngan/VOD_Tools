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
#import ImgProc.OpticFlow
import InputAnalyzer
import DefaultEvRouter
import RangeStats
import PoseAnalyzer

mousex = 0
mousey = 0
mouse_text = ''
frame_num = 0
frame_data = np.array([])
frame_db = {}
frame_wait = False
waiters = []
    
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
    value = frame_data[y,x,:]
    mouse_text = '(%s)'%(value)

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
    if (not topic.getName() == VODEvents.VOD_FRAME) and \
        (not topic.getName() == ImgEvents.PREPROCESS) and \
        (not topic.getName() == ImgEvents.APPEND) and \
        (not topic.getName() == ImgEvents.DONE):
        print ('event %s / %s'%(topic.getName(), kwargs))

def frame_append(key, imgdata, topic=pub.AUTO_TOPIC, **kwargs):
    global frame_db
    if key in frame_db:
        print ('!warning: %s already in frame db'%(key))
    frame_db[key] = imgdata
    
def frame_delay(id, topic=pub.AUTO_TOPIC, **kwargs): # at least one module requesting wait
    global frame_wait, waiters
    frame_wait = True
    waiters.append(id)

def main(args):
    global frame_data, frame_num, frame_db, frame_wait, waiters
    cap = cv2.VideoCapture(#'game.mp4')
                            'test.mkv')
    if not cap.isOpened():
        print ('error opening')
        return
        
    #pub.subscribe(cb2, VODEvents.BOT_APPEAR)
    #pub.sendMessage(Capture.BOT_START, data={'test':'test1'})
    
    SKIP_FRAMES = 205 # skip VOD preamble
    cap.set(cv2.CAP_PROP_POS_FRAMES, SKIP_FRAMES-1)
    frame_num = SKIP_FRAMES
    ret, frame = cap.read()
    ydim, xdim, depth = frame.shape
    midx = xdim//2
    midy = ydim//2
    frate = cap.get(cv2.CAP_PROP_FPS)
    frames_total = int(cap.get( cv2.CAP_PROP_FRAME_COUNT))
    print ('Video = %s @ %s fps. %s frames'%(frame.shape,frate, frames_total))
    pub.subscribe(dbg_event, pub.ALL_TOPICS)
    pub.subscribe(frame_append, ImgEvents.APPEND)
    pub.subscribe(frame_delay, ImgEvents.DELAY)
    pub.sendMessage(VODEvents.VOD_START, 
                    width= xdim,
                    height= ydim,
                    depth= depth,
                    frame_rate= frate,) # initialize all modules
    
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', get_mouse)
    show_delta = True
    autoplay = False
    while(cap.isOpened()):
        lframe = frame
        ret, frame = cap.read()
        if not ret: # out of frames, VOD done
            break
        frame_num += 1
        if frame_num < SKIP_FRAMES:
            continue
            
        # reset frame db
        frame_db.clear()
        frame_db['base'] = frame
        frame_db['last'] = lframe
        frame_db['debug'] = frame
        
        frame_wait = True
        while (frame_wait):
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
                # it is a legit wait
        pub.sendMessage(ImgEvents.DONE) # preprocessing is done, clean up
        
        # all preprocessors done, go to analysis
        pub.sendMessage(VODEvents.VOD_FRAME,
                    timestamp= frame_num,
                    img= frame,
                    aux_imgs= frame_db,)
        # continue # skip user interface
        
        frame_data = frame
        if 'debug' in frame_db:
            dout = frame_db['debug']
        else:
            dout = VideoAnalysis.dbg_frame
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
            
        ''' show flow arrows on debug
        for y in range(0,ydim,30):
            for x in range(0,xdim,30):
                u,v = x+flint[y,x,0], y+flint[y,x,1]
                cv2.line(dout, (x,y), ((x+u)//2,(y+v)//2), (0,255,0), 1)
                cv2.line(dout, ((x+u)//2,(y+v)//2), (u,v), (0,0,255), 1) #'''
        while (1): # show frame and wait for user input
            if show_delta:
                output = np.array(dout)
                #output = np.array(np.abs(delta)*100)
                #output = np.array(VideoAnalysis.dbg_frame)
            else:
                output = np.array(frame)
            cv2.rectangle(output,(midx-1,midy-1),(midx+1,midy+1),(255,255,255),1)
            draw_text(output, mouse_text, 50,50)
            
            draw_text(output, '%d'%(frame_num), 1800,50)
            cv2.imshow('frame',output)
            key = cv2.pollKey()
            
            if key==-1:
                if autoplay:
                    break
                else:
                    time.sleep(0.060)
                    continue
            if key==ord('f'): # flip display
                show_delta = not show_delta
                continue
            if key==ord('k'): # play/pause
                autoplay = not autoplay
                continue
            break # any other key is go next frame
        
        if key & 0xFF == ord('q'): # exit condition
            break

    with open('zz2.csv', 'w') as dumpfile:
        for x in InputAnalyzer.key_data:
            dumpfile.write('%s\n'%(x))
    cap.release()
    cv2.destroyAllWindows()

    print ("ok finished")
    


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
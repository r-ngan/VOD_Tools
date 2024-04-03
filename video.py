import math
import sys
import traceback
import json
from pubsub import pub
import numpy as np
import cv2

import VODEvents
import VODState
import Capture
import VideoAnalysis

#import automatically activates the module
import InputAnalyzer
import DefaultEvRouter
import RangeStats

mousex = 0
mousey = 0
mouse_text = ''
frame_num = 0
frame_data = np.array([])
    
def get_mouse(event, x, y, flags, param):
    global mouse_text, frame_data
    if event == cv2.EVENT_LBUTTONDOWN:
        mousex = x
        mousey = y
        value = frame_data[y,x,:]#VideoAnalysis.dbg_frame[y,x]
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
    if not topic.getName() == VODEvents.VOD_FRAME:
        print ('event %s / %s'%(topic.getName(), kwargs))

def main(args):
    global frame_data, frame_num
    cap = cv2.VideoCapture('test.mkv')
    if not cap.isOpened():
        print ('error opening')
        return
        
    #pub.subscribe(cb2, VODEvents.BOT_APPEAR)
    #pub.sendMessage(Capture.BOT_START, data={'test':'test1'})
    
    ret, frame = cap.read()
    fnext = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ydim, xdim, depth = frame.shape
    midx = xdim//2
    midy = ydim//2
    frate = cap.get(cv2.CAP_PROP_FPS)
    frames_total = int(cap.get( cv2.CAP_PROP_FRAME_COUNT))
    print ('Video = %s @ %s fps. %s frames'%(frame.shape,frate, frames_total))
    pub.subscribe(dbg_event, pub.ALL_TOPICS)
    pub.sendMessage(VODEvents.VOD_START, 
                    width= xdim,
                    height= ydim,
                    depth= depth,
                    frame_rate= frate,)
    
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', get_mouse)
    frame_num = 0
    SKIP_FRAMES = 210 #185 # skip VOD preamble
    show_delta = False
    flow = None
    while(cap.isOpened()):
        lframe = frame
        fprev = fnext
        ret, frame = cap.read()
        if not ret:
            break
        fnext = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        frame_num += 1
        if frame_num < SKIP_FRAMES:
            continue
        delta = frame.astype(np.int16) - lframe # allow negative range, >128 delta
        fdelta = fnext - fprev
        
        pub.sendMessage(VODEvents.VOD_FRAME,
                    timestamp= frame_num,
                    img= frame,
                    img_delta= delta,)
        # continue # skip user interface
        
        ''' calculate flow info
        abd = np.abs(fdelta)
        #print ('max=%s motion=%s'%(abd.max(), (abd>40).sum() ))
        
        CLIP_E = 50 # remove border pixels
        flow = cv2.calcOpticalFlowFarneback(fprev, fnext, flow, 0.5, 6, 35, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        mag_flow = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        
        avg_flow = (np.mean(flow[CLIP_E:-CLIP_E,CLIP_E:-CLIP_E,0]),
                    np.mean(flow[CLIP_E:-CLIP_E,CLIP_E:-CLIP_E,1]))
        avg_mag = 100*math.sqrt(avg_flow[0]**2+ avg_flow[1]**2)
        max_flow = np.max(mag_flow[CLIP_E:-CLIP_E,CLIP_E:-CLIP_E])
        #print ('magflow=%0.5f max=%s'%(avg_mag, max_flow))
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = np.clip(mag*2,0,255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #'''
        
        frame_data = delta
        #dout = np.array(delta*100)
        
        #flint = flow.astype(np.int16)
        #dout = np.zeros_like(fnext).astype(np.int16)
        dout = np.array(delta)
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
            key = cv2.waitKey(20)
            if key==-1:
                continue
            if key==ord('f'): # flip display
                show_delta = not show_delta
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
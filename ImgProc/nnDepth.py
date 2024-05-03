import math
import sys
import traceback
import json
import numpy as np
import cv2
import torch
import torchvision

from ImgProc import ImgTask

DOWNSCALE = 2
class nnDepth(ImgTask.ImgTask):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        device = torch.device('cpu')
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
            device = torch.device('cuda')
        self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
        self.model.to(device).eval()
        midas_tx = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.tx = midas_tx.dpt_transform
            
    def requires(self):
        return [ImgTask.IMG_BASE]
        
    def outputs(self):
        return ['depth']

    def proc_frame(self, frame):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
            device = torch.device('cuda')
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        (0, 0), fx=1./DOWNSCALE, fy=1./DOWNSCALE)
        model_in = self.tx(img).to(device)
        with torch.no_grad():
            prediction = self.model(model_in)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        out = prediction.cpu().numpy()/2048. + 1e-5
        out = cv2.resize(out, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        return out

_ = nnDepth()
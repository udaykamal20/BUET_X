# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 04:17:14 2019

@author: User
"""



import sys
import time
from PIL import Image, ImageDraw
from utils import *
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
# %matplotlib inline
import numpy as np
import dataset
import random
import math
import json
from region_loss import RegionLoss
from models import *
import h5py
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.manual_seed(int(time.time()))

transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],std = [ 0.25, 0.25, 0.25 ]),
                       ])

a = 'FullChannels_shift_nobias_final()'

model = eval(a)

load_net('bestFullChannels_shift_nobias().weights', model)
model.cuda()
region_loss = model.loss
count = 0



with open('test.txt', 'r') as outfile:
    lines = json.load(outfile)

test_infos = lines[0]

pred_infos = []
anchors     = model.anchors
num_anchors = model.num_anchors
anchor_step = int(len(anchors)/num_anchors)
proposals = 0.0
total = 0.0
all_iou = np.zeros(len(test_infos))

for i in tqdm(range(len(test_infos))):
    imgpath = test_infos[i][0]
    box = test_infos[i][1]
    label = np.zeros(4)
    label[0:2] = box[2:4]
    label[2:4] = box[0:2]
    img = Image.open(imgpath).convert('RGB')
    img = img.resize((model.width, model.height))
    timg = transform(img)
    timg = timg.view(1, 3, model.height, model.width)
    output = model(timg.cuda())
    output = output.data
    batch = output.size(0)
    h = output.size(2)
    w = output.size(3)
    output = output.view(batch*num_anchors, 5, h*w).transpose(0,1).contiguous().view(5, batch*num_anchors*h*w)
    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
    det_confs = torch.sigmoid(output[4])
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)        

    for b in range(batch):
        det_confs_inb = det_confs[b*sz_hwa:(b+1)*sz_hwa].numpy()
        xs_inb = xs[b*sz_hwa:(b+1)*sz_hwa].numpy()
        ys_inb = ys[b*sz_hwa:(b+1)*sz_hwa].numpy()
        ws_inb = ws[b*sz_hwa:(b+1)*sz_hwa].numpy()
        hs_inb = hs[b*sz_hwa:(b+1)*sz_hwa].numpy()      
        ind = np.argmax(det_confs_inb)

        bcx = xs_inb[ind]
        bcy = ys_inb[ind]
        bw = ws_inb[ind]
        bh = hs_inb[ind]

        box = [bcx/w, bcy/h, bw/w, bh/h]

        iou = bbox_iou(box, label, x1y1x2y2=False)
#        print (iou)
        all_iou[i]=iou
        
        pred_infos.append([imgpath,box])
        proposals = proposals + iou
        total = total+1
#    break
print(proposals/total)



# -*- coding: utf-8 -*-

import json
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import random

dataset_path = 'F:\\code_new\\dac\\data_training\\'
classes_name = []
data_train = []
data_test = []

class_tag_test = []
for path in tqdm(glob.glob(os.path.join(dataset_path, '*'))):
    
    classes_name.append(path.replace(dataset_path,''))
    class_tag_train = []
    
    all_gt = glob.glob(os.path.join(path, '*.xml'))
    random.shuffle(all_gt)
    ## train test split first and last 5% in test, remaining mid 90% in train
    train_gt = all_gt[int(0.05*len(all_gt)):int(0.95*len(all_gt))]
    test_gt = all_gt[0:int(0.05*len(all_gt))]
    test_gt = test_gt + all_gt[int(0.95*len(all_gt)):]
    
    ## train file create
    for gt_file in train_gt:
        class_tag_pair = []
        gt_img = gt_file.replace('.xml','.jpg')
        tree = ET.ElementTree(file=gt_file)
        b=np.zeros(4,dtype=float)
        for elem in tree.iter():
            if(elem.tag=='xmin'):
                b[0]=float(elem.text)
            if(elem.tag=='xmax'):
                b[1]=float(elem.text)
            if(elem.tag=='ymin'):
                b[2]=float(elem.text)
            if(elem.tag=='ymax'):
                b[3]=float(elem.text)
        if b[2]!= 0  and b[3]!= 0:
            img = Image.open(gt_img).convert('RGB')
            size = img.size
            bs = [(b[1]-b[0])*1./size[0],\
                  (b[3]-b[2])*1./size[1],\
                  (b[1]+b[0])*0.5/size[0],\
                 (b[2]+b[3])*0.5/size[1]]
            class_tag_pair.append(gt_img)
            class_tag_pair.append(bs)
            class_tag_train.append(class_tag_pair)
    data_train.append(class_tag_train)
    
    ##test file create
    for gt_file in test_gt:
        class_tag_pair = []
        gt_img = gt_file.replace('.xml','.jpg')
        tree = ET.ElementTree(file=gt_file)
        b=np.zeros(4,dtype=float)
        for elem in tree.iter():
            if(elem.tag=='xmin'):
                b[0]=float(elem.text)
            if(elem.tag=='xmax'):
                b[1]=float(elem.text)
            if(elem.tag=='ymin'):
                b[2]=float(elem.text)
            if(elem.tag=='ymax'):
                b[3]=float(elem.text)
        if b[2]!= 0  and b[3]!= 0:
            img = Image.open(gt_img).convert('RGB')
            size = img.size
            bs = [(b[1]-b[0])*1./size[0],\
                  (b[3]-b[2])*1./size[1],\
                  (b[1]+b[0])*0.5/size[0],\
                 (b[2]+b[3])*0.5/size[1]]
            class_tag_pair.append(gt_img)
            class_tag_pair.append(bs)
            class_tag_test.append(class_tag_pair)
    
    print(path)

data_test.append(class_tag_test)
## file dump 
with open('train.txt', 'w') as outfile:
    json.dump(data_train, outfile)


with open('test.txt', 'w') as outfile:
    json.dump(data_test, outfile)


#%% check the files just dumped

## helper functions for plotting
def show_img(img, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img),
    return ax

def draw_outline(matplt_plot_obj, lw):    
    matplt_plot_obj.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, bbox, color='white'):
    patch = ax.add_patch(patches.Rectangle(bbox[:2], *bbox[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, text, sz=14, color='white'):
    pass
    txt = ax.text(*xy, text, verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(txt, 1)

def adjust_bb(img,bb):
    w = bb[0]*img.size[0]
    h = bb[1]*img.size[1]
    x = bb[2]*img.size[0]
    y = bb[3]*img.size[1]
    x_min = x - w/2
    y_min = y - h/2
    return [x_min, y_min, w, h]
    
def draw_image(im, annot):
    ax = show_img(im, figsize=(16, 8))
    cat = annot[0].split('\\')[-2]
    bb = adjust_bb(im,annot[1])
    draw_rect(ax, bb)
    draw_text(ax, bb[:2], cat, sz=16)
        
def draw_idx(data_loaded,class_num,sample_num):
    im_annot = data_loaded[class_num][sample_num]
    im = Image.open(im_annot[0])
    draw_image(im, im_annot)

#for debugging purpose

#with open('train.txt', 'r') as outfile:
#    data_loaded_train = json.load(outfile)
#
#with open('test.txt', 'r') as outfile:
#    data_loaded_test = json.load(outfile)
#
#ind = np.random.randint(0,len(classes_name))
#
#draw_idx(data_loaded_train,ind,5)
#draw_idx(data_loaded_test,ind,-1)

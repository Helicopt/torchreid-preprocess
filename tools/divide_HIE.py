#!/usr/bin/python
# -*- coding: utf8 -*-
#########################################################################
# File Name: tools/divide_dataset.py
# Author: Toka
# mail: <empty>
# Created Time: Thu Jul 23 16:31:15 2020
#########################################################################
import numpy as np
import cv2
from math import *
import random
import os
import sys
from senseTk.common import *

HIE20_train = [str(i + 1) for i in range(19)]
HIE20_root = '/mnt/lustre/share/fengweitao/LHVACE/HIE20'
MOT16_train = ['MOT16-%02d'%i for i in [2, 4, 5, 9, 10, 11, 13]]
MOT16_root = '/mnt/lustre/share/fengweitao/MOT16'

def divide_dataset(seqs, root='', val=0.5, gap=30, gstart=0, tag='st.i'):
    gt_dir = os.path.join(root, 'labels/train/track1/%s.txt')
    dt_dir = os.path.join(root, 'dets/train/%s.txt')
    img_dir = os.path.join(root, 'images/train/%s/')
    trains = []
    train_f = 0
    train_b = 0
    vals = []
    val_f = 0
    val_b = 0
    for seq in seqs:
        #gt_file = os.path.join(gt_dir, seq + '.txt')
        #dt_file = os.path.join(dt_dir, seq + '.txt')
        gt_file = gt_dir % seq
        dt_file = dt_dir % seq
        if not os.path.exists(gt_file):
            gt_file = os.path.join(gt_dir, seq)
        assert os.path.exists(gt_file)
        gt = TrackSet(gt_file)
        dt = TrackSet(dt_file)
        imgset = VideoClipReader(os.path.join(img_dir % seq))
        n = len(imgset)
        val_n = int((n - gap) * val)
        train_n = n - gap - val_n
        if val_n < 100 :
            train_n = 0
            val_n = n
        if train_n:
            seq_t = seq + '-t'
            new_imdir = os.path.join(img_dir % seq_t)
            os.makedirs(new_imdir, exist_ok=True)
            #os.makedirs(os.path.join(new_imdir.replace('img1', 'det')), exist_ok=True)
            #os.makedirs(os.path.join(new_imdir.replace('img1', 'gt')), exist_ok=True)
            new_trkset = TrackSet()
            new_trkset2 = TrackSet()
            for i in range(train_n):
                im = imgset[i]
                cv2.imwrite(os.path.join(new_imdir, '%06d.jpg'%(i+1)), im)
                for d in dt[i + gstart]:
                    d.fr = i + 1
                    new_trkset2.append_data(d)
                for d in gt[i + gstart]:
                    d.fr = i + 1
                    new_trkset.append_data(d)
                    train_b += 1
                train_f += 1
            with open(os.path.join(gt_dir % seq_t), 'w') as fd:
                new_trkset.dump(fd, formatter='fr.i, id.i, x1, y1, w, h, %s, -1, -1, -1'%tag)
            with open(os.path.join(dt_dir % seq_t), 'w') as fd:
                new_trkset2.dump(fd, formatter='fr.i, id.i, x1, y1, w, h, cf, -1, -1, -1')
            trains.append(seq_t)
        if val_n:
            seq_t = seq + '-v'
            new_imdir = os.path.join(img_dir % seq_t)
            os.makedirs(new_imdir, exist_ok=True)
            #os.makedirs(os.path.join(new_imdir.replace('img1', 'det')), exist_ok=True)
            #os.makedirs(os.path.join(new_imdir.replace('img1', 'gt')), exist_ok=True)
            new_trkset = TrackSet()
            new_trkset2 = TrackSet()
            offset = (train_n + gap) if train_n else 0
            for i in range(val_n):
                i = i + offset
                im = imgset[i]
                cv2.imwrite(os.path.join(new_imdir, '%06d.jpg'%(i-offset+1)), im)
                for d in dt[i + gstart]:
                    d.fr = i - offset + 1
                    new_trkset2.append_data(d)
                for d in gt[i + gstart]:
                    d.fr = i - offset + 1
                    new_trkset.append_data(d)
                    val_b += 1
                val_f += 1
            with open(os.path.join(gt_dir % seq_t), 'w') as fd:
                new_trkset.dump(fd, formatter='fr.i, id.i, x1, y1, w, h, %s, -1, -1, -1'%tag)
            with open(os.path.join(dt_dir % seq_t), 'w') as fd:
                new_trkset2.dump(fd, formatter='fr.i, id.i, x1, y1, w, h, cf, -1, -1, -1')
            vals.append(seq_t)
    with open(os.path.join(root, 'train.list.txt'), 'w') as fd:
        for i in trains:
            fd.write(i + '\n')
    with open(os.path.join(root, 'val.list.txt'), 'w') as fd:
        for i in vals:
            fd.write(i + '\n')
    with open(os.path.join(root, 'trainval.info.txt'), 'w') as fd:
        fd.write('train: %d frames, %d boxes, %d videos\n'%(train_f, train_b, len(trains)))
        fd.write('val: %d frames, %d boxes, %d videos\n'%(val_f, val_b, len(vals)))

if __name__=='__main__':
    # __author__ == '__toka__'
    divide_dataset(HIE20_train, root=HIE20_root, val=0.3, gap=30, gstart=0, tag='cf.i')
    # divide_dataset(MOT16_train, root=MOT16_root, val=0.5, gap=30, gstart=1, tag='st.i')

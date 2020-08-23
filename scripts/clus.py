#!/usr/bin/python
# -*- coding: utf8 -*-
#########################################################################
# File Name: scripts/clus.py
# Author: Toka
# mail: <empty>
# Created Time: Sun Aug  9 21:47:06 2020
#########################################################################
import numpy as np
import cv2
from math import *
import random
import os
import sys
import yaml
import torch
import pickle
from sklearn.cluster import SpectralClustering, KMeans

feats_list = [
    '/mnt/lustre/share/fengweitao/MOT16_feats.pkl',
    '/mnt/lustre/share/fengweitao/LHVACE/HIE20_feats.pkl',
    # '/mnt/lustre/share/fengweitao/crowd_human_feats.pkl',
]
num_cluster = 30

if __name__ == '__main__':
    # __author__ == '__toka__'
    data = []
    for i in feats_list:
        data_ = torch.load(i)
        data += data_
    X = np.stack([da['feat'] for da in data])
    X = X / np.sqrt((X * X).sum(axis=1)).reshape(-1, 1)
    # adjacency_matrix = np.matmul(X, X.transpose(1, 0))
    #modlen = np.sqrt((X * X).sum(axis=1)).reshape(-1, 1)
    #modlen = np.matmul(modlen, modlen.transpose(1, 0))
    #adjacency_matrix /= modlen
    #print(adjacency_matrix)
    #sc = SpectralClustering(num_cluster, affinity='precomputed', n_init=100, assign_labels='discretize')
    #ret = sc.fit_predict(adjacency_matrix)
    km = KMeans(init='k-means++', n_clusters=num_cluster, n_init=num_cluster // 3)
    ret = km.fit_predict(X)
    for i in range(num_cluster):
        inds = (ret == i).nonzero()
        print(ind.shape[0])
    results = {}
    for data, label in zip(data, ret):
        seq, uid = data['filename'], data['uid']
        if seq not in results:
            results[seq] = {}
        # results.append([seq, int(uid), int(label)])
        results[seq][int(uid)] = int(label)
    with open('meta_cluster.pkl', 'wb') as fd:
        pickle.dump(results, fd)
    # print(ret)
    # inds = []
    # for i in range(num_cluster):
    #     ind = ret == i
    #     inds.append(ind)
    # dm = adjacency_matrix
    # for j in range(dm.shape[0]):
    #     dm[j, j] = -2
    # m = np.argmax(adjacency_matrix, axis=1)
    # print(m)
    # for i in range(num_cluster):
    #     mmax = -1.
    #     mmin = 1.
    #     feats1 = X[inds[i]]
    #     dm = np.matmul(feats1, feats1.transpose(1, 0))
    #     for j in range(dm.shape[0]):
    #         dm[j, j] = dm[j, (j+1) % dm.shape[0]]
    #     max_self, min_self = dm.max(), dm.min()
    #     for j in range(num_cluster):
    #         if i == j: continue
    #         feats2 = X[inds[j]]
    #         dm = np.matmul(feats1, feats2.transpose(1, 0))
    #         #print(dm.max(), dm.min())
    #         mmax = max(mmax, dm.max())
    #         mmin = min(mmin, dm.min())
    #     print(mmax, mmin, max_self, min_self, len(inds[i].nonzero()[0]))

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

if __name__=='__main__':
    # __author__ == '__toka__'
    import torch
    num_cluster = 30
    data = torch.load('MOT16_cluster.pkl')
    X = np.stack([da['feat'] for da in data])
    X = X / np.sqrt((X * X).sum(axis=1)).reshape(-1, 1)
    adjacency_matrix = np.matmul(X, X.transpose(1, 0))
    #modlen = np.sqrt((X * X).sum(axis=1)).reshape(-1, 1)
    #modlen = np.matmul(modlen, modlen.transpose(1, 0))
    #adjacency_matrix /= modlen
    #print(adjacency_matrix)
    from sklearn.cluster import SpectralClustering, KMeans
    #sc = SpectralClustering(num_cluster, affinity='precomputed', n_init=100, assign_labels='discretize')
    #ret = sc.fit_predict(adjacency_matrix)
    km = KMeans(init='k-means++', n_clusters=num_cluster, n_init=30)
    ret = km.fit_predict(X)
    results = []
    for data, label in zip(data, ret):
        seq, uid = data['seq'], data['uid']
        results.append([seq, int(uid), int(label)])
    with open('meta_res.yaml', 'w') as fd:
        yaml.dump(results, fd)
    print(ret)
    inds = []
    for i in range(num_cluster):
        ind = ret == i
        inds.append(ind)
    dm = adjacency_matrix
    for j in range(dm.shape[0]):
        dm[j, j] = -2
    m = np.argmax(adjacency_matrix, axis=1)
    print(m)
    for i in range(num_cluster):
        mmax = - 1.
        mmin = 1.
        feats1 = X[inds[i]]
        dm = np.matmul(feats1, feats1.transpose(1,0))
        for j in range(dm.shape[0]):
            dm[j, j] = dm[j, (j + 1) % dm.shape[0]]
        max_self, min_self = dm.max(), dm.min()
        for j in range(num_cluster):
            if i == j : continue
            feats2 = X[inds[j]]
            dm = np.matmul(feats1, feats2.transpose(1, 0))
            #print(dm.max(), dm.min())
            mmax = max(mmax, dm.max())
            mmin = min(mmin, dm.min())
        print(mmax, mmin, max_self, min_self, len(inds[i].nonzero()[0]))

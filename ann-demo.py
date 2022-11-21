#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:27:38 2020

@author: ttonteri
"""

from mrpt import MRPT
from utility import dist

import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def sim(ax, m, n, k, M=2, method='RP', seed=None):
    if seed: np.random.seed(seed)

    X_corpus, Y_corpus = make_blobs(n_samples=m, n_features=2, 
                                    centers=1, center_box=(-.1,.1),
                                    cluster_std=.8)
    X, Y = make_blobs(n_samples=n, n_features=2, centers=1, center_box=(-.1,.1),
                      cluster_std=.8)

    if seed: np.random.seed(10000000+seed)  # fix RPs for different n

    # scale the data so that all values are between 0.0 and 1.0
    #X = MinMaxScaler().fit(X).transform(X)

    X_train = X
    X_test = np.array(np.random.normal(scale=0.3, size=2), ndmin=2)

    square = Polygon([(-M,-M), (-M,M), (M,M), (M,-M)])
    partition = MRPT(square, X_train, min_n=k*1.33, kd=method)

    [cell] = [part for part in partition if part.contains(Point(X_test[0]))]

    for i in range(len(X_test)):
        # calculate the distances to all training points
        D = np.empty(len(X_corpus))
        for j in range(len(X_corpus)):
            D[j] = dist(X_corpus[j], X_test[i])
            
        # find the index of the nearest neighbor
        near = np.argsort(D)[:k]

    # find the nearest neighbors for the data in the cell

    counts = np.zeros(len(X_corpus), dtype=np.int64) 
    trN = 0
    for i in range(len(X_train)):
        if cell.contains(Point(X_train[i])):
            trN += 1
            D = np.empty(len(X_corpus))
            for j in range(len(X_corpus)):
                D[j] = dist(X_corpus[j], X_train[i])
            
            # find the index of the nearest neighbor
            near_loop = np.argsort(D)[:k]
            for j in near_loop:
                counts[j] += 1

    # plot the chart
    
    #ax = plt.gca()
    ax.set_axis_off()
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,
        labelleft=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) 

    #fig.set_facecolor('#EBE9EF')
    #cm = ListedColormap(['#5EC798', '#46479D'])
    
    #ax.set_facecolor('#FCFDF1')
    cm = ListedColormap(['#CCD1D1', '#E74C3C'])
    
    show_counts = False
    
    S = np.argsort(-counts)[:k]
    
    Y_corpus[S] = 1
    
    ax.scatter(X_corpus[near,0], X_corpus[near,1], marker='o', 
                cmap=cm, s=80, c='#F0B27A', edgecolors=None)
    if show_counts:
        labs = [str(c)+'/'+str(trN) for c in counts]
        for i in range(len(labs)):
            ax.annotate(labs[i], xy=(X_corpus[i,0],X_corpus[i,1]), 
                        xytext=(0,8), textcoords='offset points',
                        fontsize=8,
                        horizontalalignment='center', verticalalignment='bottom')
    
    ax.scatter(X_corpus[:,0], X_corpus[:,1], c=Y_corpus, marker='o',
               cmap=cm, s=15, vmin=0, vmax=1, edgecolors=None)
    
    for part in partition:
        ax.plot(*part.exterior.xy, color='#616A6B', lw=.5)
    ax.plot(*cell.exterior.xy, color='blue', lw=3)
    ax.scatter(X_test[:,0], X_test[:,1], c='#FFF681', marker='*', 
                s=260, vmin=0, vmax=1, edgecolors='k')
    ax.set(xlim=(-M-.013, M+.013), ylim=(-M-.013, M+.013))

    #ax.xlim([-M, M])
    #ax.ylim([-M, M])
        
#np.random.seed(1)  # 289 (n=2000,k=20)
seed = 8 #np.random.randint(10000)
m = 512  # corpus size
ns = [2**r for r in range(5,15)]  # training data size
k = 25    # k
M = 2     # bounding box size
method = 'RP' #'PCA' # KD, RP, PCA, or 0 (=chron. KD)

R = np.random.normal(size=(int(np.ceil(np.log2(max(ns)))),2))

fig = plt.figure(figsize=(26,12))
fig.set_tight_layout(True)

for (i,n) in [(i,ns[i]) for i in range(len(ns))]:
    ax = fig.add_subplot(2,5,i+1)
    sim(ax, m, n, k, M, method, seed)
    
plt.savefig("mrptdemo_grid-demo-test.pdf")

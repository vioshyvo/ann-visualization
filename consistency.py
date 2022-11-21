#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:27:38 2020

@author: ttonteri
"""

from utility import generate_toy_data, dist, nearest_neighbors
from mrpt import MRPT

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


######################################################################
# Main plotting code

training_set_sizes = [50, 250, 1000]
k = 5   # set k=0 to just show the data
M = 0.7                                                 
n_0 = 8       
tau = 0.2                                          # maximum leaf size
method = 'PCA'
show_counts = False
X_corpus, Y_corpus = generate_toy_data(15)              # corpus
X_test, Y_test = (np.array([[.5,.5]]), np.array([0]))   # query point
square = Polygon([(X_test[0][0]-M,X_test[0][1]-M), (X_test[0][0]-M,X_test[0][1]+M),
                (X_test[0][0]+M,X_test[0][1]+M), (X_test[0][0]+M,X_test[0][1]-M)])
n_max = training_set_sizes[len(training_set_sizes)-1]
np.random.seed(20)
X = np.resize(np.random.uniform(X_test[0][0]-M, X_test[0][0]+M,2*n_max), (n_max,2))  

for n in training_set_sizes:    
    X_train = X[0:n,:]  

    partition = MRPT(square, X_train, min_n=n_0, kd=method)
    [cell] = [part for part in partition if part.contains(Point(X_test[0]))]
    
    # place-holder for the predicted classes
    Y_predict = np.empty(len(Y_test), dtype=np.int64)
        
    for i in range(len(X_test)):
        # calculate the distances to all training points
        D = np.empty(len(X_corpus))
        for j in range(len(X_corpus)):
            D[j] = dist(X_corpus[j], X_test[i])
                
        # find the index of the nearest neighbor
        near = np.argsort(D)[:k]
    
    # do the nearest neighbors for the data in the cell
    counts, trN = nearest_neighbors(X_train, X_corpus, cell, k)
        
    # plot the chart
    fig = plt.figure(figsize=(6,6))
    fig.set_tight_layout(True)
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))
    plt.tick_params(
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
    cm = ListedColormap(['#86B853', '#FFF681'])
    
    S = np.argsort(-counts)[:k]
    #Y_corpus = [1 if c>2 else 0 for c in counts]
    #Y_corpus[S] = 1
    
    if n == n_max:
        alpha = .5
    else:
        alpha = .7
    
    ann = counts / trN > tau
    plt.scatter(X_train[:,0], X_train[:,1], c='#FFAFEC', s=30, alpha = alpha)
    plt.scatter(X_corpus[near,0], X_corpus[near,1], marker='o', 
                cmap=cm, s=301, c='#F0B27A', edgecolors=None)
    plt.scatter(X_corpus[:,0], X_corpus[:,1], c=Y_corpus, marker='o',
                cmap=cm, s=51, vmin=0, vmax=1, edgecolors='k')
    plt.scatter(X_test[:,0], X_test[:,1], c='#FFF681', marker='*', 
                cmap=cm, s=250, vmin=0, vmax=1, edgecolors='k')
    
    for part in partition:
        ax.plot(*part.exterior.xy, color='#3977AF', alpha = 0.1)
        
    plt.scatter(X_test[:,0], X_test[:,1], marker='*',
                s=220, vmin=0, vmax=1, facecolors='#FFF681', edgecolors='k')
    
    plt.savefig("fig/fig-" + method + "-n_0-" + str(n_0)  + "-n-" + str(n) + "-consistency.pdf")

    px, py = cell.exterior.xy
    plt.plot(px, py) # color='#3977AF'
    
    if show_counts:
        labs = [str(c)+'/'+str(trN) for c in counts]
        for i in range(len(labs)):
            plt.annotate(labs[i], xy=(X_corpus[i,0],X_corpus[i,1]), 
                         xytext=(0,8), textcoords='offset points',
                         fontsize=8,
                         horizontalalignment='center', verticalalignment='bottom')
    
    for i in np.arange(len(X_train)):
        if cell.contains(Point(X_train[i])):
            plt.scatter(X_train[i,0], X_train[i,1], c='#FFAFEC', s=30, edgecolors='k')

    plt.savefig("fig/fig-" + method + "-n_0-" + str(n_0)  + "-n-" + str(n) + "-consistency-cell-highlighted.pdf")
            
    plt.scatter(X_corpus[ann,0], X_corpus[ann,1], marker='o', 
                cmap=cm, s=51, vmin=0, vmax=1, c='#FFF681', edgecolors='k')

    plt.savefig("fig/fig-" + method + "-n_0-" + str(n_0)  + "-n-" + str(n) + "-consistency-cell-no_labels.pdf")


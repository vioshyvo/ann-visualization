#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:27:38 2020

@author: ttonteri
"""

from utility import generate_toy_data, nearest_neighbors, knn
from mrpt import MRPT

import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


def main():
    training_set_sizes = [50, 250, 1000]
    k = 5   # set k=0 to just show the data
    M = 0.7                                                 
    n_0 = 8       
    tau = 0.2                                          
    method = 'PCA'
    X_corpus, Y_corpus = generate_toy_data(15)             
    X_test, Y_test = (np.array([[.5,.5]]), np.array([0]))   
    square = Polygon([(X_test[0][0]-M,X_test[0][1]-M), (X_test[0][0]-M,X_test[0][1]+M),
                    (X_test[0][0]+M,X_test[0][1]+M), (X_test[0][0]+M,X_test[0][1]-M)])
    n_max = training_set_sizes[len(training_set_sizes)-1]
    np.random.seed(20)
    X = np.resize(np.random.uniform(X_test[0][0]-M, X_test[0][0]+M,2*n_max), (n_max,2))  
    
    for n in training_set_sizes:    
        X_train = X[0:n,:]  
    
        partition = MRPT(square, X_train, min_n=n_0, kd=method)
        [cell] = [part for part in partition if part.contains(Point(X_test[0]))]
            
        # true knn of the query point
        near = knn(X_test, X_corpus, k)    
        
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
        
        cm = ListedColormap(['#86B853', '#FFF681'])
        c_train = '#FFAFEC'
        c_neighbor = '#F0B27A'
        msize_corpus = 51
        msize_query = 250
        msize_neighbor = 301
        msize_train = 30
        alpha = 0.5 if n == n_max else 0.7
        
        ann = counts / trN > tau
        plt.scatter(X_train[:,0], X_train[:,1], color=c_train, s=msize_train, alpha = alpha)
        plt.scatter(X_corpus[near,0], X_corpus[near,1], marker='o', 
                    cmap=cm, s=msize_neighbor, color=c_neighbor, edgecolors=None)
        plt.scatter(X_corpus[:,0], X_corpus[:,1], c=Y_corpus, marker='o',
                    cmap=cm, s=msize_corpus, vmin=0, vmax=1, edgecolors='k')
        plt.scatter(X_test[:,0], X_test[:,1], c=1, marker='*', 
                    cmap=cm, s=msize_query, vmin=0, vmax=1, edgecolors='k')
        
        for part in partition:
            ax.plot(*part.exterior.xy, color='#3977AF', alpha = 0.1)
        
        plt.savefig("fig/fig-" + method + "-n_0-" + str(n_0)  + "-n-" + str(n) + "-consistency.pdf")
    
        px, py = cell.exterior.xy
        plt.plot(px, py) # color='#3977AF'
            
        for i in np.arange(len(X_train)):
            if cell.contains(Point(X_train[i])):
                plt.scatter(X_train[i,0], X_train[i,1], color=c_train, s=msize_train, edgecolors='k')
    
        plt.savefig("fig/fig-" + method + "-n_0-" + str(n_0)  + "-n-" + str(n) + "-consistency-cell-highlighted.pdf")
                
        plt.scatter(X_corpus[:,0], X_corpus[:,1], marker='o', 
                    cmap=cm, s=msize_corpus, vmin=0, vmax=1, c=ann, edgecolors='k')
    
        plt.savefig("fig/fig-" + method + "-n_0-" + str(n_0)  + "-n-" + str(n) + "-consistency-cell-candidate-set.pdf")
    
        labs = [str(c)+'/'+str(trN) for c in counts]
        for i in range(len(labs)):
            plt.annotate(labs[i], xy=(X_corpus[i,0],X_corpus[i,1]), 
                         xytext=(0,8), textcoords='offset points',
                         fontsize=8,
                         horizontalalignment='center', verticalalignment='bottom')
    
        plt.savefig("fig/fig-" + method + "-n_0-" + str(n_0)  + "-n-" + str(n) + "-consistency-cell-counts.pdf")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        print('Usage:', sys.argv[0])
        sys.exit(-1)

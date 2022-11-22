#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:27:38 2020

@author: ttonteri
"""

from utility import generate_toy_data, create_polygon, dist, nearest_neighbors

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry.polygon import Polygon


k = 5   # set k=0 to just show the data
seed = 15 
X_train, Y_train = generate_toy_data(seed)       
X_test, Y_test = (np.array([[.5,.5]]), np.array([0]))

# made up partition element
cell = Polygon(create_polygon(X_test[0,0]-.145, X_test[0,1]+.0, 6, .295))

# place-holder for the predicted classes
Y_predict = np.empty(len(Y_test), dtype=np.int64)


for i in range(len(X_test)):
    # calculate the distances to all training points
    D = np.empty(len(X_train))
    for j in range(len(X_train)):
        D[j] = dist(X_train[j], X_test[i])
            
    # find the index of the nearest neighbor
    near = np.argsort(D)[:k]

# do the nearest neighbors for the data in the cell
counts, trN = nearest_neighbors(X_train, X_train, cell, k)

# plot the chart
fig = plt.figure(figsize=(6,6))
fig.set_tight_layout(True)
ax = plt.gca()
ax.set_axis_off()
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

show_counts = False

S = np.argsort(-counts)[:k]
#Y_train = [1 if c>2 else 0 for c in counts]
#Y_train[S] = 1

if k > 0:
    plt.scatter(X_train[near,0], X_train[near,1], marker='o', 
                cmap=cm, s=301, c='#F0B27A', edgecolors=None)
    plt.scatter(X_test[:,0], X_test[:,1], c=Y_predict, marker='*', 
                cmap=cm, s=250, vmin=0, vmax=1, edgecolors='k')
    if show_counts:
        labs = [str(c)+'/'+str(trN) for c in counts]
        for i in range(len(labs)):
            plt.annotate(labs[i], xy=(X_train[i,0],X_train[i,1]), 
                         xytext=(0,8), textcoords='offset points',
                         fontsize=8,
                         horizontalalignment='center', verticalalignment='bottom')

else:
    plt.scatter(X_test[:,0], X_test[:,1], marker='*',
                s=220, vmin=0, vmax=1, facecolors='w', edgecolors='k')

plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, marker='o',
            cmap=cm, s=51, vmin=0, vmax=1, edgecolors='k')

px, py = cell.exterior.xy
plt.plot(px, py) # #3977AF

plt.savefig("fig/fig2-new.pdf")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:27:38 2020

@author: ttonteri
"""

from utility import generate_toy_data, dist
from mrpt import MRPT

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


######################################################################
# Main plotting code

X_train, Y_train = generate_toy_data(15)              # corpus
X_test, Y_test = (np.array([[.5,.5]]), np.array([0])) # query point

# made up partition element
# cell = Polygon(create_polygon(X_test[0,0]-.145, X_test[0,1]+.0, 6, .295))
M = 0.7
n_0 = 4     # maximum leaf size
method = 'KD'
square = Polygon([(X_test[0][0]-M,X_test[0][1]-M), (X_test[0][0]-M,X_test[0][1]+M),
                (X_test[0][0]+M,X_test[0][1]+M), (X_test[0][0]+M,X_test[0][1]-M)])
partition = MRPT(square, X_train, min_n=n_0, kd=method)
[cell] = [part for part in partition if part.contains(Point(X_test[0]))]


# place-holder for the predicted classes
Y_predict = np.empty(len(Y_test), dtype=np.int64)

k = 5   # set k=0 to just show the data

for i in range(len(X_test)):
    # calculate the distances to all training points
    D = np.empty(len(X_train))
    for j in range(len(X_train)):
        D[j] = dist(X_train[j], X_test[i])
            
    # find the index of the nearest neighbor
    near = np.argsort(D)[:k]

# do the nearest neighbors for the data in the cell

counts = np.zeros(len(X_train), dtype=np.int64) 
trN = 0
for i in range(len(X_train)):
    if cell.contains(Point(X_train[i])):
        trN += 1
        D = np.empty(len(X_train))
        for j in range(len(X_train)):
            D[j] = dist(X_train[j], X_train[i])
            
        # find the index of the nearest neighbor
        near_loop = np.argsort(D)[:k]
        for j in near_loop:
            counts[j] += 1

#Y_train[near] = 2

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

for part in partition:
    ax.plot(*part.exterior.xy, color='#3977AF')

# px, py = cell.exterior.xy
# print('xlim:', plt.gca().get_xlim())
# print('ylim:', plt.gca().get_ylim())
# plt.plot(px, py) # color='#3977AF'

plt.scatter(X_test[:,0], X_test[:,1], marker='*',
            s=220, vmin=0, vmax=1, facecolors='#FFF681', edgecolors='k')

plt.savefig("fig-" + method + '-' + str(n_0)  +".pdf")

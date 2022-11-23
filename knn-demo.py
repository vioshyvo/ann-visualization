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
from shapely.geometry import Point


k = 5   
tau = 0.2
seed = 15 
X_corpus, Y_train = generate_toy_data(seed)       
X_test, Y_test = (np.array([[.5,.5]]), np.array([0]))

# made up partition element
cell = Polygon(create_polygon(X_test[0,0]-.145, X_test[0,1]+.0, 6, .295))

for i in range(len(X_test)):
    # calculate the distances to all training points
    D = np.empty(len(X_corpus))
    for j in range(len(X_corpus)):
        D[j] = dist(X_corpus[j], X_test[i])
            
    # find the index of the nearest neighbor
    near = np.argsort(D)[:k]

# do the nearest neighbors for the data in the cell
counts, trN = nearest_neighbors(X_corpus, X_corpus, cell, k)

# plot the chart
cm = ListedColormap(['#86B853', '#FFF681'])
msize_corpus = 51
msize_neighbor = 250

fig = plt.figure(figsize=(6,6))
fig.set_tight_layout(True)
ax = plt.gca()
ax.set_axis_off()
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,
    labelleft=False,
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) 

plt.scatter(X_corpus[:,0], X_corpus[:,1], c=Y_train, marker='o',
            cmap=cm, s=msize_corpus, vmin=0, vmax=1, edgecolors='k')
plt.scatter(X_test[:,0], X_test[:,1], c=1, marker='*', 
            cmap=cm, s=msize_neighbor, vmin=0, vmax=1, edgecolors='k')

plt.savefig("fig/fig2-plain.pdf")

plt.scatter(X_corpus[near,0], X_corpus[near,1], marker='o', 
            cmap=cm, s=301, c='#F0B27A', edgecolors=None)
plt.scatter(X_corpus[:,0], X_corpus[:,1], c=Y_train, marker='o',
            cmap=cm, s=msize_corpus, vmin=0, vmax=1, edgecolors='k')

plt.savefig("fig/fig2-corpus.pdf")

px, py = cell.exterior.xy
plt.plot(px, py) # #3977AF

plt.savefig("fig/fig2-new.pdf")

candidate_set = np.full(len(X_corpus), False)
for i in np.arange(len(X_corpus)):
    if cell.contains(Point(X_corpus[i])):
        candidate_set[i] = True
    
plt.scatter(X_corpus[:,0], X_corpus[:,1], c=candidate_set, marker='o',
            cmap=cm, s=msize_corpus, vmin=0, vmax=1, edgecolors='k')

plt.savefig("fig/fig2-candidate-set.pdf")

candidate_set = counts / trN > tau
plt.scatter(X_corpus[:,0], X_corpus[:,1], c=candidate_set,
            marker='o', cmap=cm, s=msize_corpus, vmin=0, vmax=1, edgecolors='k')

plt.savefig("fig/fig2-candidate-set-multilabel.pdf")

labs = [str(c)+'/'+str(trN) for c in counts]
for i in range(len(labs)):
    plt.annotate(labs[i], xy=(X_corpus[i,0],X_corpus[i,1]), 
                 xytext=(0,8), textcoords='offset points',
                 fontsize=8,
                 horizontalalignment='center', verticalalignment='bottom')

plt.savefig("fig/fig2-counts.pdf")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:48:43 2022

@author: DJKesoil
"""

import math
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Point


######################################################################
# Function for generating the toy data set

def generate_toy_data(seed=15, n_samples=25, n_features=2, centers=1, 
                      center_box=(-2,2), mask_xy = (0.7,0.7)):
    np.random.seed(seed)  #15 9    
    X, Y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=centers, center_box=center_box)
    
    # scale the data so that all values are between 0.0 and 1.0
    
    X = MinMaxScaler().fit(X).transform(X)
    
    mask = (X[:,0]<mask_xy[0]) | (X[:,1]<mask_xy[1])
    return (X[mask], Y[mask])

######################################################################
# Functions for creating a generic toy example polygon

def polar_to_xy(x0, y0, a, r):
    return (x0 + math.cos(a)*r, y0 + math.sin(a)*r)

def create_polygon(x0, y0, corners, r):
    angle_inc = np.random.dirichlet(([2]*corners), 1)[0,:] * math.pi*2
    angles = [sum(angle_inc[0:(i+1)]) for i in range(corners)]
    rs = np.random.normal(np.sqrt(r), r/5, corners)**2
    cs = zip(angles, rs)
    xy = [polar_to_xy(x0, y0, a, r) for (a,r) in cs]
    return xy

######################################################################
# Utility functions

# distance function
def dist(a, b):
    sum = 0
    for i in range(len(a)):
        sum = sum + (a[i] - b[i])**2
    return math.sqrt(sum)

def knn(X_test, X_corpus, k):
    for i in range(len(X_test)):
        # calculate the distances to all training points
        D = np.empty(len(X_corpus))
        for j in range(len(X_corpus)):
            D[j] = dist(X_corpus[j], X_test[i])
                
        # find the index of the nearest neighbor
        return np.argsort(D)[:k]


def nearest_neighbors(X_train, X_corpus, cell, k):
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
    return (counts, trN)


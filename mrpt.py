#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:49:03 2022

@author: DJKesoil
"""

import sys 
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import split
from sklearn.decomposition import PCA


######################################################################
# Functions for creating RP / k-d / PCA trees

def split_region(parent, X, kd='RP', depth=0, path=None):
    global R
    
    d = X.shape[1]
    if kd == 'RP':
        # random projections
        # r = R[depth] # same on each level
        if path: np.random.seed(path)
        r = np.random.normal(size=d)
    elif kd == 'PCA':
        pca = PCA(2).fit(X)
        r = pca.components_[0]
    elif kd == 'KD':
        r = np.zeros(d)
        r[np.argmax([np.var(X[:,l]) for l in range(d)])] = 1.0
    elif isinstance(kd,int):
        # chronological kd-tree
        r = np.zeros(d)
        r[kd] = 1.0
    else:
        sys.exit("I'm afraid I don't know method " + str(kd) + ", Dave")
        
    xp = X @ r
    t = np.median(xp)

    par_min = np.min(parent.exterior.xy, axis=1)
    par_max = np.max(parent.exterior.xy, axis=1)
        
    if r[1]:
        p1 = Point([par_min[0], (t-par_min[0]*r[0])/r[1]])
        p2 = Point([par_max[0], (t-par_max[0]*r[0])/r[1]])
    else:
        p1 = Point([(t-par_min[1]*r[1])/r[0], par_min[1]])
        p2 = Point([(t-par_max[1]*r[1])/r[0], par_max[1]])
        
    hpp = LineString([p1, p2])

    children = split(parent, hpp)
    
    return children.geoms
                          
def MRPT(cell, X, min_n=5, kd='RP', depth=0, path=0):
    if len(X) < min_n:
        return [cell]
    
    partition = []
    children = split_region(cell, X, kd, depth, path)
    for (i,child) in [(i,children[i]) for i in range(len(children))]:
        partition += MRPT(child, 
                          X[[child.contains(Point(x)) for x in X]],
                          min_n, 
                          (kd+1)%X.shape[1] if isinstance(kd,int) else kd,
                          depth+1, 2*path+i)
    
    return partition        

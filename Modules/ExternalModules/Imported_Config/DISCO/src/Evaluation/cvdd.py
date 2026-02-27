# Implementation of CVDD by
# - Author: Lianyu Hu - Github user `hulianyu`
# - Source: https://github.com/hulianyu/CVDD/blob/master/CVDD.m
# - License: MIT license (https://github.com/hulianyu/CVDD/blob/master/LICENSE)

# Paper: An Internal Validity Index Based on Density-Involved Distance
# Authors: Lianyu Hu and Caiming Zhong
# Link: https://ieeexplore.ieee.org/document/8672850

# Our modifications:
#    (1) translated from Matlab


import numpy as np
from scipy.spatial.distance import pdist, squareform

"""
%Aim: The matlab code of "An internal validity index based on density-involved distance"
%Algorithm 1: CVDD
% -------------------------------------------------------------------------
%Input:
%pi: the partition of X
%d: the Euclidean distance between objects in X
% -------------------------------------------------------------------------
%Output:
%results: the CVDD index of pi
% -------------------------------------------------------------------------
% Written by Lianyu Hu
% Department of Computer Science, Ningbo University 
% February 2019

https://github.com/hulianyu/CVDD/blob/master/CVDD.m
"""


def fast_PathbasedDist(d):
    return np.std(d, axis=1) / np.mean(d, axis=1)


def cvdd_score(X, piX, k=7):
    d = squareform(pdist(X, metric="minkowski", p=2))  # Euclidean distance of X
    try:
        DD = Density_involved_distance(d, k)  # Density-involved distance of X
        cvddindex = CVDD(piX, d, DD)
    except:
        cvddindex = 0
    return cvddindex


def CVDD(pi, d, DD):
    NC = len(np.unique(pi))
    sc_list = np.zeros(NC)  # separation
    com_list = np.zeros(NC)  # compactness
    for i in range(NC):
        a = np.where(pi == i)[0]
        b = np.where(pi != i)[0]
        n = len(a)
        if len(a) > 0:
            # compute the separation sep[i]
            sc_list[i] = np.min(np.min(DD[np.ix_(a, b)]))
            # compute the compactness com[i]
            try:
                Ci = fast_PathbasedDist(d[np.ix_(a, a)])
                com_list[i] = (np.std(Ci) / n) * np.mean(Ci)
            except:
                com_list[i] = np.max(com_list)
        else:
            sc_list[i] = 0
            com_list[i] = np.max(com_list)
    # compute the validity index CVDD
    sep = np.sum(sc_list)
    com = np.sum(com_list)
    CVDD = sep / com
    return CVDD


def Density_involved_distance(d, K):
    """
    Compute the Density-involved distance DD.

    Parameters:
    d (numpy.ndarray): The Euclidean distance between objects in X.
    K (int): The number of neighborhoods.

    Returns:
    numpy.ndarray: The DD of d.
    """
    N = len(d)
    KNNG = KNearestNeighborGraph(d, K)
    Den = np.zeros((N, 1))
    for j in range(N):
        Den[j, 0] = np.sum(d[j, KNNG[j, 0]]) / K
    fDen = Den / np.max(Den)  # absolute-density distance factor
    Rel = Den[:, None] / Den[None, :]
    tmp1 = Rel + Rel.T
    fRel = 1 - np.exp(-np.abs(tmp1 - 2))  # relative-density distance factor
    nD = Den[:, None] + Den[None, :]
    relD = nD * fRel
    drD = d + relD  # directly density-reachable distance
    conD = fast_PathbasedDist_(drD)  # connectivity distance
    tmp2 = np.sqrt(fDen[:, None] * fDen[None, :])
    DD = conD * tmp2  # density-involved distance
    return DD


def KNearestNeighborGraph(d, K):
    """
    Compute the K-Nearest Neighbor Graph.

    Parameters:
    d (numpy.ndarray): The Euclidean distance between objects in X.
    K (int): The number of neighborhoods.

    Returns:
    list: The K-Nearest Neighbor Graph.
    """
    N = len(d)
    KNNG = [[] for _ in range(N)]
    for i in range(N):
        dists = d[i, :]
        indices = np.argsort(dists)
        KNNG[i] = indices[:K]
    return KNNG


def fast_PathbasedDist_(drD):
    """
    Compute the connectivity distance.

    Parameters:
    drD (numpy.ndarray): The directly density-reachable distance.

    Returns:
    numpy.ndarray: The connectivity distance.
    """
    N = len(drD)
    conD = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                conD[i, j] = np.min(drD[i, j], np.min(drD[i, :] + drD[:, j] - drD[i, i]))
    return conD

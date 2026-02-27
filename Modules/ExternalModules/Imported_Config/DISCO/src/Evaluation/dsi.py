# Implementation of DSI by
# - Author: Shuyue Guan - Github user `ShuyueG`
# - Source: https://github.com/ShuyueG/CVI_using_DSI/blob/main/cluster_DSI_example.py
# - License: GPL-3.0 licence (https://github.com/ShuyueG/CVI_using_DSI/blob/main/LICENSE)

# Paper: An Internal Cluster Validity Index Using a Distance-based Separability Measure
# Authors: Shuyue Guan and Murray Loew
# Link: https://ieeexplore.ieee.org/document/9288314


import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import ks_2samp


def dists(data, dist_func=distance.euclidean):  # compute ICD
    num = data.shape[0]
    data = data.reshape((num, -1))
    dist = []
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            dist.append(dist_func(data[i], data[j]))
    return np.array(dist)


def dist_btw(a, b, dist_func=distance.euclidean):  # compute BCD
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    dist = []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            dist.append(dist_func(a[i], b[j]))
    return np.array(dist)


def dsi_score(data, labels):  # KS test on ICD and BCD
    classes = np.unique(labels)
    SUM = 0
    for c in classes:
        pos = data[np.squeeze(labels == c)]
        neg = data[np.squeeze(labels != c)]

        dist_pos = dists(pos)
        distbtw = dist_btw(pos, neg)
        D, _ = ks_2samp(dist_pos, distbtw)  # KS test
        SUM += D
    SUM = SUM / classes.shape[0]  # normed: b/c ks_2samp ranges [0,1]
    return SUM

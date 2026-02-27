# Implementation of CVNN by
# - Author: Jana Gauss - Github user `JanaGauss`
# - Source: https://github.com/JanaGauss/dcsi/blob/main/code/functions/separability_functions.R
# - License: -

# Paper: Understanding and Enhancement of Internal Clustering Validation Measures
# Authors: Yanchi Liu, Zhongmou Li, Hui Xiong, Xuedong Gao, Junjie Wu, and Sen Wu
# Link: https://ieeexplore.ieee.org/document/6341117

# Our modifications:
#    (1) translated from R to python


import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph


def cvnn_score(data, labels, k=10):
    """
    WE WANT THE ORIGINAL VERSION therefore we exclude the modifications (can be found in the comments)
    Function to calculate (a modified version of) CVNN

    Calculates CVNN
    (with some modifications, originally from Understanding and Enhancement of Internal Clustering Validation Measures, Liu et al.)
    Modifications:
    1. to calculate the overall Compactness, the mean is used instead of the sum of intra-cluster compactness-values
    2. Compactness is divided by the mean distance of points, so that CVNN can be compared between different data sets

    Args:
        dist (numpy.ndarray): a distance matrix
        labels (numpy.ndarray): a vector with labels
        k (int): number of nearest neighbors

    Returns:
        float: CVNN score
    """
    # what kind of distance measure?
    dist = euclidean_distances(data, data)
    dist = np.array(dist)

    # calculate separation and compactness for every class
    Sep_list = []
    Comp_list = []

    ### R-Code
    #knn_graph <- cccd::nng(dx = dist, k = k)
    #knn_matrix <- as.matrix(igraph::as_adjacency_matrix(knn_graph))
    knn_graph = kneighbors_graph(data, k, mode='distance')
    knn_matrix = knn_graph.toarray()

    for i in np.unique(labels):
        # add this to ensure that noise is not seen as a separate cluster
        if i != -1:
            ind_i = np.where(labels == i)[0]

            # Separation: proportion of k-NN of objects in i that don't belong to i
            knn_i = knn_matrix[ind_i, :][:, ind_i]
            sep_i = np.sum(knn_i) / (len(ind_i) * k)
            Sep_list.append(sep_i)

            # Compactness: average pairwise distance between objects in same cluster
            dist_i = dist[np.ix_(ind_i, ind_i)]
            comp_i = np.mean(dist_i[np.triu_indices(len(ind_i), 1)])
            Comp_list.append(comp_i)

    # calculate CVNN
    # Modification for compactness: mean instead of sum, normalize by mean distance so that CVNN of different data can be compared
    dist[np.diag_indices_from(dist)] = np.nan

    #Comp = np.mean(Comp_list) / np.nanmean(dist)
    Comp = np.sum(Comp_list)
    Sep = np.max(Sep_list)

    cvnn_value = Comp + Sep

    # return result
    return cvnn_value

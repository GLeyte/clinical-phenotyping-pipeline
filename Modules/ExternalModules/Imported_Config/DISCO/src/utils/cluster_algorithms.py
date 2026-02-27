import os
import sys

parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)


import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering, MeanShift, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from src.Clusterer.DPC import DensityPeakCluster
from src.Evaluation.dcdistances.dctree import DCTree


def optimal_k_dbscan(X, l):
    dctree = DCTree(X, min_points=5, min_points_mr=3)

    l_ = np.full(len(l), -1)
    best_ari = 0

    for k in range(2, 4 * len(set(l))):
        eps = dctree.get_eps_for_k(k)
        l_dbscan = DBSCAN(eps).fit(X).labels_
        l_kcenter = dctree.get_k_center(k)
        ari = ARI(l_dbscan, l_kcenter)
        if best_ari - 0.01 <= ari:
            l_ = l_dbscan
            best_ari = ari
    return l_


CLUSTER_ALGORITHMS = {
    "GroundTruth": lambda X, l: l,
    # "OptimalKDBSCAN": lambda X, l: optimal_k_dbscan(X, l),
    "DBSCAN": lambda X, l: DBSCAN(DCTree(X).get_eps_for_k(len(set(l)))).fit(X).labels_,
    # "KCenter": lambda X, l: DCTree(X).get_k_center(len(set(l))),
    "HDBSCAN": lambda X, l: HDBSCAN().fit(X).labels_,
    "DPC": lambda X, l: DensityPeakCluster().fit(X).labels_,
    "SpectralClustering": lambda X, l: SpectralClustering(len(set(l))).fit(X).labels_,
    "Agglomerative": lambda X, l: AgglomerativeClustering(len(set(l))).fit(X).labels_,
    "MeanShift": lambda X, l: MeanShift().fit(X).labels_,
    "KMeans": lambda X, l: KMeans(len(set(l))).fit(X).labels_,
    "Random_k": lambda X, l: np.random.choice(len(set(l)), size=len(l)),
    "Random_100": lambda X, l: np.random.choice(100, size=len(l)),
}

# SELECTED_CLUSTER_ALGORITHMS = [
#     "DPC",
# ]

SELECTED_CLUSTER_ALGORITHMS = CLUSTER_ALGORITHMS.keys()

CLUSTER_ABBREV = {
    "GroundTruth": "GT",
    "OptimalKDBSCAN": "K'-DBSCAN",
    "DBSCAN": "DBSCAN",
    "KCenter": "KCenter",
    "HDBSCAN": "HDBSCAN",
    "DPC": "DPC",
    "SpectralClustering": "SC",
    "Agglomerative": "Aggl.",
    "MeanShift": "MeanShift",
    "KMeans": "KMeans",
}

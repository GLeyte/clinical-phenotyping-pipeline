# Implementation of Dunn index with dc-distance by
# - Author: The Lightning team, us
# - Source: https://github.com/Lightning-AI/torchmetrics/blob/v1.4.0.post0/src/torchmetrics/functional/clustering/dunn_index.py
# - License: Apache License, Version 2.0 (https://github.com/Lightning-AI/torchmetrics/blob/v1.4.0.post0/LICENSE)

# Paper: Well-Separated Clusters and Optimal Fuzzy Partitions
# Authors: J.C. Dunn
# Link: https://www.tandfonline.com/doi/abs/10.1080/01969727408546059

# Paper: Connecting the Dots -- Density-Connectivity Distance unifies DBSCAN, k-Center and Spectral Clustering
# Authors: Anna Beer, Andrew Draganov, Ellen Hohma, Philipp Jahn, Christian M.M. Frey, and Ira Assent
# Link: https://doi.org/10.1145/3580305.3599283

# Our modifications:
#    (1) apply dunn on dc-distance


import numpy as np
import torch
from torch import Tensor
from src.Evaluation.dcdistances.dctree import DCTree


def _dunn_index_update(data, labels, min_points):
    """Update and return variables required to compute the Dunn index.

    Args:
        data: feature vectors of shape (n_samples, n_features)
        labels: cluster labels
        p: p-norm (distance metric)

    Returns:
        intercluster_distance: intercluster distances
        max_intracluster_distance: max intracluster distances

    """
    unique_labels, inverse_indices = labels.unique(return_inverse=True)
    clusters = [data[inverse_indices == label_idx] for label_idx in range(len(unique_labels))]
    # centroids = [c.mean(dim=0) for c in clusters]
    dc_distance = DCTree(data, min_points=min_points, no_fastindex=False).dc_distances()
    dc_distance_intra_cluster = [dc_distance[inverse_indices == label_idx,] for label_idx
                                 in range(len(unique_labels))]
    dc_distance_intra_cluster = [dc_intra[:, inverse_indices == label_idx] for dc_intra, label_idx
                                 in zip(dc_distance_intra_cluster, range(len(unique_labels)))]
    dc_distance_inter_cluster = [dc_distance[inverse_indices == label_idx, :] for label_idx
                                 in range(len(unique_labels))]
    dc_distance_inter_cluster = [dc_inter[:, inverse_indices != label_idx] for dc_inter, label_idx
                                 in zip(dc_distance_inter_cluster, range(len(unique_labels)))]
    # min distance between cluster i and all the others
    intercluster_distance = torch.from_numpy(np.array([distances.min() for distances in dc_distance_inter_cluster]))

    # maximum distance inside every cluster
    max_intracluster_distance = torch.from_numpy(np.array([
        distances.max() for distances in dc_distance_intra_cluster
    ]))
    return intercluster_distance, max_intracluster_distance


def _dunn_index_compute(intercluster_distance: Tensor, max_intracluster_distance: Tensor) -> Tensor:
    """Compute the Dunn index based on updated state.

    Args:
        intercluster_distance: intercluster distances
        max_intracluster_distance: max intracluster distances

    Returns:
        scalar tensor with the dunn index

    """
    return intercluster_distance.min() / max_intracluster_distance.max()


def dc_dunn_score(data, labels, min_points=5):
    """Compute the Dunn index.

    Args:
        data: feature vectors
        labels: cluster labels
        p: p-norm used for distance metric

    Returns:
        scalar tensor with the dunn index

    Example:
        >>> from torchmetrics.functional.clustering import dunn_index
        >>> data = torch.tensor([[0, 0], [0.5, 0], [1, 0], [0.5, 1]])
        >>> labels = torch.tensor([0, 0, 0, 1])
        >>> dunn_index(data, labels)
        tensor(2.)

    """
    data = np.array(data, dtype=np.float64)
    labels = np.array(labels, dtype=int)

    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    pairwise_distance, max_distance = _dunn_index_update(data, labels, min_points)
    result = _dunn_index_compute(pairwise_distance, max_distance)
    return result.numpy().item()

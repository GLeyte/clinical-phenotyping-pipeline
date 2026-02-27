# Implementation of Dunn index by
# - Author: The Lightning team
# - Source: https://github.com/Lightning-AI/torchmetrics/blob/v1.4.0.post0/src/torchmetrics/functional/clustering/dunn_index.py
# - License: Apache License, Version 2.0 (https://github.com/Lightning-AI/torchmetrics/blob/v1.4.0.post0/LICENSE)

# Paper: Well-Separated Clusters and Optimal Fuzzy Partitions
# Authors: J.C. Dunn
# Link: https://www.tandfonline.com/doi/abs/10.1080/01969727408546059


import numpy as np
import torch
from torch import Tensor
from itertools import combinations


def _dunn_index_update(data, labels, p: float):
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
    centroids = [c.mean(dim=0) for c in clusters]

    intercluster_distance = torch.linalg.norm(
        torch.stack([a - b for a, b in combinations(centroids, 2)], dim=0), ord=p, dim=1
    )

    max_intracluster_distance = torch.stack([
        torch.linalg.norm(ci - mu, ord=p, dim=1).max() for ci, mu in zip(clusters, centroids)
    ])

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


def dunn_score(data, labels, p: float = 2):
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
    pairwise_distance, max_distance = _dunn_index_update(data, labels, p)
    result = _dunn_index_compute(pairwise_distance, max_distance)
    return result.numpy().item()

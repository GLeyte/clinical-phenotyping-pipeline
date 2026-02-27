import numpy as np
from . import base_measure
from sklearn.metrics import silhouette_score, silhouette_samples

def function_clusters_swc(data: np.ndarray, labels: np.ndarray, **kwargs):
    sample_sils = silhouette_samples(data, labels, **kwargs)
    ret = {}
    for cluster in np.unique(labels):
        ret[cluster] = np.mean(sample_sils[labels == cluster])
    return ret

class Silhouette_Coefficient(base_measure.BaseMeasure):
    """
    Rousseeuw, Peter J.
    "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis."
    Journal of computational and applied mathematics 20 (1987): 53-65.
    """
    def __init__(self):
        super().__init__()
        self.name = "SWC"
        self.worst_value = -1
        self.best_value = 1
        self.function = silhouette_score
        self.function_norm = silhouette_score
        self.kwargs = {"metric": "precomputed"}
        self.needs_quadratic = True
        self.function_clusters = function_clusters_swc
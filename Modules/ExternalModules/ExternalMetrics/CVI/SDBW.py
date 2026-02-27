import numpy as np
from . import base_measure
from s_dbw import S_Dbw

class SDBW(base_measure.BaseMeasure):
    """
    Halkidi, Maria, and Michalis Vazirgiannis.
    "Clustering validity assessment: Finding the optimal partitioning of a data set."
    Proceedings 2001 IEEE international conference on data mining. IEEE, 2001
    """
    def __init__(self):
        super().__init__()
        self.name = "S-Dbw"
        self.worst_value = np.inf
        self.best_value = 0
        self.normalization_params = (0.380854, 0.180388)
        self.function_norm = ValueError
        self.function = S_Dbw
        self.kwargs = {}
        self.needs_quadratic = False
        self.less_is_better = True

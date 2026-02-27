from .CVNN import CVNN, CVNN_halkidi
from .DSI import DSI
from .VIASCKDE import VIASCKDE
from .Silhouette_Coefficient import Silhouette_Coefficient
from .SDBW import SDBW

registered_measures = [Silhouette_Coefficient, DSI, VIASCKDE, CVNN_halkidi, SDBW, CVNN]


def get_measures():
    return registered_measures


def get_measures_dict():
    return {measure().name: measure() for measure in registered_measures}

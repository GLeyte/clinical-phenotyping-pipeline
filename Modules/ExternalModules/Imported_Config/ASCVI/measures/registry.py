from .CDR_index import CDR_Index
from .CVDD import CVDD
from .CVNN import CVNN, CVNN_halkidi
from .DCVI import DCV_Index
from .DSI import DSI
from .ICAV import IC_av
from .VIASCKDE import VIASCKDE
from .dbcv_measures import DBCV
from .standard_measures import Silhouette_Coefficient, VRC, SDBW

registered_measures = [Silhouette_Coefficient, DBCV, DSI, CDR_Index, VIASCKDE, VRC, CVNN_halkidi, SDBW, CVDD, DCV_Index,
                       IC_av, CVNN]


def get_measures():
    return registered_measures


def get_measures_dict():
    return {measure().name: measure() for measure in registered_measures}

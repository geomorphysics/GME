import numpy as np
from gme.core.equations_extended import EquationsGeodesic
from gme.ode.time_invariant import TimeInvariantSolution
from gme.plot.base import Graphing
from typing import Dict, List, Optional, Tuple

class RayGeodesics(Graphing):
    x_array: np.ndarray
    t_array: np.ndarray
    rz_array: np.ndarray
    vx_array: np.ndarray
    vz_array: np.ndarray
    gstar_matrices_list: List
    gstar_matrices_array: List
    g_matrices_list: List
    g_matrices_array: List
    def __init__(self, gmes: TimeInvariantSolution, gmeq: EquationsGeodesic, n_points: int, do_recompute: bool = ...) -> None: ...
    def profile_g_properties(self, gmes: TimeInvariantSolution, gmeq: EquationsGeodesic, sub: Dict, name: str, fig_size: Optional[Tuple[float, float]] = ..., dpi: Optional[int] = ..., y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ..., do_gstar: bool = ..., do_det: bool = ..., do_eigenvectors: bool = ..., eta_label_xy: Optional[Tuple[float, float]] = ..., do_etaxi_label: bool = ..., legend_loc: str = ..., do_pv: bool = ...) -> None: ...

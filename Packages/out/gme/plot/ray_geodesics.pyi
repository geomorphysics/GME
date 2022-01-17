from gme.core.equations_extended import EquationsGeodesic
from gme.ode.time_invariant import TimeInvariantSolution
from gme.plot.base import Graphing
from typing import Any, Dict, Optional, Tuple

class RayGeodesics(Graphing):
    x_array: Any
    t_array: Any
    rz_array: Any
    vx_array: Any
    vz_array: Any
    gstar_matrices_list: Any
    gstar_matrices_array: Any
    g_matrices_list: Any
    g_matrices_array: Any
    def __init__(self, gmes: TimeInvariantSolution, gmeq: EquationsGeodesic, n_points: int, do_recompute: bool = ...) -> None: ...
    def profile_g_properties(self, gmes: TimeInvariantSolution, gmeq: EquationsGeodesic, sub: Dict, name: str, fig_size: Optional[Tuple[float, float]] = ..., dpi: Optional[int] = ..., y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ..., do_gstar: bool = ..., do_det: bool = ..., do_eigenvectors: bool = ..., eta_label_xy: Optional[Tuple[float, float]] = ..., do_etaxi_label: bool = ..., legend_loc: str = ..., do_pv: bool = ...) -> None: ...

import abc
import numpy as np
from abc import ABC, abstractmethod
from gme.core.equations import Equations
from typing import Any, Callable, Dict, List, Optional, Tuple

class BaseSolution(ABC, metaclass=abc.ABCMeta):
    gmeq: Any
    parameters: Any
    choice: Any
    method: Any
    do_dense: Any
    x_stop: Any
    tp_xiv0_list: Any
    n_rays: Any
    t_end: Any
    t_distribn: Any
    t_slip_end: Any
    n_t: Any
    t_ensemble_max: Any
    interp1d_kind: str
    pz_velocity_boundary_eqn: Any
    model_dXdt_lambda: Any
    customize_t_fn: Any
    ic_list: Any
    ref_t_array: Any
    rpt_arrays: Any
    ivp_solns_list: Any
    rp_t_interp_fns: Any
    t_isochrone_max: float
    n_isochrones: int
    x_subset: int
    tolerance: float
    rpt_isochrones: Any
    trxz_cusps: Any
    cusps: Any
    cx_pz_tanbeta_lambda: Any
    cx_pz_lambda: Any
    cx_v_lambda: Any
    vx_interp_fast: Any
    vx_interp_slow: Any
    def __init__(self, gmeq: Equations, parameters: Dict, choice: str = ..., method: str = ..., do_dense: bool = ..., x_stop: float = ..., t_end: float = ..., t_slip_end: float = ..., t_distribn: float = ..., n_rays: int = ..., n_t: int = ..., tp_xiv0_list: Optional[List[Tuple[float, float]]] = ..., customize_t_fn: Optional[Callable] = ...): ...
    @abstractmethod
    def initial_conditions(self) -> Tuple[float, float, float, float]: ...
    @abstractmethod
    def solve(self) -> None: ...
    def make_model(self) -> Callable[[float, Tuple[Any, Any, Any, Any]], np.ndarray]: ...
    def postprocessing(self, spline_order: int = ..., extrapolation_mode: int = ...) -> None: ...
    def resolve_isochrones(self, x_subset: int = ..., t_isochrone_max: float = ..., n_isochrones: int = ..., n_resample_pts: int = ..., tolerance: float = ..., do_eliminate_caustics: bool = ..., dont_crop_cusps: bool = ...) -> None: ...
    def measure_cusp_propagation(self) -> None: ...
    def save(self, rpt_arrays: Dict, idx: int) -> None: ...

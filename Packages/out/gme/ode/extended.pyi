import abc
import numpy as np
from gme.core.equations import Equations
from gme.core.equations_extended import EquationsGeodesic, EquationsIbc, EquationsIdtx
from gme.ode.base import BaseSolution
from typing import Any, Callable, Dict, List, Union

class ExtendedSolution(BaseSolution, metaclass=abc.ABCMeta):
    pz0: float
    rays: List
    t_array: np.ndarray
    rx_array: np.ndarray
    rz_array: np.ndarray
    p_array: np.ndarray
    px_array: np.ndarray
    pz_array: np.ndarray
    rdot_array: np.ndarray
    rdotx_array: np.ndarray
    rdotz_array: np.ndarray
    pdot_array: np.ndarray
    pdotx_array: np.ndarray
    pdotz_array: np.ndarray
    tanalpha_array: np.ndarray
    tanbeta_array: np.ndarray
    alpha_array: np.ndarray
    beta_array: np.ndarray
    xiv_p_array: np.ndarray
    xiv_v_array: np.ndarray
    uhorizontal_p_array: np.ndarray
    uhorizontal_v_array: np.ndarray
    cosbeta_array: np.ndarray
    sinbeta_array: np.ndarray
    u_array: np.ndarray
    x_array: np.ndarray
    h_array: np.ndarray
    h_x_array: np.ndarray
    h_z_array: np.ndarray
    dhdx_array: np.ndarray
    beta_vt_array: np.ndarray
    beta_ts_array: np.ndarray
    t_interp_x: Callable
    rz_interp: Callable
    rx_interp_t: Callable
    rz_interp_t: Callable
    x_interp_t: Callable
    p_interp: Callable
    px_interp: Callable
    pz_interp: Callable
    rdot_interp: Callable
    rdotx_interp: Callable
    rdotz_interp: Callable
    pdot_interp: Callable
    pdot_interp_t: Callable
    rdotx_interp_t: Any
    rdotz_interp_t: Any
    rddotx_interp_t: Callable
    rddotz_interp_t: Callable
    beta_p_interp: Callable
    beta_ts_interp: Callable
    beta_ts_error_interp: Callable
    beta_vt_interp: Callable
    beta_vt_error_interp: Callable
    u_interp: Callable
    uhorizontal_p_interp: Callable
    uhorizontal_v_interp: Callable
    u_from_rdot_interp: Callable
    xiv_v_interp: Callable
    xiv_p_interp: Callable
    alpha_interp: Callable
    h_interp: Callable
    def __init__(self, gmeq: Union[Equations, EquationsGeodesic, EquationsIdtx, EquationsIbc], parameters: Dict, **kwargs) -> None: ...

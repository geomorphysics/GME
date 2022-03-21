import numpy as np
from gme.ode.extended import ExtendedSolution
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Any, List, Tuple

class SingleRaySolution(ExtendedSolution):
    ic_list: List[Tuple[float, float, float, float]]
    model_dXdt_lambda: Any
    ivp_solns_list: List
    pz0: Any
    p_array: np.ndarray
    rdot_array: np.ndarray
    pdot_array: np.ndarray
    t_interp_x: InterpolatedUnivariateSpline
    rx_interp_t: InterpolatedUnivariateSpline
    rz_interp_t: InterpolatedUnivariateSpline
    x_interp_t: InterpolatedUnivariateSpline
    rz_interp: InterpolatedUnivariateSpline
    p_interp: InterpolatedUnivariateSpline
    rdot_interp: InterpolatedUnivariateSpline
    rdotx_interp: InterpolatedUnivariateSpline
    rdotz_interp: InterpolatedUnivariateSpline
    pdot_interp: InterpolatedUnivariateSpline
    px_interp: InterpolatedUnivariateSpline
    pz_interp: InterpolatedUnivariateSpline
    rdotx_interp_t: InterpolatedUnivariateSpline
    rdotz_interp_t: InterpolatedUnivariateSpline
    rddotx_interp_t: InterpolatedUnivariateSpline
    rddotz_interp_t: InterpolatedUnivariateSpline
    tanbeta_array: np.ndarray
    tanalpha_array: np.ndarray
    beta_array: np.ndarray
    beta_p_interp: InterpolatedUnivariateSpline
    alpha_array: np.ndarray
    alpha_interp: InterpolatedUnivariateSpline
    cosbeta_array: np.ndarray
    sinbeta_array: np.ndarray
    u_array: np.ndarray
    xiv_p_array: np.ndarray
    xiv_v_array: np.ndarray
    uhorizontal_p_array: np.ndarray
    uhorizontal_v_array: np.ndarray
    u_interp: InterpolatedUnivariateSpline
    u_from_rdot_interp: InterpolatedUnivariateSpline
    xiv_p_interp: InterpolatedUnivariateSpline
    xiv_v_interp: InterpolatedUnivariateSpline
    uhorizontal_p_interp: InterpolatedUnivariateSpline
    uhorizontal_v_interp: InterpolatedUnivariateSpline
    def initial_conditions(
        self, t_lag: float = ..., xiv_0_: float = ...
    ) -> Tuple[float, float, float, float]: ...
    def solve(self) -> None: ...
    def postprocessing(
        self, spline_order: int = ..., extrapolation_mode: int = ...
    ) -> None: ...

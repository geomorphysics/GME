import numpy as np
from gme.ode.single_ray import SingleRaySolution
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import List

class TimeInvariantSolution(SingleRaySolution):
    beta_vt_array: np.ndarray
    beta_vt_interp: InterpolatedUnivariateSpline
    beta_vt_error_interp: InterpolatedUnivariateSpline
    rays: List
    h_interp: InterpolatedUnivariateSpline
    beta_ts_interp: InterpolatedUnivariateSpline
    dhdx_array: np.ndarray
    beta_ts_array: np.ndarray
    beta_ts_error_interp: InterpolatedUnivariateSpline
    h_x_array: np.ndarray
    h_z_array: np.ndarray
    def postprocessing(
        self, spline_order: int = ..., extrapolation_mode: int = ...
    ) -> None: ...
    def integrate_h_profile(
        self,
        n_pts: int = ...,
        x_max: float = ...,
        do_truncate: bool = ...,
        do_use_newton: bool = ...,
    ) -> None: ...

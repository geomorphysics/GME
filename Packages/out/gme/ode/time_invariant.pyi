from gme.ode.single_ray import SingleRaySolution
from typing import Any

class TimeInvariantSolution(SingleRaySolution):
    beta_vt_array: Any
    beta_vt_interp: Any
    beta_vt_error_interp: Any
    rays: Any
    h_interp: Any
    beta_ts_interp: Any
    dhdx_array: Any
    beta_ts_array: Any
    beta_ts_error_interp: Any
    def postprocessing(self, spline_order: int = ..., extrapolation_mode: int = ...) -> None: ...
    h_x_array: Any
    h_z_array: Any
    def integrate_h_profile(self, n_pts: int = ..., x_max: float = ..., do_truncate: bool = ..., do_use_newton: bool = ...) -> None: ...

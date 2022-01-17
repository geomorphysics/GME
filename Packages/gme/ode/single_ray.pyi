from gme.ode.extended import ExtendedSolution
from typing import Any, Tuple

class SingleRaySolution(ExtendedSolution):
    def initial_conditions(self) -> Tuple[float, float, float, float]: ...
    ic_list: Any
    model_dXdt_lambda: Any
    ivp_solns_list: Any
    def solve(self) -> None: ...
    pz0: Any
    p_array: Any
    rdot_array: Any
    pdot_array: Any
    t_interp_x: Any
    rx_interp_t: Any
    rz_interp_t: Any
    x_interp_t: Any
    rz_interp: Any
    p_interp: Any
    rdot_interp: Any
    rdotx_interp: Any
    rdotz_interp: Any
    pdot_interp: Any
    px_interp: Any
    pz_interp: Any
    rdotx_interp_t: Any
    rdotz_interp_t: Any
    rddotx_interp_t: Any
    rddotz_interp_t: Any
    tanbeta_array: Any
    tanalpha_array: Any
    beta_array: Any
    beta_p_interp: Any
    alpha_array: Any
    alpha_interp: Any
    cosbeta_array: Any
    sinbeta_array: Any
    u_array: Any
    xiv_p_array: Any
    xiv_v_array: Any
    uhorizontal_p_array: Any
    uhorizontal_v_array: Any
    u_interp: Any
    u_from_rdot_interp: Any
    xiv_p_interp: Any
    xiv_v_interp: Any
    uhorizontal_p_interp: Any
    uhorizontal_v_interp: Any
    def postprocessing(self, spline_order: int = ..., extrapolation_mode: int = ...) -> None: ...

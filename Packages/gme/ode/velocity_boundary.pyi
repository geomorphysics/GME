from gme.ode.extended import ExtendedSolution
from typing import Any, Tuple

class VelocityBoundarySolution(ExtendedSolution):
    def initial_conditions(self, t_lag: float, xiv_0_: float) -> Tuple[float, float, float, float]: ...
    t_ensemble_max: float
    tp_xiv0_list: Any
    n_rays: Any
    ic_list: Any
    ivp_solns_list: Any
    def solve(self, report_pc_step: int = ...) -> None: ...

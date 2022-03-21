from gme.ode.extended import ExtendedSolution
from typing import List, Tuple

class VelocityBoundarySolution(ExtendedSolution):
    tp_xiv0_list: List
    n_rays: int
    ic_list: List
    ivp_solns_list: List
    def initial_conditions(
        self, t_lag: float = ..., xiv_0_: float = ...
    ) -> Tuple[float, float, float, float]: ...
    t_ensemble_max: float
    def solve(self, report_pc_step: int = ...) -> None: ...

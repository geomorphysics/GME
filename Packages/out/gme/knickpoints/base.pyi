import abc
from gme.core.equations import Equations
from gme.ode.base import BaseSolution
from typing import Any, Callable, Dict, Optional, Tuple

class InitialProfileSolution(BaseSolution):
    t_ensemble_max: float
    model_dXdt_lambda: Optional[Callable]
    rpt_arrays: Optional[Dict]
    rz_initial_surface_eqn: Any
    def __init__(self, gmeq: Equations, parameters: Dict, **kwargs) -> None: ...
    def initial_conditions(self, x_) -> Tuple[float, float, float, float]: ...
    ic: Any
    def solve(self, report_pc_step: int = ...) -> None: ...

class InitialCornerSolution(BaseSolution):
    t_ensemble_max: float
    model_dXdt_lambda: Optional[Callable]
    rpt_arrays: Optional[Dict]
    px_initial_corner_eqn: Any
    pz_initial_corner_eqn: Any
    beta_surface_corner: Any
    beta_velocity_corner: Any
    rdot: Any
    def __init__(self, gmeq: Equations, parameters: Dict, **kwargs) -> None: ...
    def initial_conditions(self, beta0_) -> Tuple: ...
    ic: Any
    def solve(self, report_pc_step: int = ..., verbose: bool = ...) -> None: ...

class CompositeSolution(BaseSolution, metaclass=abc.ABCMeta):
    t_ensemble_max: float
    model_dXdt_lambda: Optional[Callable]
    rpt_arrays: Optional[Dict]
    def __init__(self, gmeq: Equations, parameters: Dict, **kwargs) -> None: ...
    do_solns: Any
    t_end: Any
    t_slip_end: Any
    ips: Any
    ics: Any
    vbs: Any
    def create_solutions(self, t_end: float = ..., t_slip_end: float = ..., do_solns=..., n_rays=..., n_t=...) -> None: ...
    def solve(self) -> None: ...
    n_rays: Any
    def merge_rays(self) -> None: ...

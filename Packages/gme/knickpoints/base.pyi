import abc
from gme.core.equations import Equations
from gme.ode.base import BaseSolution
from sympy import Eq, Matrix
from typing import Any, Callable, Dict, Optional, Tuple

class InitialProfileSolution(BaseSolution):
    t_ensemble_max: float
    model_dXdt_lambda: Optional[Callable]
    rpt_arrays: Optional[Dict]
    rz_initial_surface_eqn: Eq
    ic: Tuple[float, float, float, float]
    def __init__(self, gmeq: Equations, parameters: Dict, **kwargs) -> None: ...
    def initial_conditions(self, x_: float) -> Tuple[float, float, float, float]: ...
    def solve(self, report_pc_step: int = ...) -> None: ...

class InitialCornerSolution(BaseSolution):
    t_ensemble_max: float
    model_dXdt_lambda: Optional[Callable]
    rpt_arrays: Optional[Dict]
    px_initial_corner_eqn: Eq
    pz_initial_corner_eqn: Eq
    beta_surface_corner: Eq
    beta_velocity_corner: Eq
    rdot: Matrix
    ic: Tuple[float, float, float, float]
    def __init__(self, gmeq: Equations, parameters: Dict, **kwargs) -> None: ...
    def initial_conditions(self, beta0_) -> Tuple[float, float, float, float]: ...
    def solve(self, report_pc_step: int = ..., verbose: bool = ...) -> None: ...

class CompositeSolution(BaseSolution, metaclass=abc.ABCMeta):
    t_ensemble_max: float
    model_dXdt_lambda: Optional[Callable]
    rpt_arrays: Optional[Dict]
    do_solns: bool
    t_end: float
    t_slip_end: float
    ips: Any
    ics: Any
    vbs: Any
    n_rays: int
    def __init__(self, gmeq: Equations, parameters: Dict, **kwargs) -> None: ...
    def create_solutions(self, t_end: float = ..., t_slip_end: float = ..., do_solns: Dict = ..., n_rays: Dict = ..., n_t: Dict = ...) -> None: ...
    def solve(self) -> None: ...
    def merge_rays(self) -> None: ...

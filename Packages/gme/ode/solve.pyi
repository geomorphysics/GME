import numpy as np
from typing import Any, Callable, Dict, Tuple

def solve_ODE_system(
    model: Callable,
    method: str,
    do_dense: bool,
    ic: Tuple[float, float, float, float],
    t_array: np.ndarray,
    x_stop: float = ...,
) -> Any: ...
def solve_Hamiltons_equations(
    model: Callable,
    method: str,
    do_dense: bool,
    ic: Tuple[float, float, float, float],
    parameters: Dict,
    t_array: np.ndarray,
    x_stop: float = ...,
    t_lag: float = ...,
) -> Tuple[Any, Dict[str, np.ndarray]]: ...

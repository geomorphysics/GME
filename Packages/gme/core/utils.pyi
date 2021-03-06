from sympy import Eq, Poly, Rational, Symbol
from typing import Any, Dict, Optional, Tuple

def pxpz0_from_xiv0(
    parameters: Dict[str, Any], pz0_xiv0_eqn: Eq, poly_px_xiv0_eqn: Eq
) -> Tuple[Any, Any]: ...
def gradient_value(
    x_: float, pz_: float, px_poly_eqn: Eq, do_use_newton: bool = ...
) -> float: ...
def px_value_search(
    x_: float,
    pz_: float,
    px_poly_eqn: Eq,
    method: str = ...,
    px_guess: float = ...,
    px_var_: Symbol = ...,
    pz_var_: Symbol = ...,
    bracket: Tuple[float, float] = ...,
) -> float: ...
def px_value(
    x_: float,
    pz_: float,
    px_poly_eqn: Eq,
    px_var_: Symbol = ...,
    pz_var_: Symbol = ...,
) -> float: ...
def find_dzdx_poly_root(
    dzdx_poly_: Poly,
    xhat_: float,
    xivhat0_: float,
    guess: float = ...,
    eta_: Optional[Rational] = ...,
    method: str = ...,
    bracket: Tuple[float, float] = ...,
) -> Any: ...
def make_dzdx_poly(dzdx_Ci_polylike_eqn_: Eq, sub_: Dict) -> Poly: ...

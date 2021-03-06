"""
Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`SciPy <scipy>`
  -  :mod:`SymPy <sympy>`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html

---------------------------------------------------------------------
"""

# Library
import warnings
from typing import Tuple, Dict, Any, List, Callable, Optional

# SciPy
from scipy.optimize import root_scalar

# SymPy
from sympy import (
    Eq,
    solve,
    simplify,
    sqrt,
    re,
    im,
    lambdify,
    diff,
    poly,
    nroots,
    Abs,
    Symbol,
    Poly,
    N,
    Rational,
)

# GME
from gme.core.symbols import (
    eta,
    Ci,
    xiv,
    xiv_0,
    xhat,
    dzdx,
    xivhat_0,
)

# GME
from gme.core.symbols import x, rx, px, pz, xiv, xiv_0

warnings.filterwarnings("ignore")

__all__ = [
    "pxpz0_from_xiv0",
    "gradient_value",
    "px_value_search",
    "px_value",
    "find_dzdx_poly_root",
    "make_dzdx_poly",
]


def pxpz0_from_xiv0(
    parameters: Dict[str, Any],
    # xiv_0_, xih_0_,
    pz0_xiv0_eqn: Eq,
    poly_px_xiv0_eqn: Eq,
) -> Tuple[Any, Any]:
    r"""Get initial slowness covector."""
    # pz0_xiv0_eqn = pz_xiv_eqn.subs({xiv:xiv_0}).subs(parameters)
    px0_poly_rx0_eqn = simplify(poly_px_xiv0_eqn.subs({rx: 0}))
    # .subs(parameters))
    # eta_ = eta.subs(parameters)
    # if True: #eta==Rational(1,2) or eta==Rational(3,2):
    px0sqrd_solns = solve(px0_poly_rx0_eqn, px ** 2)
    px0_: Any = sqrt(
        [
            px0sqrd_
            for px0sqrd_ in px0sqrd_solns
            if re(px0sqrd_) > 0 and im(px0sqrd_) == 0
        ][0]
    )
    # else:
    #     px0_poly_lambda = lambdify( [px], px0_poly_rx0_eqn.lhs.as_expr() )
    #     dpx0_poly_lambda \
    #  = lambdify( [px], diff(px0_poly_rx0_eqn.lhs.as_expr(),px) )
    #     px0_root_search = None
    #     for px_guess_ in [px_guess]:
    #         px0_root_search = root_scalar( px0_poly_lambda,
    # fprime=dpx0_poly_lambda,
    #                 method='newton', x0=px_guess_ )
    #         if px0_root_search.converged:
    #             break
    #     px0_ = px0_root_search.root

    pz0_: Any = pz0_xiv0_eqn.rhs.subs({xiv: xiv_0}).subs(parameters)
    return (px0_, pz0_)


def gradient_value(
    x_: float, pz_: float, px_poly_eqn: Eq, do_use_newton: bool = False
) -> float:
    """Compute surface gradient from slowness covector components."""
    px_: float = (
        -px_value_search(x_, pz_, px_poly_eqn)
        if do_use_newton
        else -px_value(x_, pz_, px_poly_eqn)
    )
    return float(px_ / pz_)


def px_value_search(
    x_: float,
    pz_: float,
    px_poly_eqn: Eq,
    method: str = "newton",
    px_guess: float = 0.01,
    px_var_: Symbol = px,
    pz_var_: Symbol = pz,
    bracket: Tuple[float, float] = (0, 30),
) -> float:
    r"""Find :math:`p_x` root by nonlinear search."""
    px_poly_eqn_: Eq = px_poly_eqn.subs({rx: x_, x: x_, pz_var_: pz_})
    px_poly_lambda: Callable = lambdify([px_var_], px_poly_eqn_.as_expr())
    dpx_poly_lambda: Callable = (
        lambdify([px_var_], diff(px_poly_eqn_.as_expr(), px_var_))
        if method == "newton"
        else None
    )
    bracket_: Optional[Tuple[float, float]] = (
        bracket if method == "brentq" else None
    )
    for px_guess_ in [1, px_guess]:
        px_root_search = root_scalar(
            px_poly_lambda,
            fprime=dpx_poly_lambda,
            bracket=bracket_,
            method=method,
            x0=px_guess_,
        )
        if px_root_search.converged:
            break
    # px_root_search = root_scalar( px_poly_lambda, fprime=dpx_poly_lambda,
    #     method='newton', x0=px_guess )
    px_: float = px_root_search.root
    return px_


def px_value(
    x_: float,
    pz_: float,
    px_poly_eqn: Eq,
    px_var_: Symbol = px,
    pz_var_: Symbol = pz,
) -> float:
    """
    TODO.

    Args:
        TODO
    """
    px_poly_eqn_: Poly = poly(px_poly_eqn.subs({rx: x_, x: x_, pz_var_: pz_}))
    px_poly_roots: List[float] = nroots(px_poly_eqn_)
    pxgen: float = [
        root_
        for root_ in px_poly_roots
        if Abs(im(root_)) < 1e-10 and re(root_) > 0
    ][0]
    return solve(Eq(px_poly_eqn_.gens[0], pxgen), px_var_)[0]


def find_dzdx_poly_root(
    dzdx_poly_: Poly,
    xhat_: float,
    xivhat0_: float,
    guess: float = 0.1,
    eta_: Optional[Rational] = None,
    method: str = "brentq",
    bracket: Tuple[float, float] = (0, 1e3),
) -> Any:
    """
    TODO.

    Args:
        TODO
    """
    if eta_ is not None and eta_ == Rational(1, 4):
        poly_eqn_ = dzdx_poly_.subs({xhat: xhat_, xivhat_0: xivhat0_})
        poly_lambda: Callable = lambdify([dzdx], poly_eqn_.as_expr())
        # return poly_eqn_
        dpoly_lambda: Optional[Callable] = lambdify(
            [dzdx],
            diff(poly_eqn_.as_expr(), dzdx) if method == "newton" else None,
        )
        bracket_: Optional[Tuple[float, float]] = (
            bracket if method == "brentq" else None
        )
        for guess_ in [0, guess]:
            root_search = root_scalar(
                poly_lambda,
                fprime=dpoly_lambda,
                method=method,
                bracket=bracket_,
                x0=guess_,
            )
            if root_search.converged:
                break
        dzdx_poly_root: float = root_search.root
    else:
        dzdx_poly_roots = nroots(
            dzdx_poly_.subs({xhat: xhat_, xivhat_0: xivhat0_})
        )
        dzdx_poly_root = [
            root_
            for root_ in dzdx_poly_roots
            if Abs(im(root_)) < 1e-10 and re(root_) > 0
        ][0]
    return dzdx_poly_root


def make_dzdx_poly(
    dzdx_Ci_polylike_eqn_: Eq,
    sub_: Dict,
) -> Poly:
    """
    TODO.

    Args:
        TODO
    """
    eta_ = eta.subs(sub_)
    # Bit of a hack - assumes the dzdx ODE has only two "arg" terms
    #  for $\eta=1/4$
    if eta_ == Rational(1, 4):
        dzdx_eqn_ = dzdx_Ci_polylike_eqn_.subs({eta: Rational(1, 4)})
        dzdx_eqn_ = Eq(
            dzdx_eqn_.lhs.args[0] ** 2 - dzdx_eqn_.lhs.args[1] ** 2, 0
        )
    else:
        dzdx_eqn_ = dzdx_Ci_polylike_eqn_
    return poly(N(dzdx_eqn_.lhs.subs(sub_)), dzdx)

    #

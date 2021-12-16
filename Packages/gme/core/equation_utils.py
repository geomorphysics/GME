"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

This module provides a derivation of GME theory using :mod:`SymPy <sympy>`
for a 2D slice of a 3D landscape along a channel transect.

Starting from a model equation for the surface-normal erosion rate in terms of
the tilt angle of the topographic surface and the distance downstream
(used to infer the flow component of the model erosion process), we derive
the fundamental function (of a co-Finsler metric space), the corresponding
Hamiltonian,and the equivalent metric tensor.
The rest of the equations are derived from these core results.

---------------------------------------------------------------------


Requires Python packages/modules:
  -  :mod:`scipy`, :mod:`sympy`
  -  :mod:`gmplib.utils`
  -  :mod:`gme.core.symbols`

---------------------------------------------------------------------

"""
import warnings

# Typing
from typing import Tuple, Dict, Any, List

# SciPy
from scipy.optimize import root_scalar

# SymPy
from sympy import Eq, solve, simplify, sqrt, re, im, lambdify, diff, \
                  poly, nroots, Abs

# GME
from gme.core.symbols import x, rx, px, pz, xiv, xiv_0

warnings.filterwarnings("ignore")

__all__ = ['pxpz0_from_xiv0', 'gradient_value', 'px_value_search', 'px_value']


def pxpz0_from_xiv0(parameters: Dict[str, Any],
                    # xiv_0_, xih_0_,
                    pz0_xiv0_eqn, poly_px_xiv0_eqn) -> Tuple[float, float]:
    """
    TBD
    """
    # pz0_xiv0_eqn = pz_xiv_eqn.subs({xiv:xiv_0}).subs(parameters)
    px0_poly_rx0_eqn = simplify(poly_px_xiv0_eqn.subs({rx: 0}))
    # .subs(parameters))
    # eta_ = eta.subs(parameters)
    # if True: #eta==Rational(1,2) or eta==Rational(3,2):
    px0sqrd_solns = solve(px0_poly_rx0_eqn, px**2)
    px0_: float = sqrt([px0sqrd_ for px0sqrd_ in px0sqrd_solns
                        if re(px0sqrd_) > 0 and im(px0sqrd_) == 0][0])
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

    pz0_: float = pz0_xiv0_eqn.rhs.subs({xiv: xiv_0}).subs(parameters)
    return (px0_, pz0_)


def gradient_value(x_, pz_, px_poly_eqn, do_use_newton=False) -> float:
    """
    TBD
    """
    px_: float = -px_value_search(x_, pz_, px_poly_eqn) if do_use_newton \
        else -px_value(x_, pz_, px_poly_eqn)
    return float(px_/pz_)


def px_value_search(x_, pz_, px_poly_eqn, method='newton',
                    px_guess=0.01, px_var_=px, pz_var_=pz,
                    bracket=(0, 30)) -> float:
    """
    TBD
    """
    px_poly_eqn_ = px_poly_eqn.subs({rx: x_, x: x_, pz_var_: pz_})
    px_poly_lambda = lambdify([px_var_], px_poly_eqn_.as_expr())
    dpx_poly_lambda = lambdify([px_var_],
                               diff(px_poly_eqn_.as_expr(), px_var_)) \
        if method == 'newton' else None
    bracket_: Tuple[float, float] = bracket if method == 'brentq' else None
    for px_guess_ in [1, px_guess]:
        px_root_search \
            = root_scalar(px_poly_lambda, fprime=dpx_poly_lambda,
                          bracket=bracket_, method=method, x0=px_guess_)
        if px_root_search.converged:
            break
    # px_root_search = root_scalar( px_poly_lambda, fprime=dpx_poly_lambda,
    #     method='newton', x0=px_guess )
    px_: float = px_root_search.root
    return px_


def px_value(x_, pz_, px_poly_eqn, px_var_=px, pz_var_=pz) -> float:
    """
    TODO.

    Args:
        TODO
    """
    px_poly_eqn_ = poly(px_poly_eqn.subs({rx: x_, x: x_, pz_var_: pz_}))
    px_poly_roots = nroots(px_poly_eqn_)
    pxgen: List[float] = [root_ for root_ in px_poly_roots
                          if Abs(im(root_)) < 1e-10 and re(root_) > 0][0]
    return solve(Eq(px_poly_eqn_.gens[0], pxgen), px_var_)[0]

"""
---------------------------------------------------------------------

ODE integration functions tailored to solving Hamilton's equations.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`NumPy <numpy>`
  -  :mod:`SciPy <scipy>`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html

---------------------------------------------------------------------

"""
# Library
import warnings
import logging
# from functools import lru_cache
# from enum import Enum, auto
from typing import Dict, Any, Tuple, Callable

# NumPy
import numpy as np

# SciPy
from scipy.integrate import solve_ivp

# GME
from gme.core.symbols import Lc

warnings.filterwarnings("ignore")

rp_tuple: Tuple[str, str, str, str] = ('rx', 'rz', 'px', 'pz')
rpt_tuple: Tuple[str, str, str, str, str] = rp_tuple+('t',)

__all__ = ['solve_ODE_system', 'solve_Hamiltons_equations']


def eventAttr():
    """
    TBD
    """
    def decorator(func):
        """
        TBD
        """
        # func.direction = 0
        func.terminal = True
        return func
    return decorator

# Caching would only work if t_array were replaced with something hashable
#   - worse, to prevent recomputation for variable rz but constant rx,
#     initial conditions ic would have to be frozen in the calling function
#     and corrected afterwards


def solve_ODE_system(
    model: Callable,
    method: str,
    do_dense: bool,
    ic: Tuple[float, float, float, float],
    # t0,t1,nt,
    t_array: np.ndarray,
    x_stop: float = 0.999
) -> Any:
    """
    Integrate a coupled system of ODEs - presumed to be Hamilton's equations.
    """
    # Define stop condition
    @eventAttr()
    def almost_reached_divide(_, y):
        # function yielding >0 if rx<x1*x_stop ~ along profile
        #              and  <0 if rx>x1*x_stop ≈ @divide
        #  - thus triggers an event when rx surpasses x1*x_stop
        #    because = zero-crossing in -ve sense
        return y[0]-x_stop
    #   almost_reached_divide.terminal = True

    # Perform ODE integration
    # t_array = np.linspace(t0,t1,nt)
    return solve_ivp(model,
                     [t_array[0], t_array[-1]],
                     ic,
                     method=method,
                     t_eval=t_array,
                     dense_output=do_dense,
                     # min_step=0, #max_step=np.inf,
                     # rtol=1e-3, atol=1e-6,
                     events=almost_reached_divide,
                     vectorized=False)


def solve_Hamiltons_equations(
    model: Callable,
    method: str,
    do_dense: bool,
    ic: Tuple[float, float, float, float],
    parameters: Dict,
    t_array: np.ndarray,
    x_stop: float = 1.0,
    t_lag: float = 0
) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Perform ray tracing by integrating Hamilton's ODEs for r and p.
    """
    # Do ODE integration
    # t0, t1, nt = t_array[0], t_array[-1], len(t_array)
    ivp_soln = solve_ODE_system(model,
                                method,
                                do_dense,
                                ic,
                                # t0,t1,nt,
                                t_array,
                                x_stop=x_stop)

    # Process solution
    rp_t_soln = ivp_soln.y
    rx_array, rz_array = rp_t_soln[0], rp_t_soln[1]
    logging.debug('ode.base.solve_Hamiltons_equations:'
                  + f' ic={ic}'
                  + f' rx[0]={rx_array[0]} rz[0]={np.round(rz_array[0],5)}')
    # Did we exceed the domain bounds?
    # If so, find the index of the first point out of bounds,
    #    otherwise set as None
    i_end = np.argwhere(rx_array >= parameters[Lc])[0][0] \
        if len(np.argwhere(rx_array >= parameters[Lc])) > 0 else None
    if i_end is not None:
        if rx_array[0] != parameters[Lc]:
            i_end = min(len(t_array), i_end)
        else:
            i_end = min(len(t_array), 2)

    # Record solution
    rpt_lag_arrays = {}
    if t_lag > 0:
        dt = t_array[1]-t_array[0]
        n_lag = int(t_lag/dt)
        rpt_lag_arrays['t'] = np.linspace(0, t_lag, num=n_lag, endpoint=False)
        for rp_idx, rp_ in enumerate(rp_tuple):
            rpt_lag_arrays[rp_] = np.full(n_lag, rp_t_soln[rp_idx][0])
    else:
        n_lag = 0
        for rpt_ in rpt_tuple:
            rpt_lag_arrays[rpt_] = np.array([])

    # Report
    if i_end is not None:
        logging.debug(
            'ode.base.solve_Hamiltons_equations:\n\t'
            + f' from {np.round(rx_array[0],5)},{np.round(rz_array[0],5)}:'
            + ' out of bounds @ i='
            + f'{n_lag+i_end if i_end is not None else len(t_array)} '
            + f'x={np.round(rx_array[-1],5)} t={np.round(t_array[-1],3)}'
        )
    else:
        logging.debug(
            'ode.base.solve_Hamiltons_equations:\n\t'
            + f' from {np.round(rx_array[0],5)},{np.round(rz_array[0],5)}: '
            + f'terminating @ i={len(t_array)} '
            + f'x={np.round(rx_array[-1],5)} t={np.round(t_array[-1],3)}'
        )

    rpt_arrays: Dict[str, np.ndarray] = {}
    rpt_arrays['t'] = np.concatenate(
        (rpt_lag_arrays['t'], t_array[0:i_end]+t_lag))
    for rp_idx, rp_ in enumerate(rp_tuple):
        rpt_arrays[rp_] = np.concatenate(
            (rpt_lag_arrays[rp_], rp_t_soln[rp_idx][0:i_end]))

    return (ivp_soln, rpt_arrays)


#

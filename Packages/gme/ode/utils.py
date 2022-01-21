"""
---------------------------------------------------------------------

Utility functions

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`NumPy <numpy>`

---------------------------------------------------------------------

"""
# Library
import warnings
from typing import Dict, Tuple

# NumPy
import numpy as np

# Quadrature method from SciPy
from scipy.integrate import cumtrapz

# SymPy
from sympy import Eq

# GME
from gme.core.utils import find_dzdx_poly_root, make_dzdx_poly
from gme.core.symbols import (
    xivhat_0,
)

warnings.filterwarnings("ignore")

__all__ = ["report_progress"]


def report_progress(
    i: int,
    n: int,
    progress_was: float = 0.0,
    pc_step: float = 1,
    is_initial_step: bool = False,
) -> float:
    """
    Print percentage estimated progress of some ongoing job
    """
    progress_now: float = (
        100
        * np.round((100 / pc_step) * i / (n - 1 if n > 1 else 1))
        / np.round(100 / pc_step)
    )
    if progress_now > progress_was or is_initial_step:
        print(
            f"{progress_now:0.0f}% ",
            end="" if progress_now < 100 else "\n",
            flush=True,
        )
    return progress_now


def integrate_dzdx(
    gmeq: Eq,
    sub_: Dict,
    n_pts: int = 200,
    xivhat0_: float = 1,
    x_end: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    sub_copy = sub_.copy()
    sub_copy[xivhat_0] = xivhat0_
    dzdx_poly_ = make_dzdx_poly(gmeq.dzdx_Ci_polylike_eqn, sub_copy)
    xhat_array = np.linspace(0, x_end, n_pts, endpoint=True)
    dzdxhat_array = [
        find_dzdx_poly_root(dzdx_poly_, xhat_, xivhat0_) for xhat_ in xhat_array
    ]
    zhat_array = cumtrapz(dzdxhat_array, xhat_array, initial=0)
    return (xhat_array, zhat_array)

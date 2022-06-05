"""
Exploration of whether triangle inequality holds for non-convex 
portions of the Hamiltonian/Lagrangian.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`NumPy <numpy>`
  -  :mod:`SciPy <scipy>`
  -  :mod:`SymPy <sympy>`
  -  :mod:`MatPlotLib <matplotlib>`
  -  `GMPLib`_
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html

---------------------------------------------------------------------
"""
# pylint: disable = not-callable

# Library
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

# NumPy
import numpy as np

# Scipy utils
from scipy.linalg import norm

# SymPy
from sympy import (
    Eq,
    N,
    Abs,
    lambdify,
    Rational,
    Matrix,
    simplify,
    tan,
    solve,
    sqrt,
    im,
    oo,
)

# MatPlotLib
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.colors import LinearSegmentedColormap  # , ListedColormap

# GMPLib
from gmplib.utils import e2d
from gmplib.parameters import Parameters

# GME
from gme.core.symbols import (
    eta,
    alpha_ext,
)
from gme.plot.base import Graphing
from gme.core.equations import Equations
from gme.core.equations_extended import EquationsIdtx

warnings.filterwarnings("ignore")

__all__ = ["TriangleInequality"]


class TriangleInequality(Graphing):
    """
    Exploration of whether triangle inequality holds for non-convex
    portions of the Hamiltonian/Lagrangian.

    Extends :class:`gme.plot.base.Graphing`.
    """

    # Definitions
    # H_parametric_eqn: Eq
    # tanbeta_max: float
    # px_H_lambda: Any  # should be Callable
    # p_infc_array: np.ndarray
    # p_supc_array: np.ndarray
    # v_from_gstar_lambda: Any
    # v_infc_array: np.ndarray
    # v_supc_array: np.ndarray

    def __init__(
        self,
        gmeq: Union[Equations, EquationsIdtx],
        # pr: Parameters,
        # sub_: Dict,
    ) -> None:
        """Initialize: constructor method."""
        super().__init__()
        self.alpha_ext_eqn_: Eq = gmeq.alpha_ext_eqn.subs({eta: gmeq.eta_}).n()

    def contoured_delta_t12(
        self,
        job_name: str,
        alpha1_arrays: np.ndarray,
        delta_t1_arrays: np.ndarray,
        delta_t12_grids: np.ndarray,
        f_alpha0_list: Tuple[float] = [0.10, 0.4, 0.75],
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
    ) -> None:
        """xxx"""
        name: str
        axes: Axes
        cmap_: LinearSegmentedColormap = plt.get_cmap("Greys")

        f_alpha0_array = np.array(f_alpha0_list)
        alpha_ext_: float = float(self.alpha_ext_eqn_.rhs)
        alpha0_array = -np.deg2rad(f_alpha0_array * alpha_ext_)
        alpha0_array
        for f_alpha0_, alpha0_ in zip(f_alpha0_array, alpha0_array):
            name = f"{job_name}_falpha{f_alpha0_}".replace(".", "p")
            _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
            # print(f"alpha_0 = {np.rad2deg(alpha0_):.2f}º", flush=True)
            # figs[alpha0_] = gr.create_figure("tri ineq", fig_size=(6, 6))
            axes: Axes = plt.gca()
            contour_step = 0.1
            delta_t12_grid_ = delta_t12_grids[alpha0_].copy()
            delta_t12_min = np.round(
                np.min(delta_t12_grid_[~np.isnan(delta_t12_grid_)])
            )
            delta_t12_max = np.round(
                np.max(delta_t12_grid_[~np.isnan(delta_t12_grid_)])
                + contour_step
            )
            contour_levels = np.arange(
                delta_t12_min, delta_t12_max, contour_step
            )
            # delta_t12_grid_[np.isnan(delta_t12_grid_)] = 0
            delta_t12_grid_dummy = delta_t12_grid_.copy()
            delta_t12_grid_dummy[~np.isnan(delta_t12_grid_)] = -1
            delta_t12_grid_dummy[np.isnan(delta_t12_grid_)] = 1
            delta_t12_grid_masked = np.ma.array(
                delta_t12_grid_, mask=np.isnan(delta_t12_grid_)
            )
            xlabel = r"$\alpha_1$  [$^{\circ}$]"
            ylabel = r"$\Delta{t}_1/\Delta{t}_0$"
            contours = axes.contour(
                -np.rad2deg(alpha1_arrays[alpha0_]),
                delta_t1_arrays[alpha0_],
                delta_t12_grid_masked,
                levels=contour_levels,
                # cmap=cmap_,
            )
            axes.clabel(contours, inline=True, fmt=None, fontsize=10)
            _ = axes.contourf(
                -np.rad2deg(alpha1_arrays[alpha0_]),
                delta_t1_arrays[alpha0_],
                delta_t12_grid_dummy,
                levels=[0, 1],
                cmap=cmap_,
                alpha=0.3,
            )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            text_annotation_ = axes.text(
                0.13,
                0.85,
                r"$\dfrac{\alpha_0}{\alpha_{\mathrm{ext}}} = "
                + f"{f_alpha0_}$",
                horizontalalignment="left",
                verticalalignment="center",
                transform=axes.transAxes,
                fontsize=14,
                color="k",
            )
            text_annotation_.set_bbox(
                dict(facecolor="white", alpha=0.9, edgecolor="white")
            )

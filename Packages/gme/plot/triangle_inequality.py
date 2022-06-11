"""
Exploration of triangle inequality for geomorphic 'metric'.

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
    Check if triangle inequality holds for non-convex Hamiltonian/Lagrangian.

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
        self.gmeq: Equations = gmeq
        self.alpha_ext_eqn_: Eq = gmeq.alpha_ext_eqn.subs({eta: gmeq.eta_}).n()

    def contoured_delta_t12(
        self,
        job_name: str,
        alpha1_arrays: np.ndarray,
        ndelta_t1_arrays: np.ndarray,
        ndelta_t12_grids: np.ndarray,
        alpha0_array: np.ndarray,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        do_smooth: bool = False,
    ) -> None:
        """Contour plot shortest time for indirect paths."""
        name: str
        axes: Axes
        color_cmap_: LinearSegmentedColormap = plt.get_cmap("winter_r")
        # color_cmap_: LinearSegmentedColormap = plt.get_cmap("copper_r")
        # color_cmap_: LinearSegmentedColormap = plt.get_cmap("brg")
        grey_cmap_: LinearSegmentedColormap = plt.get_cmap("Greys")

        alpha_ext_: float = np.deg2rad(float(self.alpha_ext_eqn_.rhs))
        for alpha0_ in alpha0_array:
            f_alpha0_ = alpha0_ / alpha_ext_
            name = f"{job_name}_falpha{np.round(f_alpha0_+0.05,2)}".replace(
                ".", "p"
            )
            _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
            axes: Axes = plt.gca()
            contour_step = 0.1 / 5
            # print(alpha0_)
            ndelta_t12_grid_ = np.abs(ndelta_t12_grids[alpha0_].copy())
            # delta_t12_min = np.round(
            #     np.min(ndelta_t12_grid_[~np.isnan(ndelta_t12_grid_)]), 1
            # )
            # delta_t12_max = (
            #     np.max(ndelta_t12_grid_[~np.isnan(ndelta_t12_grid_)])
            #     + contour_step
            # )
            # print(delta_t12_min, delta_t12_max)
            # contour_levels = np.arange(
            #     delta_t12_min, delta_t12_max + contour_step, contour_step
            # )
            # contour_levels = np.concatenate(
            #     [
            #         np.arange(delta_t12_min, 0.97, 0.02),
            #         # np.arange(0.98, 1.0 - 0.001, 0.001),
            #         np.array(
            #             [
            #                 0.97,
            #                 0.98,
            #                 0.99,
            #                 0.999,
            #                 0.9999,
            #                 # 0.99999,
            #             ]
            #         ),
            #     ]
            # )
            ndelta_t12_grid_dummy = ndelta_t12_grid_.copy()
            ndelta_t12_grid_dummy[~np.isnan(ndelta_t12_grid_)] = -1
            ndelta_t12_grid_dummy[np.isnan(ndelta_t12_grid_)] = 1
            ndelta_t12_grid_masked = np.ma.array(
                ndelta_t12_grid_,
                mask=(
                    np.isnan(ndelta_t12_grid_)
                    | np.iscomplex(np.isnan(ndelta_t12_grid_))
                ),
            )
            ylabel = r"$\alpha_1$  [$\degree$]"
            xlabel = r"$\Delta{t}_{12}/\Delta{t}_0$"
            contour_fn = axes.contourf if do_smooth else axes.contour
            contours = contour_fn(
                ndelta_t1_arrays[alpha0_],
                np.rad2deg(alpha1_arrays[alpha0_]),
                ndelta_t12_grid_masked,
                levels=51,  # if do_smooth else contour_levels,
                cmap=color_cmap_,
            )

            def fmt(x):
                s = f"{x:.5f}"
                # Some horrible code here
                if s.endswith("0"):
                    s = f"{x:.4f}"
                    if s.endswith("0"):
                        s = f"{x:.3f}"
                        if s.endswith("0"):
                            s = f"{x:.2f}"
                return rf"{s}"

            if not do_smooth:
                axes.clabel(contours, inline=True, fmt=fmt, fontsize=10)

            _ = axes.contourf(
                ndelta_t1_arrays[alpha0_],
                np.rad2deg(alpha1_arrays[alpha0_]),
                ndelta_t12_grid_dummy,
                levels=[0, 1],
                cmap=grey_cmap_,
                alpha=0.3,
            )

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            text_props = {
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "white",
            }
            y_ = 0.55
            [
                axes.text(
                    *xy_,
                    text_,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes.transAxes,
                    fontsize=14,
                    color="k",
                    # bbox=text_props,
                )
                for (xy_, text_) in (
                    (
                        (
                            0.85,
                            y_,
                        ),
                        rf"$\eta = ${self.gmeq.eta_}",
                    ),
                    (
                        (
                            0.85,
                            y_ - 0.07 * 1,
                        ),
                        r"$\alpha_{\mathrm{ext}} = $"
                        + rf"{np.rad2deg(alpha_ext_):.1f}$\degree$",
                    ),
                    (
                        (
                            0.85,
                            y_ - 0.07 * 2,
                        ),
                        r"$\alpha_0 = $"
                        + rf"{np.rad2deg(alpha0_):.0f}$\degree$",
                    ),
                )
            ]
            plt.grid(":", alpha=0.3)
            plt.ylim(None, None)
            plt.xlim(None, 1.0)

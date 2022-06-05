"""
New, alternative way of visualizing indicatrix & figuratrix.

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

# GMPLib
from gmplib.utils import e2d
from gmplib.parameters import Parameters

# GME
from gme.core.symbols import (
    px,
    pz,
    varphi,
    varphi_r,
    rvec,
    xiv,
    xiv_0,
    px_min,
    pz_min,
    H,
    beta_max,
    alpha,
)
from gme.plot.base import Graphing
from gme.core.equations import Equations
from gme.core.equations_extended import EquationsIdtx

warnings.filterwarnings("ignore")

__all__ = ["IndicatrixNew"]


class IndicatrixNew(Graphing):
    """
    New, alternative way of visualizing indicatrix & figuratrix.

    Extends :class:`gme.plot.base.Graphing`.
    """

    # Definitions
    H_parametric_eqn: Eq
    tanbeta_max: float
    px_H_lambda: Any  # should be Callable
    p_infc_array: np.ndarray
    p_supc_array: np.ndarray
    v_from_gstar_lambda: Any
    v_infc_array: np.ndarray
    v_supc_array: np.ndarray

    def __init__(
        self,
        gmeq: Union[Equations, EquationsIdtx],
        pr: Parameters,
        sub_: Dict,
        varphi_: float = 1,
    ) -> None:
        """Initialize: constructor method."""
        super().__init__()
        self.H_parametric_eqn = (
            Eq((2 * gmeq.H_eqn.rhs) ** 2, 1)
            .subs({varphi_r(rvec): varphi_, xiv: xiv_0})
            .subs(sub_)
        )

        if pr.model.eta == Rational(3, 2):
            pz_min_eqn = Eq(
                pz_min,
                (
                    solve(
                        Eq(
                            (
                                (
                                    solve(
                                        Eq(4 * gmeq.H_eqn.rhs ** 2, 1).subs(
                                            {varphi_r(rvec): varphi}
                                        ),
                                        px ** 2,
                                    )[2]
                                )
                                .args[0]
                                .args[0]
                                .args[0]
                            )
                            ** 2,
                            0,
                        ),
                        pz ** 4,
                    )[0]
                )
                ** Rational(1, 4),
            )
            px_min_eqn = Eq(
                px_min,
                solve(
                    simplify(
                        gmeq.H_eqn.subs({varphi_r(rvec): varphi}).subs(
                            {pz: pz_min_eqn.rhs}
                        )
                    ).subs({H: Rational(1, 2)}),
                    px,
                )[0],
            )
            tanbeta_max_eqn = Eq(
                tan(beta_max),
                ((px_min / pz_min).subs(e2d(px_min_eqn))).subs(e2d(pz_min_eqn)),
            )
            self.tanbeta_max = float(N(tanbeta_max_eqn.rhs))
        else:
            pz_min_eqn = Eq(pz_min, 0)
            px_min_eqn = Eq(
                px_min,
                sqrt(
                    solve(
                        Eq(
                            (
                                solve(
                                    Eq(4 * gmeq.H_eqn.rhs ** 2, 1).subs(
                                        {varphi_r(rvec): varphi}
                                    ),
                                    pz ** 2,
                                )[:]
                            )[0],
                            0,
                        ),
                        px ** 2,
                    )[1]
                ),
            )
            tanbeta_max_eqn = Eq(tan(beta_max), oo)
            self.tanbeta_max = 0.0

        pz_min_ = round(float(N(pz_min_eqn.rhs.subs({varphi: varphi_}))), 8)

        px_H_solns = [
            simplify(sqrt(soln))
            for soln in solve(self.H_parametric_eqn, px ** 2)
        ]
        px_H_soln_ = [
            soln
            for soln in px_H_solns
            if Abs(im(N(soln.subs({pz: 1})))) < 1e-10
        ][0]
        self.px_H_lambda = lambdify([pz], simplify(px_H_soln_))

        pz_max_ = 10 ** 4 if pr.model.eta == Rational(3, 2) else 10 ** 2
        pz_array = -(
            10
            ** np.linspace(
                np.log10(pz_min_ if pz_min_ > 0 else 1e-6),
                np.log10(pz_max_),
                1000,
            )
        )
        px_array = self.px_H_lambda(pz_array)
        p_array = np.vstack([px_array, pz_array]).T
        tanbeta_crit = float(N(gmeq.tanbeta_crit_eqn.rhs))

        self.p_infc_array = p_array[
            np.abs(p_array[:, 0] / p_array[:, 1]) < tanbeta_crit
        ]
        self.p_supc_array = p_array[
            np.abs(p_array[:, 0] / p_array[:, 1]) >= tanbeta_crit
        ]

        v_from_gstar_lambda_tmp = lambdify(
            (px, pz),
            N(
                gmeq.gstar_varphi_pxpz_eqn.subs({varphi_r(rvec): varphi_}).rhs
                * Matrix([px, pz])
            ),
        )
        self.v_from_gstar_lambda = lambda px_, pz_: (
            v_from_gstar_lambda_tmp(px_, pz_)
        ).flatten()

        def v_lambda(pa):
            return np.array(
                [(self.v_from_gstar_lambda(px_, pz_)) for px_, pz_ in pa]
            )

        self.v_infc_array = v_lambda(self.p_infc_array)
        self.v_supc_array = v_lambda(self.p_supc_array)

    def convex_concave_annotations(self, do_zoom: bool, eta_: float) -> None:
        """Annotate with 'convex' or 'concave' labels."""
        if do_zoom:
            if eta_ > 1:
                plt.text(
                    *(1.025, 0.184),
                    "convex",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=40,
                    fontsize=15,
                    color="DarkRed",
                )
                plt.text(
                    *(1.054, 0.208),
                    "concave",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=60,
                    fontsize=11,
                    color="r",
                )
            else:
                plt.text(
                    *(1.07, -0.264),
                    "concave",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=15,
                    fontsize=15,
                    color="r",
                )
                plt.text(
                    *(0.955, -0.15),
                    "convex",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=70,
                    fontsize=15,
                    color="DarkRed",
                )
        else:
            if eta_ > 1:
                plt.text(
                    *(0.7, -0.05),
                    "convex",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=11,
                    fontsize=15,
                    color="DarkRed",
                )
                plt.text(
                    *(1.15, 0.42),
                    "concave",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=60,
                    fontsize=10,
                    color="r",
                )
                plt.text(
                    *(1.4, -0.72),
                    "concave",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=10,
                    fontsize=15,
                    color="b",
                )
                plt.text(
                    *(1.5, -2.3),
                    "convex",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=-80,
                    fontsize=15,
                    color="DarkBlue",
                )
            else:
                plt.text(
                    *(1.3, -0.26),
                    "concave",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=10,
                    fontsize=13,
                    color="r",
                )
                plt.text(
                    *(0.9, -0.14),
                    "convex",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=65,
                    fontsize=11,
                    color="DarkRed",
                )
                plt.text(
                    *(0.98, -0.65),
                    "convex",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=75,
                    fontsize=15,
                    color="DarkBlue",
                )
                plt.text(
                    *(0.66, -1.65),
                    "concave",
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=75,
                    fontsize=15,
                    color="b",
                )

    def Fstar_F_rectlinear(
        self,
        gmeq: Union[Equations, EquationsIdtx],
        job_name: str,
        pr: Parameters,
        v_eqns: Optional[Tuple[Eq, Eq]] = None,
        do_zoom: bool = False,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
    ) -> None:
        """Plot :math:`F^*`, :math:`F` on rectilinear axes."""
        name: str = f'{job_name}_Fstar_F_rectlinear{"_zoom" if do_zoom else ""}'
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes: Axes = plt.gca()

        eta_ = pr.model.eta
        if do_zoom:
            if eta_ == Rational(3, 2):
                plt.xlim(0.98, 1.07)
                plt.ylim(0.15, 0.23)
                eta_xy_label = (0.2, 0.85)
            else:
                plt.xlim(0.7, 1.2)
                plt.ylim(-0.4, 0)
                eta_xy_label = (0.8, 0.8)
        else:
            if eta_ == Rational(3, 2):
                plt.xlim(0, 2)
                plt.ylim(-4, 0.6)
                eta_xy_label = (0.7, 0.8)
            else:
                plt.xlim(0, 2.5)
                plt.ylim(-2, 0)
                eta_xy_label = (0.8, 0.7)

        # Critical, bounding angles
        pz_max_ = -1.5
        px_abmax_ = -pz_max_ * (self.tanbeta_max if self.tanbeta_max > 0 else 1)
        pz_abmax_ = pz_max_
        vx_abmax_, vz_abmax_ = self.v_from_gstar_lambda(px_abmax_, pz_abmax_)
        px_abcrit_ = -pz_max_ * gmeq.tanbeta_crit
        pz_abcrit_ = pz_max_
        vx_abcrit_, vz_abcrit_ = self.v_from_gstar_lambda(
            px_abcrit_, pz_abcrit_
        )

        # Lines visualizing critical, bounding angles: ray velocity
        if eta_ == Rational(3, 2):
            plt.plot(
                [0, vx_abmax_],
                [0, vz_abmax_],
                "-",
                color="r",
                alpha=0.4,
                lw=2,
                label=r"$\alpha_{\mathrm{lim}}$",
            )

        # Fundamental function F=1 as r(alpha) from closed-form Lagrangian
        if v_eqns is not None:
            vx_array = self.v_infc_array[:, 0]
            vy_array = self.v_infc_array[:, 1]
            alpha_array = np.arctan(vy_array / vx_array)
            for i_, r_eqn_ in enumerate(v_eqns):
                v_alpha_lambdified = lambdify((alpha), r_eqn_.rhs)
                v_alpha_lambda = lambda alpha_: v_alpha_lambdified(alpha_)
                v_array = v_alpha_lambda(alpha_array)
                plt.plot(
                    v_array * np.cos(alpha_array),
                    v_array * np.sin(alpha_array),
                    ("r", "DarkRed")[i_] if eta_ < 1 else ("DarkRed", "r_")[i_],
                    lw=3,
                    ls=":",
                )

        # Indicatrix aka F=1 for rays
        plt.plot(
            self.v_supc_array[:, 0],
            self.v_supc_array[:, 1],
            "r" if eta_ > 1 else "DarkRed",
            lw=2,
            ls="-",
            label=r"$F=1$,  $\beta\geq\beta_\mathrm{c}$",
        )
        plt.plot(
            [0, vx_abcrit_],
            [0, vz_abcrit_],
            "-.",
            color="DarkRed" if eta_ > 1 else "r",
            lw=1,
            label=r"$\alpha_{\mathrm{ext}}$",
        )
        plt.plot(
            self.v_infc_array[:, 0],
            self.v_infc_array[:, 1],
            "DarkRed" if eta_ > 1 else "r",
            lw=1 if eta_ == Rational(3, 2) and not do_zoom else 2,
            ls="-",
            label=r"$F=1$,  $\beta <\beta_\mathrm{c}$",
        )

        # Lines visualizing critical, bounding angles: normal slowness
        if eta_ == Rational(3, 2) and not do_zoom:
            plt.plot(
                np.array([0, px_abmax_]),
                [0, pz_abmax_],
                "-b",
                alpha=0.4,
                lw=1.5,
                label=r"$\beta_{\mathrm{max}}$",
            )

        # Figuratrix aka F*=1 for surfaces
        if not do_zoom:
            plt.plot(
                self.p_supc_array[:, 0],
                self.p_supc_array[:, 1],
                "b" if eta_ > 1 else "DarkBlue",
                lw=2,
                ls="-",
                label=r"$F^*\!\!=1$,  $\beta\geq\beta_\mathrm{c}$",
            )
            plt.plot(
                [0, px_abcrit_],
                [0, pz_abcrit_],
                "--",
                color="DarkBlue" if eta_ > 1 else "b",
                lw=1,
                label=r"$\beta_{\mathrm{c}}$",
            )
            plt.plot(
                self.p_infc_array[:, 0],
                self.p_infc_array[:, 1],
                "DarkBlue" if eta_ > 1 else "b",
                lw=2,
                ls="-",
                label=r"$F^*\!\!=1$,  $\beta<\beta_\mathrm{c}$",
            )

        pz_ = -float(
            solve(
                self.H_parametric_eqn.subs({px: pz * (gmeq.tanbeta_crit)}), pz
            )[0]
        )
        px_ = self.px_H_lambda(pz_)
        (vx_, vz_) = self.v_from_gstar_lambda(px_, pz_)
        if eta_ != Rational(3, 2):
            plt.plot([vx_], [-vz_], "o", color="r", ms=5)
        if not do_zoom:
            plt.plot(
                [px_], [-pz_], "o", color="DarkBlue" if eta_ > 1 else "b", ms=5
            )

        plt.xlabel(r"$p_x$ (for $F^*$)  or  $v^x$ (for $F$)", fontsize=14)
        plt.ylabel(r"$p_z$ (for $F^*$)  or  $v^z$ (for $F$)", fontsize=14)

        axes.set_aspect(1)
        plt.text(
            *eta_xy_label,
            rf"$\eta={gmeq.eta_}$",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=15,
            color="k",
        )

        if eta_ == Rational(3, 2):
            loc_ = "lower right" if do_zoom else "lower left"
        else:
            loc_ = "upper left" if do_zoom else "lower right"
        plt.legend(loc=loc_)

        self.convex_concave_annotations(do_zoom, eta_)

        plt.grid(True, ls=":")

    def Fstar_F_polar(
        self,
        gmeq: Union[Equations, EquationsIdtx],
        job_name: str,
        pr: Parameters,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
    ) -> None:
        """Plot :math:`F^*`, :math:`F` on log-polar axes."""
        name = f"{job_name}_Fstar_F_polar"
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        eta_ = pr.model.eta

        scale_fn = np.log10
        if eta_ > 1:
            r_min_, r_max_ = 0.1, 100

            def alpha_fn(a):
                return np.pi - a

        else:
            r_min_, r_max_ = 0.1, 10

            def alpha_fn(a):
                return a

        def v_scale_fn(v):
            return scale_fn(v) * 1

        # Lines visualizing critical, bounding angles: ray velocity
        if eta_ > 1:
            plt.polar(
                [np.pi / 2 + (np.arctan(gmeq.tanalpha_ext))] * 2,
                [scale_fn(r_min_), scale_fn(r_max_)],
                "-",
                color="r" if eta_ > 1 else "DarkRed",
                alpha=0.4,
                lw=2,
                label=r"$\alpha_{\mathrm{lim}}$",
            )
        plt.polar(
            alpha_fn(
                np.arcsin(
                    self.v_supc_array[:, 0] / norm(self.v_supc_array, axis=1)
                )
            ),
            v_scale_fn(norm(self.v_supc_array, axis=1)),
            "r" if eta_ > 1 else "DarkRed",
            label=r"$F=1$,  $\beta\geq\beta_\mathrm{c}$",
        )
        plt.polar(
            [np.pi / 2 + (np.arctan(gmeq.tanalpha_ext))] * 2,
            [scale_fn(r_min_), scale_fn(r_max_)],
            "-.",
            color="DarkRed" if eta_ > 1 else "r",
            lw=1,
            label=r"$\alpha_{\mathrm{ext}}$",
        )
        plt.polar(
            alpha_fn(
                np.arcsin(
                    self.v_infc_array[:, 0] / norm(self.v_infc_array, axis=1)
                )
            ),
            v_scale_fn(norm(self.v_infc_array, axis=1)),
            "DarkRed" if eta_ > 1 else "r",
            lw=None if eta_ == Rational(3, 2) else None,
            label=r"$F=1$,  $\beta <\beta_\mathrm{c}$",
        )

        unit_circle_array = np.array(
            [[theta_, 1] for theta_ in np.linspace(0, (np.pi / 2) * 1.2, 100)]
        )
        plt.polar(
            unit_circle_array[:, 0],
            scale_fn(unit_circle_array[:, 1]),
            "-",
            color="g",
            lw=1,
            label="unit circle",
        )

        if eta_ > 1:
            plt.polar(
                [np.arctan(self.tanbeta_max)] * 2,
                [scale_fn(r_min_), scale_fn(r_max_)],
                "-",
                color="b",
                alpha=0.3,
                lw=1.5,
                label=r"$\beta_{\mathrm{max}}$",
            )
        plt.polar(
            np.arcsin(
                self.p_supc_array[:, 0] / norm(self.p_supc_array, axis=1)
            ),
            scale_fn(norm(self.p_supc_array, axis=1)),
            "b" if eta_ > 1 else "DarkBlue",
            label=r"$F^*\!\!=1$,  $\beta\geq\beta_\mathrm{c}$",
        )
        plt.polar(
            [np.arctan(gmeq.tanbeta_crit)] * 2,
            [scale_fn(r_min_), scale_fn(r_max_)],
            "--",
            color="DarkBlue" if eta_ > 1 else "b",
            lw=1,
            label=r"$\beta_{\mathrm{c}}$",
        )
        plt.polar(
            np.arcsin(
                self.p_infc_array[:, 0] / norm(self.p_infc_array, axis=1)
            ),
            scale_fn(norm(self.p_infc_array, axis=1)),
            "DarkBlue" if eta_ > 1 else "b",
            label=r"$F^*\!\!=1$,  $\beta<\beta_\mathrm{c}$",
        )

        plt.polar(
            (
                np.arcsin(
                    self.p_supc_array[-1, 0] / norm(self.p_supc_array[-1])
                )
                + np.arcsin(
                    self.p_infc_array[0, 0] / norm(self.p_infc_array[0])
                )
            )
            / 2,
            (
                scale_fn(norm(self.p_infc_array[0]))
                + scale_fn(norm(self.p_supc_array[-1]))
            )
            / 2,
            "o",
            color="DarkBlue" if eta_ > 1 else "b",
        )

        axes: Axes = plt.gca()
        axes.set_theta_zero_location("S")
        horiz_label = r"$\log_{10}{p}$  or  $\log_{10}{v}$"
        vert_label = r"$\log_{10}{v}$  or  $\log_{10}{p}$"

        xtick_labels_base = [r"$\beta=0^{\!\circ}$", r"$\beta=30^{\!\circ}$"]
        theta_list: List[float]
        if eta_ > 1:
            theta_max_ = 20
            axes.set_thetamax(90 + theta_max_)
            axes.text(
                np.deg2rad(85 + theta_max_),
                0.5,
                vert_label,
                rotation=theta_max_,
                ha="center",
                va="bottom",
                fontsize=15,
            )
            axes.text(
                np.deg2rad(-8),
                1.2,
                horiz_label,
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=15,
            )
            theta_list = [0, 1 / 6, 2 / 6, 3 / 6, np.deg2rad(110) / np.pi]
            xtick_labels = xtick_labels_base + [
                r"$\beta=60^{\!\circ}$",
                r"$\alpha=0^{\!\circ}$",
                r"$\alpha=20^{\!\circ}$",
            ]
            eta_xy_label = (1.15, 0.9)
            legend_xy = (1.0, 0.0)
            plt.text(
                *[(np.pi / 2) * 1.07, 0.4],
                "convex",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=8,
                fontsize=15,
                color="DarkRed",
            )
            plt.text(
                *[(np.pi / 2) * 1.17, 0.28],
                "concave",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=13,
                fontsize=11,
                color="r",
            )
            plt.text(
                *[(np.pi / 3) * 0.925, 0.5],
                "concave",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=-35,
                fontsize=15,
                color="b",
            )
            plt.text(
                *[(np.pi / 6) * 0.7, 0.85],
                "convex",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=68,
                fontsize=15,
                color="DarkBlue",
            )
        else:
            theta_max_ = 0
            axes.set_thetamax(90 + theta_max_)
            axes.text(
                np.deg2rad(92 + theta_max_),
                axes.get_rmax() / 5,
                vert_label,
                rotation=theta_max_,
                ha="right",
                va="bottom",
                fontsize=15,
            )
            axes.text(
                np.deg2rad(-8),
                axes.get_rmax() / 5,
                horiz_label,
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=15,
            )
            theta_list = [0, 1 / 6, 2 / 6, 3 / 6]
            xtick_labels = xtick_labels_base + [
                r"$\beta=60^{\!\circ}\!\!,\, \alpha=-30^{\!\circ}$",
                r"$\beta=90^{\!\circ}\!\!,\, \alpha=0^{\!\circ}$",
            ]
            eta_xy_label = (1.2, 0.75)
            legend_xy = (0.9, 0.0)
            plt.text(
                *[(np.pi / 2) * 0.94, 0.4],
                "concave",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=11,
                fontsize=15,
                color="r",
            )
            plt.text(
                *[(np.pi / 2) * 0.9, -0.07],
                "convex",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=72,
                fontsize=13,
                color="DarkRed",
            )
            plt.text(
                *[(np.pi / 4) * 1.2, 0.12],
                "convex",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=60,
                fontsize=15,
                color="DarkBlue",
            )
            plt.text(
                *[(np.pi / 6) * 0.5, 0.4],
                "concave",
                horizontalalignment="center",
                verticalalignment="center",
                rotation=50,
                fontsize=15,
                color="b",
            )
            plt.polar(
                alpha_fn(
                    np.arcsin(
                        self.v_supc_array[:, 0]
                        / norm(self.v_supc_array, axis=1)
                    )
                ),
                v_scale_fn(norm(self.v_supc_array, axis=1)),
                "DarkRed",
            )

        plt.polar(
            alpha_fn(
                (
                    np.arcsin(
                        self.v_supc_array[-1, 0] / norm(self.v_supc_array[-1])
                    )
                    + np.arcsin(
                        self.v_infc_array[0, 0] / norm(self.v_infc_array[0])
                    )
                )
                / 2
            ),
            (
                v_scale_fn(norm(self.v_infc_array[0]))
                + v_scale_fn(norm(self.v_supc_array[-1]))
            )
            / 2,
            "o",
            color="DarkRed" if eta_ > 1 else "r",
        )

        xtick_posns = [np.pi * theta_ for theta_ in theta_list]
        plt.xticks(xtick_posns, xtick_labels, ha="left", fontsize=15)

        plt.text(
            *eta_xy_label,
            rf"$\eta={gmeq.eta_}$",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            color="k",
        )
        plt.legend(loc=legend_xy)

        axes.tick_params(
            axis="x", pad=0, left=True, length=5, width=1, direction="out"
        )

        axes.set_aspect(1)
        axes.set_rmax(scale_fn(r_max_))
        axes.set_rmin(scale_fn(r_min_))
        axes.set_thetamin(0)
        plt.grid(False, ls=":")

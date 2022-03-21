"""
Visualization of solutions with time-varying boundary conditions.

Requires Python packages:
  -  :mod:`NumPy <numpy>`
  -  :mod:`SymPy <sympy>`
  -  :mod:`MatPlotLib <matplotlib>`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html
"""
# Library
import warnings
from typing import Dict, Tuple, Optional

# NumPy
import numpy as np

# SymPy
from sympy import deg

# MatPlotLib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# GME
from gme.core.symbols import Ci
from gme.core.equations import Equations
from gme.ode.base import rpt_tuple
from gme.ode.velocity_boundary import VelocityBoundarySolution
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ["TimeDependent"]


class TimeDependent(Graphing):
    """
    Visualization of solutions with time-varying boundary conditions.

    Extends :class:`gme.plot.base.Graphing`.
    """

    def profile_isochrones(
        self,
        gmes: VelocityBoundarySolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        do_zero_isochrone: bool = True,
        do_rays: bool = True,
        ray_subsetting: int = 5,
        ray_lw: float = 0.5,
        ray_ls: str = "-",
        ray_label: str = "ray",
        do_isochrones: bool = True,
        isochrone_subsetting: int = 1,
        isochrone_lw: float = 0.5,
        isochrone_ls: str = "-",
        do_annotate_rays: bool = False,
        n_arrows: int = 10,
        arrow_sf: float = 0.7,
        arrow_offset: int = 4,
        do_annotate_cusps: bool = False,
        cusp_lw: float = 1.5,
        do_smooth_colors: bool = False,
        x_limits: Tuple[float, float] = (-0.001, 1.001),
        y_limits: Tuple[float, float] = (-0.025, 0.525),
        aspect: float = None,
        do_legend: bool = True,
        do_alt_legend: bool = False,
        do_grid: bool = True,
        do_infer_initiation: bool = True,
        do_pub_label: bool = False,
        do_etaxi_label: bool = True,
        pub_label: Optional[str] = None,
        eta_label_xy: Tuple[float, float] = (0.65, 0.85),
        pub_label_xy: Tuple[float, float] = (0.5, 0.92),
    ) -> None:
        """
        Plot isochrones and rays for a time-dependent b.c. solution.

        Args:
            gmes:
                instance of velocity boundary solution class defined in
                :mod:`gme.ode.velocity_boundary`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for equation
                substitutions
            name:
                name of plot in figures dictionary
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            do_zero_isochrone:
                optionally plot initial surface?
            do_rays:
                optionally plot rays?
            ray_subsetting:
                optional rate of ray subsetting
            ray_lw:
                optional ray line width
            ray_ls:
                optional ray line style
            ray_label:
                optional ray line label
            do_isochrones:
                optionally plot isochrones?
            isochrone_subsetting:
                optional rate of isochrone subsetting
            do_isochrone_p:
                optionally plot isochrone herringbones?
            isochrone_lw:
                optional isochrone line width
            isochrone_ls:
                optional isochrone line style
            do_annotate_rays:
                optionally plot arrowheads along rays?
            n_arrows:
                optional number of arrowheads to annotate along rays or
                cusp-line
            arrow_sf:
                optional scale factor for arrowhead sizes
            arrow_offset:
                optional offset to start annotating arrowheads
            do_annotate_cusps:
                optionally plot line to visualize cusp initiation and
                propagation
            cusp_lw:
                optional cusp propagation curve line width
            x_limits:
                optional [x_min, x_max] horizontal plot range
            y_limits:
                optional [z_min, z_max] vertical plot range
            aspect:
                optional figure aspect ratio
            do_legend:
                optionally plot legend
            do_alt_legend:
                optionally plot slightly different legend
            do_grid:
                optionally plot dotted grey gridlines
            do_infer_initiation:
                optionally draw dotted line inferring cusp initiation at the
                left boundary
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Unpack for brevity
        if hasattr(gmes, "rpt_isochrones"):
            rx_isochrones, rz_isochrones, _, _, t_isochrones = [
                gmes.rpt_isochrones[rpt_] for rpt_ in rpt_tuple
            ]

        # Initial boundary
        if hasattr(gmes, "rpt_isochrones") and do_zero_isochrone:
            n_isochrones: int = len(rx_isochrones)
            plt.plot(
                rx_isochrones[0],
                rz_isochrones[0],
                "-",
                color=self.gray_color(0, n_isochrones),
                lw=2,
                label=("zero isochrone" if do_legend else None),
            )

        # Rays
        axes = plt.gca()
        if do_rays:
            n_rays = len(gmes.rpt_arrays["rx"])
            for i_ray, (rx_array, rz_array, _) in enumerate(
                zip(
                    reversed(gmes.rpt_arrays["rx"]),
                    reversed(gmes.rpt_arrays["rz"]),
                    reversed(gmes.rpt_arrays["t"]),
                )
            ):
                if (i_ray // ray_subsetting - i_ray / ray_subsetting) == 0:
                    this_ray_label = (
                        ray_label + r" ($t_{\mathrm{oldest}}$)"
                        if i_ray == 0
                        else ray_label + r" ($t_{\mathrm{newest}}$)"
                        if i_ray == n_rays - 1
                        else ""
                    )
                    if do_annotate_rays:
                        self.arrow_annotate_ray_custom(
                            rx_array,
                            rz_array,
                            axes,
                            i_ray,
                            ray_subsetting,
                            n_rays,
                            n_arrows,
                            arrow_sf,
                            arrow_offset,
                            x_limits=x_limits,
                            y_limits=y_limits,
                            line_style=ray_ls,
                            line_width=ray_lw,
                            ray_label=this_ray_label,
                            do_smooth_colors=do_smooth_colors,
                        )
                    else:
                        color_ = self.mycolors(
                            i_ray,
                            ray_subsetting,
                            n_rays,
                            do_smooth=do_smooth_colors,
                        )
                        plt.plot(
                            rx_array,
                            rz_array,
                            lw=ray_lw,
                            color=color_,
                            linestyle=ray_ls,
                            label=this_ray_label,
                        )

        # Time slices or isochrones of erosion front
        if hasattr(gmes, "rpt_isochrones") and do_isochrones:
            n_isochrones = len(rx_isochrones)
            delta_t: float = t_isochrones[1]
            i_isochrone, rx_isochrone, rz_isochrone = None, None, None
            # suppresses annoying pylint warning
            for i_isochrone, (rx_isochrone, rz_isochrone, _) in enumerate(
                zip(rx_isochrones, rz_isochrones, t_isochrones)
            ):
                i_subsetted: float = float(
                    (
                        i_isochrone // isochrone_subsetting
                        - i_isochrone / isochrone_subsetting
                    )
                )
                i_subsubsetted: float = float(
                    (
                        i_isochrone // (isochrone_subsetting * 10)
                        - i_isochrone / (isochrone_subsetting * 10)
                    )
                )
                if (
                    i_isochrone > 0
                    and i_subsetted == 0.0
                    and rx_isochrone is not None
                ):
                    lw_: float = (
                        1.3 * isochrone_lw
                        if i_subsubsetted == 0.0
                        else 0.5 * isochrone_lw
                    )
                    plt.plot(
                        rx_isochrone,
                        rz_isochrone,
                        self.gray_color(i_isochrone, n_isochrones),
                        linestyle=isochrone_ls,
                        lw=lw_,
                    )
            # Hack legend items
            if rx_isochrone is not None:
                label_part = r"isochrone $\Delta{\hat{t}}=$"
                label_ = rf"{label_part}${int(10*delta_t)}$"
                plt.plot(
                    rx_isochrone,
                    rz_isochrone,
                    self.gray_color(i_isochrone, n_isochrones),
                    linestyle=isochrone_ls,
                    lw=1.3 * isochrone_lw,
                    label=label_,
                )
                label_ = rf"{label_part}${round(delta_t,1)}$"
                plt.plot(
                    rx_isochrone,
                    rz_isochrone,
                    self.gray_color(i_isochrone, n_isochrones),
                    linestyle=isochrone_ls,
                    lw=0.5 * isochrone_lw,
                    label=label_,
                )

        # Knickpoint aka cusp propagation
        if do_annotate_cusps:
            rxz_array = np.array(
                [
                    rxz
                    for (t, rxz), _, _ in gmes.trxz_cusps
                    if rxz != [] and rxz[0] <= 1.01
                ]
            )
            if (n_cusps := len(rxz_array)) > 0:
                # Plot locations of cusps as a propagation curve
                plt.plot(
                    rxz_array.T[0][:-1],
                    rxz_array.T[1][:-1],
                    lw=cusp_lw,
                    color="r",
                    alpha=0.4,
                    label="cusp propagation",
                )

                # Plot inferred initiation of cusp propagation
                #   from LHS boundary
                if do_infer_initiation:
                    ((x1_, y1_), (x2_, y2_)) = rxz_array[0:2]
                    dx_, dy_ = (x2_ - x1_) / 100, (y2_ - y1_) / 100
                    x0_ = 0
                    y0_ = y1_ + (dy_ / dx_) * (x0_ - x1_)
                    plt.plot(
                        (x0_, x1_),
                        (y0_, y1_),
                        ":",
                        lw=cusp_lw,
                        color="r",
                        alpha=0.4,
                        label="inferred initiation",
                    )

                # Plot arrow annotations using subset of cusps
                cusp_subsetting = max(1, n_cusps // n_arrows)
                rxz_array_subset = rxz_array[::cusp_subsetting].T
                sf: float = 0.7
                my_arrow_style = mpatches.ArrowStyle.Fancy(
                    head_length=0.99 * sf,
                    head_width=0.6 * sf,
                    tail_width=0.0001,
                )
                for ((x1_, y1_), (x0_, y0_)) in zip(
                    rxz_array_subset.T[:-1], rxz_array_subset.T[1:]
                ):
                    dx_, dy_ = (x1_ - x0_) / 100, (y1_ - y0_) / 100
                    if (
                        x_limits is None
                        or (
                            x0_ >= x_limits[0]
                            if x_limits[0] is not None
                            else True
                        )
                        and (
                            x0_ <= x_limits[1]
                            if x_limits[1] is not None
                            else True
                        )
                    ):
                        axes.annotate(
                            "",
                            xy=(x0_, y0_),
                            xytext=(x0_ + dx_, y0_ + dy_),
                            arrowprops=dict(
                                arrowstyle=my_arrow_style, color="r", alpha=0.4
                            ),
                        )

        # Label axes
        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=16)
        plt.ylabel(r"Elevation, $z/L_{\mathrm{c}}$  [-]", fontsize=16)

        if do_legend or do_alt_legend:
            plt.legend(loc="upper left", fontsize=10, framealpha=0.95)

        # Tidy axes etc
        plt.xlim(*x_limits)
        plt.ylim(*y_limits)
        if do_grid:
            plt.grid(True, ls=":")
        axes.set_aspect(1 if aspect is None else aspect)

        if do_etaxi_label:
            plt.text(
                *eta_label_xy,
                rf"$\eta={gmeq.eta_}$"
                + r"$\quad\mathsf{Ci}=$"
                + rf"${round(float(deg(Ci.subs(sub))))}\degree$",
                transform=axes.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                color="k",
            )
        if do_pub_label:
            plt.text(
                *pub_label_xy,
                pub_label,
                transform=axes.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                color="k",
            )

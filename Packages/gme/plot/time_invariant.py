"""
---------------------------------------------------------------------

Visualization of time-invariant solutions.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`NumPy <numpy>`
  -  :mod:`SymPy <sympy>`
  -  :mod:`MatPlotLib <matplotlib>`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html

---------------------------------------------------------------------

"""
# pylint: disable = too-few-public-methods, no-self-use

# Library
import warnings
from typing import Dict, Tuple, Optional

# NumPy
import numpy as np

# SymPy
from sympy import N, Rational, deg

# MatPlotLib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# GME
from gme.core.symbols import Ci, mu
from gme.core.equations import Equations
from gme.ode.time_invariant import TimeInvariantSolution
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ["TimeInvariant"]


class TimeInvariant(Graphing):
    """
    Visualization of time-invariant solutions.

    Extends :class:`gme.plot.base.Graphing`.
    """

    def profile_aniso(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        n_points: int = 51,
        xf_stop: float = 0.995,
        sf: Optional[Tuple[float, float]] = None,
        n_arrows: int = 26,
        y_limits: Optional[Tuple[float, float]] = None,
        v_scale: float = 0.4,
        v_exponent: float = 1.0,
        do_pub_label: bool = False,
        pub_label: str = "",
        eta_label_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        r"""
        Plot time-invariant profile annotated with ray vectors,
        normal-slowness covector herringbones,
        and colorized for anisotropy (defined as the difference
        :math:`(\alpha-\beta)`).

        Args:
            gmes:
                instance of time invariant solution class
                defined in :mod:`gme.ode.time_invariant`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for
                equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            n_points:
                optional sample rate along each curve
            xf_stop:
                optional x-fraction at which to stop plotting
            sf:
                optional scale factor(s) for vertical axes (bottom and top)
            n_arrows:
                optional number of :math:`\mathbf{v}` vector arrows and
                :math:`\mathbf{\widetilde{p}}` covector herringbones to plot
            y_limits:
                optional [z_min, z_max] vertical plot range
            v_scale:
                optional velocity arrow prefactor
            v_exponent:
                optional velocity arrow exaggeration power function exponent
            eta_label_xy:
                optional where to plot :math:`\eta` annotation text
            do_pub_label:
                optionally do 'publication' annotation?
            pub_label:
                optional 'publication' annotation text
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        eta_label_xy = (0.5, 0.8) if eta_label_xy is None else eta_label_xy
        x_array = np.linspace(0, 1 * xf_stop, n_points)
        h_array = gmes.h_interp(x_array)

        profile_lw = 1
        # Solid line = topo profile from direct integration of gradient array
        plt.plot(
            gmes.h_x_array,
            (gmes.h_z_array - gmes.h_z_array[0]),
            "k",
            lw=profile_lw,
            label=rf"$h(x)\quad\eta={gmeq.eta_}$",
        )
        plt.plot(
            x_array,
            h_array,
            "o",
            mec="k",
            mfc="gray",
            ms=3,
            fillstyle="full",
            markeredgewidth=0.5,
        )

        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=16)
        plt.ylabel(r"Elevation, $z/L_{\mathrm{c}}$  [-]", fontsize=16)
        plt.grid(True, ls=":")

        axes = plt.gca()
        ylim = axes.get_ylim()

        cmap_choice = "viridis_r"
        #     cmap_choice = 'plasma_r'
        #     cmap_choice = 'magma_r'
        #     cmap_choice = 'inferno_r'
        #     cmap_choice = 'cividis_r'
        shape = "full"

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max * xf_stop, n_arrows)
        h_array = gmes.h_interp(x_array)
        beta_array = gmes.beta_p_interp(x_array)
        alpha_array = gmes.alpha_interp(x_array)
        rdot_array = gmes.rdot_interp(x_array)
        p_array = gmes.p_interp(x_array)
        aniso_array = np.rad2deg(alpha_array - beta_array)

        aniso_span = np.array((10 * np.floor(min(aniso_array) / 10), 90))
        color_map = cm.get_cmap(cmap_choice)
        # stretch_aniso_array = (((aniso_array-min(aniso_array))
        #   /(max(aniso_array)-min(aniso_array)))**30) \
        #                *(max(aniso_array)-min(aniso_array))+min(aniso_array)
        aniso_colors = [
            color_map(
                (aniso_ - aniso_span[0]) / (aniso_span[1] - aniso_span[0])
            )
            for aniso_ in aniso_array
        ]
        for rp_idx in (0, 1):
            # Fix the range of p values to be represented: smaller values
            # will be clipped
            p_range_max = 20
            p_max = max(p_array)
            p_min = min(p_array)
            p_min = (
                p_max / p_range_max if p_max / p_min > p_range_max else p_min
            )
            p_range = p_max - p_min
            np_scale = 15
            # Fix the range of rdot values to be represented:
            #     smaller values will be clipped
            rdot_range_max = 20
            rdot_max, rdot_min = max(rdot_array), min(rdot_array)
            rdot_min = (
                rdot_max / rdot_range_max
                if rdot_max / rdot_min > rdot_range_max
                else rdot_min
            )
            rdot_range = rdot_max - rdot_min
            nrdot_scale = 9
            for (x_, z_, aniso_color, alpha_, beta_, rdot_, p_) in zip(
                x_array,
                h_array,
                aniso_colors,
                alpha_array,
                beta_array,
                rdot_array,
                p_array,
            ):
                if rp_idx == 0:
                    # Ray vector
                    hw = 0.01
                    hl = 0.02
                    oh = 0.1
                    len_arrow = (
                        v_scale
                        * ((rdot_ - rdot_min) ** v_exponent / rdot_range)
                        if rdot_ >= rdot_min
                        else 0
                    ) + v_scale / nrdot_scale
                    dx, dz = (
                        len_arrow * np.cos(alpha_ - np.pi / 2),
                        len_arrow * np.sin(alpha_ - np.pi / 2),
                    )
                    plt.arrow(
                        x_,
                        z_,
                        dx,
                        dz,
                        head_width=hw,
                        head_length=hl,
                        lw=1,
                        shape=shape,
                        overhang=oh,
                        length_includes_head=True,
                        ec=aniso_color,
                        fc=aniso_color,
                    )
                else:
                    # Slowness covector
                    len_stick = 0.08
                    hw = 0.015
                    hl = 0.0
                    oh = 0.0
                    np_ = (
                        1 + int(0.5 + np_scale * ((p_ - p_min) / p_range))
                        if p_ >= p_min
                        else 1
                    )
                    dx, dz = len_stick * np.sin(beta_), -len_stick * np.cos(
                        beta_
                    )
                    plt.arrow(
                        x_,
                        z_,
                        -dx,
                        -dz,
                        head_width=hw,
                        head_length=-0.01,
                        lw=1,
                        shape=shape,
                        overhang=1,
                        length_includes_head=True,
                        head_starts_at_zero=True,
                        ec=aniso_color,
                    )
                    for i_head in list(range(1, np_)):
                        len_head = i_head / (np_)
                        plt.arrow(
                            x_,
                            z_,
                            -dx * len_head,
                            -dz * len_head,
                            head_width=hw,
                            head_length=hl,
                            lw=1,
                            shape=shape,
                            overhang=oh,
                            length_includes_head=True,
                            ec=aniso_color,
                        )
        colorbar_im = axes.imshow(
            aniso_span[0]
            + (aniso_span[1] - aniso_span[0]) * np.arange(9).reshape(3, 3) / 8,
            cmap=color_map,
            extent=(0, 0, 0, 0),
        )

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

        axes.set_aspect(1)
        plt.xlim(-0.05, 1.08)
        if sf is None:
            if gmeq.eta_ <= Rational(1, 2):
                sf = (3, 6.0)
            elif gmeq.eta_ < Rational(3, 4):
                sf = (1.5, 4.3)
            elif gmeq.eta_ >= Rational(3, 2):
                sfy = float(N((9 - 4.3) * (gmeq.eta_ - 0.5) + 4.3)) * 0.3
                sf = (sfy, sfy)
            else:
                sfy = float(N((9 - 1.0) * (gmeq.eta_ - 0.5) + 1.0)) * 0.5
                sf = (sfy, sfy)
                plt.xlim(-0.03, 1.05)
        if y_limits is None:
            plt.ylim(ylim[0] * sf[0], ylim[1] - ylim[0] * sf[1])
        else:
            plt.ylim(*y_limits)

        # Hand-made legend
        class ArrivalTime:
            """
            Not used
            """

            # pass

        class HandlerArrivalTime:
            """
            TBD
            """

            def legend_artist(
                self, legend, orig_handle, fontsize, handlebox
            ) -> mpatches.Arrow:
                """
                TBD
                """
                del legend, orig_handle, fontsize
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                # width, height = handlebox.width, handlebox.height
                patch = mpatches.Arrow(
                    4, 4, 20, 0, width=0, lw=profile_lw, ec="k", fc="k"
                )
                handlebox.add_artist(patch)
                return patch

        class RayPoint:
            """
            Not used
            """

            # pass

        class HandlerRayPoint:
            """
            TBD
            """

            def __init__(self, fc: str = "gray"):
                """
                Constructor method
                """
                super().__init__()
                self.fc = fc

            def legend_artist(
                self, legend, orig_handle, fontsize, handlebox
            ) -> mpatches.Circle:
                """
                TBD
                """
                del legend, orig_handle, fontsize
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                # width, height = handlebox.width, handlebox.height
                patch = mpatches.Circle((15, 4), radius=2.5, ec="k", fc=self.fc)
                handlebox.add_artist(patch)
                return patch

        class RayArrow:
            """
            Not used
            """

            # pass

        class HandlerRayArrow:
            """
            TBD
            """

            def legend_artist(
                self, legend, orig_handle, fontsize, handlebox
            ) -> mpatches.FancyArrow:
                """
                TBD
                """
                del legend, orig_handle, fontsize
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                color_ = aniso_colors[3]
                patch = mpatches.FancyArrow(
                    0,
                    0.5 * height,
                    width,
                    0,
                    length_includes_head=True,
                    head_width=0.75 * height,
                    overhang=0.1,
                    fc=color_,
                    ec=color_,
                )
                handlebox.add_artist(patch)
                return patch

        class NormalStick:
            """
            Not used
            """

            # pass

        class HandlerNormalStick:
            """
            TBD
            """

            def legend_artist(
                self, legend, orig_handle, fontsize, handlebox
            ) -> mpatches.FancyArrow:
                # legend, orig_handle, fontsize,
                """
                TBD
                """
                del legend, orig_handle, fontsize
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                color_ = aniso_colors[5]
                patch = mpatches.FancyArrow(
                    0.8 * height,
                    0.5 * height,
                    0.5,
                    0,
                    length_includes_head=True,
                    head_width=0.65 * height,
                    head_length=0.7 * height,
                    overhang=1,
                    fc=color_,
                    ec=color_,
                )
                handlebox.add_artist(patch)
                for w in [width * 0.25, width * 0.6, width * 0.8]:
                    patch = mpatches.FancyArrow(
                        0.8 * height + 0.5,
                        0.5 * height,
                        w,
                        0,
                        length_includes_head=True,
                        head_width=height if w < width * 0.8 else 0,
                        head_length=0,
                        overhang=0,
                        lw=1.5,
                        fc=color_,
                        ec=color_,
                    )
                    handlebox.add_artist(patch)
                return patch

        legend_fns1 = [ArrivalTime(), RayPoint(), RayArrow(), NormalStick()]
        legend_labels1 = [
            r"$T(\mathbf{r})$",
            r"$\mathbf{r}$",
            r"$\mathbf{v}$",
            r"$\mathbf{\widetilde{p}}$",
        ]
        legend_handlers1 = {
            ArrivalTime: HandlerArrivalTime(),
            RayPoint: HandlerRayPoint(),
            RayArrow: HandlerRayArrow(),
            NormalStick: HandlerNormalStick(),
        }
        legend1 = plt.legend(
            legend_fns1,
            legend_labels1,
            handler_map=legend_handlers1,
            loc="upper left",
        )
        axes.add_artist(legend1)

        divider = make_axes_locatable(axes)
        colorbar_axes = divider.append_axes("right", size="5%", pad=0.2)
        colorbar_axes.set_aspect(0.2)
        colorbar = plt.colorbar(colorbar_im, cax=colorbar_axes)
        colorbar.set_label(
            r"Anisotropy  $\psi = \alpha-\beta+90$  [${\degree}$]",
            rotation=270,
            labelpad=20,
        )

        if do_pub_label:
            plt.text(
                0.93,
                0.15,
                pub_label,
                transform=axes.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
                color="k",
            )

    def profile_beta(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        n_points: int = 26,
        xf_stop: float = 1,
        legend_loc: str = "upper left",
        eta_label_xy: Tuple[float, float] = (0.6, 0.8),
        pub_label_xy: Tuple[float, float] = (0.88, 0.7),
        do_etaxi_label: bool = True,
        do_pub_label: bool = False,
        pub_label: str = "",
    ) -> None:
        r"""
        For a time-invariant (steady-state) topographic profile,
        plot the surface-normal covector angle :math:`\beta` from vertical,
        aka the surface tilt angle from horizontal, as a function of
        dimensionless horizontal distance :math:`x/L_{\mathrm{c}}`.

        This angle is named and calculated in three ways:
        (1) :math:`\beta_p`: directly from the series of
        :math:`\mathbf{\widetilde{p}}` values
        generated by ray ODE integration, since :math:`\tan(\beta_p)=-p_x/p_z`;
        (2) :math:`\beta_{ts}`: by differentiating the computed time-invariant
        topographic profile (itself constructed from the terminations of the
        ensemble of progressively truncated, identical rays);
        (3) :math:`\beta_{vt}`: from the velocity triangle
        :math:`\tan(\beta_{vt}) = \dfrac{v^z+\xi^{\perp}}{v^x}`: generated
        by the geometric constraint of balancing the ray velocities with
        surface-normal velocities.

        Args:
            gmes:
                instance of time invariant solution class
                defined in :mod:`gme.ode.time_invariant`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for
                equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            n_points:
                optional sample rate along each curve
            xf_stop:
                optional x-fraction at which to stop plotting
            legend_loc:
                optional position of legend
            eta_label_xy:
                optional where to plot :math:`\eta` annotation text
            var_label_xy:
                optional where to plot 'var' annotation text
            do_eta_xi:
                optionally do :math:`\eta`, :math:`\xi` annotation?
            pub_label:
                optional 'publication' annotation text
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # eta_label_xy = [0.6,0.8] if eta_label_xy is None else eta_label_xy
        # pub_label_xy = [0.88,0.7] if pub_label_xy is None else pub_label_xy

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max * xf_stop, n_points)
        x_dbl_array = np.linspace(x_min, x_max * xf_stop, n_points * 2 - 1)
        plt.plot(
            x_dbl_array,
            np.rad2deg(gmes.beta_vt_interp(x_dbl_array)),
            "bs",
            ls="-",
            ms=3,
            label=r"$\beta_{vt}$ from $(v^z+\xi^{\!\downarrow\!})/v^x$",
        )
        plt.plot(
            x_array,
            np.rad2deg(gmes.beta_ts_interp(x_array)),
            "go",
            ls="-",
            ms=4,
            label=r"$\beta_{ts}$ from topo gradient",
        )
        plt.plot(
            x_dbl_array,
            np.rad2deg(gmes.beta_p_interp(x_dbl_array)),
            "r",
            ls="-",
            ms=3,
            label=r"$\beta_p$ from $p_x/p_z$",
        )
        #
        # plt.plot(gmes.rx_array,np.rad2deg(gmes.beta_array),
        #          'ks', ls='-', ms=3, label=r'$\beta_p$ from $p_x/p_z$')
        # plt.plot(gmes.rx_array,np.rad2deg(gmes.beta_ts_array),
        #          'bo', ls='-', ms=4,
        # label=r'$\beta_{ts}$ from topo gradient')
        # plt.plot(gmes.rx_array,np.rad2deg(gmes.beta_vt_array),
        #       'r', ls='-', label=r'$\beta_{vt}$
        # from $(v^z+\xi^{\!\downarrow\!})/v^x$')
        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=13)
        plt.ylabel(r"Angle $\beta$  [$\degree$]", fontsize=13)
        plt.grid(True, ls=":")
        plt.ylim(
            1e-9,
        )

        axes = plt.gca()
        plt.legend(loc=legend_loc, fontsize=11, framealpha=0.95)
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
                fontsize=16,
                color="k",
            )

    def profile_beta_error(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        n_points: int = 101,
        eta_label_xy: Tuple[float, float] = (0.5, 0.8),
        xf_stop: float = 0.995,
    ) -> None:
        r"""
        For a time-invariant (steady-state) topographic profile,
        plot the error in the estimated surface-normal covector
        angle :math:`\beta` as a function of dimensionless horizontal
        distance :math:`x/L_{\mathrm{c}}`.

        This error, expressed as a percentage, is defined as
        one of the following normalized differences:
        (1) :math:`100(\beta_{ts}-\beta_{p})/\beta_{p}`, or
        (2) :math:`100(\beta_{vt}-\beta_{p})/\beta_{p}`.
        The error in :math:`\beta_{vt}` can be non-trivial for
        :math:`x/L_{\mathrm{c}} \rightarrow 0` and :math:`\eta < 1`
        if the number of rays used to construct the topographic profile is
        insufficient.

        Args:
            gmes:
                instance of time invariant solution class
                defined in :mod:`gme.ode.time_invariant`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for
                  equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            n_points:
                optional sample rate along each curve
            eta_label_xy:
                optional where to plot :math:`\eta` annotation text
            xf_stop:
                optional x-fraction at which to stop plotting
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # eta_label_xy = [0.5,0.8] if eta_label_xy is None else eta_label_xy
        # pub_label_xy = [0.5,0.2] if pub_label_xy is None else pub_label_xy

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max * xf_stop, n_points)
        plt.plot(
            x_array,
            gmes.beta_vt_error_interp(x_array),
            "b",
            ls="-",
            label=r"$\dfrac{\beta_{vt}-\beta_{p}}{\beta_{p}}$",
        )
        plt.plot(
            x_array,
            gmes.beta_ts_error_interp(x_array),
            "g",
            ls="-",
            label=r"$\dfrac{\beta_{ts}-\beta_{p}}{\beta_{p}}$",
        )
        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=13)
        plt.ylabel(r"Error  [%]", fontsize=13)
        plt.grid(True, ls=":")

        axes = plt.gca()
        ylim = axes.get_ylim()
        plt.ylim(ylim[0] * 1.0, ylim[1] * 1.3)
        plt.legend(loc="upper left", fontsize=9, framealpha=0.95)
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

    def profile_xi(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        xf_stop: float = 1,
        n_points: int = 201,
        pub_label_xy: Tuple[float, float] = (0.5, 0.2),
        eta_label_xy: Tuple[float, float] = (0.5, 0.5),
        var_label_xy: Tuple[float, float] = (0.8, 0.5),
        do_etaxi_label=True,
        do_pub_label=False,
        pub_label: str = "(a)",
        xi_norm: Optional[float] = None,
    ) -> None:
        r"""
        Plot surface-normal erosion speed :math:`\xi^{\perp}`
        along a time-invariant profile.

        Args:
            gmes:
                instance of time invariant solution class
                defined in :mod:`gme.ode.time_invariant`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for
                equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            xf_stop:
                optional x-fraction at which to stop plotting
            n_points:
                optional sample rate along each curve
            pub_label_xy:
                optional where to plot 'publication' annotation text
            eta_label_xy:
                optional where to plot :math:`\eta` annotation text
            var_label_xy:
                optional where to plot 'var' annotation text
            do_eta_xi:
                optionally do :math:`\eta`, :math:`\xi` annotation?
            pub_label:
                optional 'publication' annotation text
            xi_norm:
                optional normalization factor
                :math:`\xi^{\rightarrow_{0}}` for :math:`\xi`
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # eta_label_xy = [0.5,0.5] if eta_label_xy is None else eta_label_xy
        # pub_label_xy = [0.5,0.2] if pub_label_xy is None else pub_label_xy
        # var_label_xy = [0.8,0.5] if var_label_xy is None else var_label_xy

        if xi_norm is None:
            xi_norm = 1
            rate_label = r"$\xi^{\!\perp\!}$"
        else:
            xi_norm = float(N(xi_norm))
            rate_label = r"$\xi^{\!\perp\!}/\xi^{\!\rightarrow_{\!\!0}}$  [-]"
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max * xf_stop, n_points)
        u_array = gmes.u_interp(x_array)
        u_from_rdot_array = gmes.u_from_rdot_interp(x_array)
        dashes = [1, 2.0]
        if not do_pub_label:
            plt.plot(
                x_array,
                u_from_rdot_array / xi_norm,
                "g",
                dashes=dashes,
                lw=3,
                label=r"$\xi^{\!\perp\!}(x)$ from $v$",
            )
            plt.plot(
                x_array,
                u_array / xi_norm,
                "b",
                ls="-",
                lw=1.5,
                label=r"$\xi^{\!\perp\!}(x)$ from $1/p$",
            )
        else:
            plt.plot(
                x_array,
                u_from_rdot_array / xi_norm,
                "g",
                dashes=dashes,
                lw=3,
                label=r"from $v$",
            )
            plt.plot(
                x_array,
                u_array / xi_norm,
                "b",
                ls="-",
                lw=1.5,
                label=r"from $1/p$",
            )
        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=13)
        plt.ylabel(r"Normal erosion rate  " + rate_label, fontsize=12)
        plt.grid(True, ls=":")

        axes = plt.gca()
        ylim = plt.ylim()
        axes.set_ylim(-(ylim[1] - ylim[0]) / 20, ylim[1] * 1.05)
        plt.legend(loc="lower left", fontsize=11, framealpha=0.95)
        plt.text(
            *eta_label_xy,
            rf"$\eta={gmeq.eta_}$"
            + r"$\quad\mathsf{Ci}=$"
            + rf"${round(float(deg(Ci.subs(sub))))}\degree$"
            if do_etaxi_label
            else "",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            color="k",
        )
        plt.text(
            *var_label_xy,
            r"$\xi^{\perp}(x)$",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            color="k",
        )
        plt.text(
            *pub_label_xy,
            pub_label if do_pub_label else "",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            color="k",
        )

    def profile_xihorizontal(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        xf_stop: float = 1,
        n_points: int = 201,
        pub_label_xy: Tuple[float, float] = (0.55, 0.81),
        eta_label_xy: Tuple[float, float] = (0.5, 0.2),
        var_label_xy: Tuple[float, float] = (0.85, 0.81),
        do_etaxi_label: bool = True,
        do_pub_label: bool = False,
        pub_label: str = "(d)",
        xi_norm: Optional[float] = None,
    ) -> None:
        r"""
        Plot horizontal erosion speed :math:`\xi^{\rightarrow}`
        along a time-invariant profile.

        Args:
            gmes:
                instance of time invariant solution class
                defined in :mod:`gme.ode.time_invariant`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for
                equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            xf_stop:
                optional x-fraction at which to stop plotting
            n_points:
                optional sample rate along each curve
            pub_label_xy:
                optional where to plot 'publication' annotation text
            eta_label_xy:
                optional where to plot :math:`\eta` annotation text
            var_label_xy:
                optional where to plot 'var' annotation text
            do_eta_xi:
                optionally do :math:`\eta`, :math:`\xi` annotation?
            pub_label:
                optional 'publication' annotation text
            xi_norm:
                optional normalization factor
                :math:`\xi^{\rightarrow_{0}}` for :math:`\xi`
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # eta_label_xy = [0.5,0.2] if eta_label_xy is None else eta_label_xy
        # pub_label_xy = [0.55,0.81] if pub_label_xy is None else pub_label_xy
        # var_label_xy = [0.85,0.81] if var_label_xy is None else var_label_xy

        if xi_norm is None:
            xi_norm = 1
            rate_label = r"$\xi^{\!\rightarrow}$"
        else:
            xi_norm = float(N(xi_norm))
            rate_label = (
                r"$\xi^{\!\rightarrow\!\!}/\xi^{\!\rightarrow_{\!\!0}}$  [-]"
            )
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max * xf_stop, n_points)
        uhorizontal_p_array = gmes.uhorizontal_p_interp(x_array) / xi_norm
        uhorizontal_v_array = gmes.uhorizontal_v_interp(x_array) / xi_norm
        dashes = [1, 2.0]
        if not do_pub_label:
            plt.plot(
                x_array,
                uhorizontal_v_array,
                "g",
                dashes=dashes,
                lw=3,
                label=r"$\xi^{\!\rightarrow\!}(x)$ from $v$",
            )
            plt.plot(
                x_array,
                uhorizontal_p_array,
                "b",
                ls="-",
                lw=1.5,
                label=r"$\xi^{\!\rightarrow\!}(x)$ from $1/p$",
            )
        else:
            plt.plot(
                x_array,
                uhorizontal_p_array,
                "g",
                dashes=dashes,
                lw=3,
                label=r"from $v$",
            )
            plt.plot(
                x_array,
                uhorizontal_v_array,
                "b",
                ls="-",
                lw=1.5,
                label=r"from $1/p$",
            )
        axes = plt.gca()
        ylim = plt.ylim()
        axes.set_ylim(-(ylim[1] - ylim[0]) / 20, ylim[1])
        plt.grid(True, ls=":")

        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=13)
        plt.ylabel(r"Horiz erosion rate  " + rate_label, fontsize=12)
        # axes.set_ylim(ylim[0]*1.1,-0)
        plt.legend(loc="lower left", fontsize=11, framealpha=0.95)
        plt.text(
            *eta_label_xy,
            rf"$\eta={gmeq.eta_}$"
            + r"$\quad\mathsf{Ci}=$"
            + rf"${round(float(deg(Ci.subs(sub))))}\degree$"
            if do_etaxi_label
            else "",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            color="k",
        )
        plt.text(
            *var_label_xy,
            r"$\xi^{\rightarrow}(x)$",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            color="k",
        )
        plt.text(
            *pub_label_xy,
            pub_label if do_pub_label else "",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            color="k",
        )

    def profile_xivertical(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        xf_stop: float = 1,
        n_points: int = 201,
        y_limits: Optional[Tuple[float, float]] = None,
        pub_label_xy: Tuple[float, float] = (0.5, 0.81),
        eta_label_xy: Tuple[float, float] = (0.5, 0.2),
        var_label_xy: Tuple[float, float] = (0.85, 0.81),
        do_etaxi_label: bool = True,
        do_pub_label: bool = False,
        pub_label: str = "(e)",
        xi_norm: Optional[float] = None,
    ) -> None:
        r"""
        Plot vertical erosion speed
        :math:`\xi^{\downarrow}` along a time-invariant profile.

        Args:
            gmes:
                instance of time invariant solution class
                defined in :mod:`gme.ode.time_invariant`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for
                equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            xf_stop:
                optional x-fraction at which to stop plotting
            n_points:
                optional sample rate along each curve
            pub_label_xy:
                optional where to plot 'publication' annotation text
            eta_label_xy:
                optional where to plot :math:`\eta` annotation text
            var_label_xy:
                optional where to plot 'var' annotation text
            do_eta_xi:
                optionally do :math:`\eta`, :math:`\xi` annotation?
            pub_label:
                optional 'publication' annotation text
            xi_norm: optional: normalization factor
                :math:`\xi^{\rightarrow_{0}}` for :math:`\xi`
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # eta_label_xy = [0.5,0.2] if eta_label_xy is None else eta_label_xy
        # pub_label_xy = [0.5,0.81] if pub_label_xy is None else pub_label_xy
        # var_label_xy = [0.85,0.81] if var_label_xy is None else var_label_xy

        if xi_norm is None:
            xi_norm = 1
            rate_label = r"$\xi^{\!\downarrow}$"
        else:
            xi_norm = float(N(xi_norm))
            rate_label = (
                r"$\xi^{\!\downarrow}\!\!/\xi^{\!\rightarrow_{\!\!0}}$  [-]"
            )
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max * xf_stop, n_points)
        xiv_p_array = gmes.xiv_p_interp(x_array) / xi_norm
        xiv_v_array = gmes.xiv_v_interp(x_array) / xi_norm
        if not do_pub_label:
            plt.plot(
                x_array,
                xiv_v_array,
                "g",
                ls="-",
                label=r"$\xi^{\!\downarrow\!}(x)$ from $v$",
            )
            plt.plot(
                x_array,
                xiv_p_array,
                "b",
                ls="-",
                label=r"$\xi^{\!\downarrow\!}(x)$ from $1/p$",
            )
        else:
            plt.plot(x_array, xiv_v_array, "g", ls=":", lw=3, label=r"from $v$")
            plt.plot(x_array, xiv_p_array, "b", ls="-", label=r"from $1/p$")
        axes = plt.gca()
        # ylim = plt.ylim()
        plt.grid(True, ls=":")

        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=13)
        plt.ylabel(r"Vertical erosion rate  " + rate_label, fontsize=12)
        if y_limits is not None:
            axes.set_ylim(*y_limits)
        else:
            xiv_mean = np.mean(xiv_p_array)
            xiv_deviation = (
                np.max(
                    [
                        xiv_mean - np.min(xiv_p_array),
                        np.max(xiv_p_array) - xiv_mean,
                    ]
                )
                * 1.1
            )
            axes.set_ylim([xiv_mean - xiv_deviation, xiv_mean + xiv_deviation])
        plt.legend(loc="upper left", fontsize=11, framealpha=0.95)
        plt.text(
            *eta_label_xy,
            rf"$\eta={gmeq.eta_}$"
            + r"$\quad\mathsf{Ci}=$"
            + rf"${round(float(deg(Ci.subs(sub))))}\degree$"
            if do_etaxi_label
            else "",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            color="k",
        )
        plt.text(
            *var_label_xy,
            r"$\xi^{\downarrow}(x)$",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            color="k",
        )
        plt.text(
            *pub_label_xy,
            pub_label if do_pub_label else "",
            transform=axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            color="k",
        )

    def profile_ensemble(
        self,
        gmes: TimeInvariantSolution,
        pr_choices: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        aspect: Optional[float] = None,
        do_direct: bool = False,
    ) -> None:
        r"""
        Plot set of time-invariant profiles for a selection of values of
        :math:`\mathsf{Ci}` and :math:`\eta`.

        Args:
            gmes:
                instance of time invariant solution class
                defined in :mod:`~.ode.time_invariant`
            pr_choices:
                dictionary of model parameter values to be used for
                equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        def plot_h_profile(eta_, Ci_, idx_, n_, lw=2, dashing="-"):
            sub_ = pr_choices[(eta_, Ci_)]
            mu_ = sub_[mu]
            gmes_ = gmes[(eta_, Ci_)]
            h_x_array = gmes_.h_x_direct_array if do_direct else gmes_.h_x_array
            h_z_array = gmes_.h_z_direct_array if do_direct else gmes_.h_z_array
            Ci_label = (
                rf"{deg(Ci_)}" if deg(Ci_) >= 1 else rf"{deg(Ci_).n():0.1}"
            )
            color_ = self.mycolors(
                idx_, 1, n_, do_smooth=False, cmap_choice="brg"
            )
            plt.plot(
                h_x_array,
                (h_z_array - h_z_array[0]),
                dashing,
                lw=lw,
                color=color_,
                label=rf"$\eta=${eta_}, "
                + rf"$\mu=${mu_}, "
                + r"$\mathsf{Ci}=$"
                + Ci_label
                + r"$\degree$",
            )

        def make_eta_Ci_list(eta_choice):
            eta_Ci_list = [
                (eta_, Ci_) for (eta_, Ci_) in gmes if eta_ == eta_choice
            ]
            return sorted(
                eta_Ci_list, key=lambda eta_Ci_: eta_Ci_[1], reverse=True
            )

        eta_Ci_list_1p5 = make_eta_Ci_list(Rational(3, 2))
        eta_Ci_list_0p5 = make_eta_Ci_list(Rational(1, 2))
        eta_Ci_list_0p25 = make_eta_Ci_list(Rational(1, 4))

        for (eta_Ci_list_, dashing_, lw_) in [
            (eta_Ci_list_1p5, "-", 2),
            (eta_Ci_list_0p5, ":", 3),
            (eta_Ci_list_0p25, "-", 2),
        ]:
            n_ = len(eta_Ci_list_)
            for i_, (eta_, Ci_) in enumerate(eta_Ci_list_):
                plot_h_profile(eta_, Ci_, i_, n_, lw=lw_, dashing=dashing_)

        axes.set_aspect(1 if aspect is None else aspect)
        plt.grid(True, ls=":")
        plt.xlim(-0.02, 1.02)
        plt.xlabel(r"Distance, $x/L_{\mathrm{c}}$  [-]", fontsize=16)
        plt.ylabel(r"Elevation, $z/L_{\mathrm{c}}$  [-]", fontsize=16)

        plt.legend(loc="upper left", fontsize=11, framealpha=0.95)


#

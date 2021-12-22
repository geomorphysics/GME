"""
---------------------------------------------------------------------

Visualization of ray properties along a profile.

---------------------------------------------------------------------

Requires Python packages/modules:
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
# Library
import warnings
from typing import Tuple, Dict, Optional

# NumPy
import numpy as np

# SymPy
from sympy import deg, tan

# MatPlotLib
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

# GME
from gme.core.symbols import Ci, xiv_0, xih_0
from gme.core.equations import Equations
from gme.ode.single_ray import SingleRaySolution
from gme.ode.time_invariant import TimeInvariantSolution
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['RayProfiles']


class RayProfiles(Graphing):
    """
    Visualization of ray properties as a function of distance along
    the profile.

    Extends :class:`gme.plot.base.Graphing`.
    """

    def profile_ray(
        self,
        gmes: SingleRaySolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        y_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        n_points: int = 101,
        aspect: Optional[float] = None,
        do_schematic: bool = False,
        do_ndim: bool = True,
        do_simple: bool = False,
        do_t_sampling: bool = True,
        do_pub_label: bool = False,
        pub_label: str = '',
        pub_label_xy: Tuple[float, float] = (0.15, 0.50),
        do_etaxi_label: bool = True,
        eta_label_xy=None,
    ) -> None:
        r"""
        Plot an erosion ray solution of Hamilton's equations.

        Args:
            gmes:
                instance of single-ray solution class
                defined in :mod:`gme.ode.single_ray`
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
            do_direct:
                plot directly integrated ray trajectory
                (from :math:`\mathbf{\widetilde{p}}` values) as a solid curve
            do_schematic:
                optionally plot in more schematic form for expository purposes?
            do_simple:
                optionally simplify?

        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes: Axes = plt.gca()
        # pub_label_xy = [0.15,0.50] if pub_label_xy is None else pub_label_xy

        t_array = gmes.t_array
        rx_array = gmes.rx_array
        # rz_array = gmes.rz_array
        # v_array = n.sqrt(gmes.rdotx_array**2 + gmes.rdotz_array**2)

        if do_t_sampling:
            t_begin, t_end = t_array[0], t_array[-1]
            t_rsmpld_array = np.linspace(t_begin, t_end, n_points)
        else:
            x_rsmpld_array = np.linspace(rx_array[0], rx_array[-1], n_points)
            t_rsmpld_array = gmes.t_interp_x(x_rsmpld_array)
        rx_rsmpld_array = gmes.rx_interp_t(t_rsmpld_array)
        rz_rsmpld_array = gmes.rz_interp_t(t_rsmpld_array)
        v_rsmpld_array = np.sqrt(gmes.rdotx_interp_t(t_rsmpld_array)**2
                                 + gmes.rdotz_interp_t(t_rsmpld_array)**2)

        # Plot arrow-annotated rays
        xi_vh_ratio = float(-tan(Ci).subs(sub)) if do_ndim \
            else float(-(xiv_0/xih_0).subs(sub))
        self.draw_rays_with_arrows_simple(axes,
                                          sub, xi_vh_ratio,
                                          t_rsmpld_array,
                                          rx_rsmpld_array,
                                          rz_rsmpld_array,
                                          v_rsmpld_array,
                                          n_rays=1,
                                          n_t=None,
                                          ls='-',
                                          sf=1,
                                          do_one_ray=True,
                                          color='0.5')

        axes.set_aspect(aspect if aspect is not None else 1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=13)
        if eta_label_xy is None:
            eta_label_xy = (0.92, 0.15)
        if not do_schematic and not do_simple:
            if do_etaxi_label:
                plt.text(*eta_label_xy,
                         rf'$\eta={gmeq.eta_}$'+r'$\quad\mathsf{Ci}=$'
                         + rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                         transform=axes.transAxes,
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=13,
                         color='k')
            if do_pub_label:
                plt.text(*pub_label_xy,
                         pub_label,
                         transform=axes.transAxes,
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=16,
                         color='k')
        if y_limits is not None:
            plt.ylim(*y_limits)

    def profile_h_rays(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        x_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        y_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        n_points: int = 101,
        do_direct: bool = True,
        n_rays: int = 4,
        do_schematic: bool = False,
        do_legend: bool = True,
        do_fault_bdry: bool = False,
        do_compute_xivh_ratio: bool = False,
        do_one_ray: bool = False,
        do_t_sampling: bool = True,
        do_pub_label: bool = False,
        pub_label: str = '',
        pub_label_xy: Tuple[float, float] = (0.93, 0.33),
        do_etaxi_label: bool = True,
        eta_label_xy: Tuple[float, float] = (0.5, 0.8)
    ) -> None:
        r"""
        Plot a set of erosion rays for a time-invariant
        topographic profile solution of Hamilton's equations.

        Hamilton's equations are integrated once
        (from the left boundary to the divide)
        and a time-invariant profile is constructed by repeating
        the ray trajectory, with a suitable truncation and vertical
        initial offset, multiple times at the left boundary:
        the set of end of truncated rays constitutes the 'steady-state'
        topographic profile.
        The point of truncation of each trajectory corresponds to the
        effective time lag imposed by the choice of vertical initial
        offset (which is controlled by the vertical slip rate).

        Visualization of this profile includes: (i) plotting a subsampling
        of the terminated points of the ray truncations;
        (ii) plotting a continuous curve generated by integrating the surface
        gradient implied by the erosion-front normal
        covector values :math:`\mathbf{\widetilde{p}}` values generated by
        solving Hamilton's equations.

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
            do_direct:
                plot directly integrated ray trajectory
                (from :math:`\mathbf{\widetilde{p}}` values) as a solid curve
            ray_subsetting:
                optional ray subsampling rate
                (typically far more rays are computed than should be plotted)
            do_schematic:
                optionally plot in more schematic form for expository purposes?
            do_simple:
                optionally simplify?

        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # pub_label_xy = [0.93,0.33] if pub_label_xy is None else pub_label_xy
        # eta_label_xy = [0.5,0.8] if eta_label_xy is None else eta_label_xy
        axes: Axes = plt.gca()

        t_array = gmes.t_array  # [::ray_subsetting]
        rx_array = gmes.rx_array  # [::ray_subsetting]
        # rz_array = gmes.rz_array #[::ray_subsetting]

        if do_t_sampling:
            (t_begin, t_end) = t_array[0], t_array[-1]
            t_rsmpld_array = np.linspace(t_begin, t_end, n_points)
        else:
            x_rsmpld_array = np.linspace(rx_array[0], rx_array[-1], n_points)
            t_rsmpld_array = gmes.t_interp_x(x_rsmpld_array)
        rx_rsmpld_array = gmes.rx_interp_t(t_rsmpld_array)
        rz_rsmpld_array = gmes.rz_interp_t(t_rsmpld_array)

        # Plot arrow-annotated rays
        xi_vh_ratio = float(-tan(Ci).subs(sub)) if do_compute_xivh_ratio \
            else float(-(xiv_0/xih_0).subs(sub))
        self.draw_rays_with_arrows_simple(axes,
                                          sub,
                                          xi_vh_ratio,
                                          t_rsmpld_array,
                                          rx_rsmpld_array,
                                          rz_rsmpld_array,
                                          n_rays=n_rays,
                                          n_t=None,
                                          ls='-' if do_schematic else '-',
                                          sf=0.5 if do_schematic else 1,
                                          do_one_ray=do_one_ray)

        if do_schematic:
            # For schematic fig, also plot mirror-image topo profile
            #                     on opposite side of drainage divide
            self.draw_rays_with_arrows_simple(axes,
                                              sub,
                                              xi_vh_ratio,
                                              t_rsmpld_array,
                                              2-rx_rsmpld_array,
                                              rz_rsmpld_array,
                                              n_rays=n_rays,
                                              n_t=None,
                                              ls='-' if do_schematic else '-',
                                              sf=0.5 if do_schematic else 1,
                                              do_labels=False)

        # # Markers = topo profile from ray terminations
        # if not do_schematic and not do_one_ray:
        #     plt.plot( gmes.x_array[::profile_subsetting],
        #                             gmes.h_array[::profile_subsetting],
        #                 'k'+('s' if do_profile_points else '-'),
        #            ms=3, label=r'$T(\mathbf{r})$ from rays $\mathbf{r}(t)$' )

        # Solid line = topo profile from direct integration of gradient array
        if (do_direct or do_schematic) and not do_one_ray:
            plt.plot(gmes.h_x_array,
                     gmes.h_z_array-gmes.h_z_array[0],
                     'k',
                     label=r'$T(\mathbf{r})$' if do_schematic
                     else r'$T(\mathbf{r})$')
            if do_schematic:
                plt.plot(gmes.h_x_array,
                         gmes.h_z_array-gmes.h_z_array[0]+0.26,
                         '0.75',
                         lw=1,
                         ls='--')
                plt.plot(gmes.h_x_array,
                         gmes.h_z_array-gmes.h_z_array[0]+0.13,
                         '0.5',
                         lw=1,
                         ls='--')
                plt.plot(2-gmes.h_x_array,
                         gmes.h_z_array-gmes.h_z_array[0],
                         'k')
                plt.plot(2-gmes.h_x_array,
                         gmes.h_z_array-gmes.h_z_array[0]+0.13,
                         '0.5',
                         lw=1,
                         ls='--')
                plt.plot(2-gmes.h_x_array,
                         gmes.h_z_array-gmes.h_z_array[0]+0.26,
                         '0.75',
                         lw=1,
                         ls='--')

        axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]',
                   fontsize=13 if do_schematic else 16)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]',
                   fontsize=13 if do_schematic else 16)
        if not do_schematic and not do_one_ray and do_legend:
            plt.legend(loc='upper right' if do_schematic else (0.065, 0.45),
                       fontsize=9 if do_schematic else 11,
                       framealpha=0.95)
        if not do_schematic:
            if do_etaxi_label:
                plt.text(*eta_label_xy,
                         rf'$\eta={gmeq.eta_}$'
                         + r'$\quad\mathsf{Ci}=$'
                         + rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                         transform=axes.transAxes,
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=16,
                         color='k')
            if do_pub_label:
                plt.text(*pub_label_xy,
                         pub_label,
                         transform=axes.transAxes,
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=16,
                         color='k')
        if x_limits is not None:
            plt.xlim(*x_limits)
        if y_limits is not None:
            plt.ylim(*y_limits)

        if do_schematic:
            for x_, align_ in ((0.03, 'center'), (1.97, 'center')):
                plt.text(x_,
                         0.45,
                         'rays initiated',
                         rotation=0,
                         horizontalalignment=align_,
                         verticalalignment='center',
                         fontsize=12,
                         color='r')
            plt.text(1,
                     0.53,
                     'rays annihilated',
                     rotation=0,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=12,
                     color='0.25')
            for x_, align_ in ((-0.03, 'right'), (2.03, 'left')):
                plt.text(x_,
                         0.17,
                         'fault slip b.c.' if do_fault_bdry
                         else 'const. erosion rate',
                         rotation=90,
                         horizontalalignment=align_,
                         verticalalignment='center',
                         fontsize=12,
                         color='r',
                         alpha=0.7)
            plt.text(0.46,
                     0.38,
                     r'surface isochrone $T(\mathbf{r})=\mathrm{past}$',
                     rotation=12,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10,
                     color='0.2')
            plt.text(0.52,
                     0.05,
                     r'surface isochrone $T(\mathbf{r})=\mathrm{now}$',
                     rotation=12,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10,
                     color='k')
            for (x_, y_, dx_, dy_, shape_) in ((0, 0.4, 0, -0.15, 'left'),
                                               (0, 0.25, 0, -0.15, 'left'),
                                               (0, 0.1, 0, -0.15, 'left'),
                                               (2, 0.4, 0, -0.15, 'right'),
                                               (2, 0.25, 0, -0.15, 'right'),
                                               (2, 0.1, 0, -0.15, 'right')):
                plt.arrow(x_,
                          y_,
                          dx_,
                          dy_,
                          head_length=0.04,
                          head_width=0.03,
                          length_includes_head=True,
                          shape=shape_,
                          facecolor='r',
                          edgecolor='r')

    def profile_h(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        y_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        do_legend: bool = True,
        do_profile_points: bool = True,
        profile_subsetting: int = 5,
        do_pub_label: bool = False,
        pub_label: str = '',
        pub_label_xy: Tuple[float, float] = (0.93, 0.33),
        do_etaxi_label=True,
        eta_label_xy: Tuple[float, float] = None,
    ) -> None:
        r"""
        Plot a time-invariant topographic profile solution of
        Hamilton's equations.

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
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # pub_label_xy = [0.93,0.33] if pub_label_xy is None else pub_label_xy
        axes: Axes = plt.gca()

        # t_array  = gmes.t_array #[::ray_subsetting]
        # rx_array = gmes.rx_array #[::ray_subsetting]
        # rz_array = gmes.rz_array #[::ray_subsetting]
        # print()

        # if do_t_sampling:
        #     t_begin, t_end = t_array[0], t_array[-1]
        #     # t_rsmpld_array = np.linspace(t_begin, t_end, n_points)
        # else:
        #     x_rsmpld_array = np.linspace(rx_array[0], rx_array[-1], n_points)
        #     t_rsmpld_array = gmes.t_interp_x(x_rsmpld_array)
        # # rx_rsmpld_array = gmes.rx_interp_t(t_rsmpld_array)
        # # rz_rsmpld_array = gmes.rz_interp_t(t_rsmpld_array)

        # Markers = topo profile from ray terminations
        plt.plot(gmes.x_array[::profile_subsetting],
                 gmes.h_array[::profile_subsetting],
                 'k'+('s' if do_profile_points else '-'),
                 ms=3,
                 label=r'$T(\mathbf{r})$ from rays $\mathbf{r}(t)$')

        # Solid line = topo profile from direct integration of gradient array
        plt.plot(gmes.h_x_array,
                 (gmes.h_z_array-gmes.h_z_array[0]),
                 'k',
                 label=r'$T(\mathbf{r})$ by integration')
        axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=15)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=15)
        if do_legend:
            plt.legend(loc=(0.38, 0.75), fontsize=11, framealpha=0.95)
        if eta_label_xy is None:
            eta_label_xy = (0.92, 0.15)
        if do_etaxi_label:
            plt.text(*eta_label_xy,
                     rf'$\eta={gmeq.eta_}$'+r'$\quad\mathsf{Ci}=$'
                     + rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                     transform=axes.transAxes,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=16,
                     color='k')
        if do_pub_label:
            plt.text(*pub_label_xy,
                     pub_label,
                     transform=axes.transAxes,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=16,
                     color='k')
        if y_limits is not None:
            plt.ylim(*y_limits)

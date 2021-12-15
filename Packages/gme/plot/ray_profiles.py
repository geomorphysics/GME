"""
---------------------------------------------------------------------

Visualization.

Provides classes to generate a range of graphics for GME visualization.
A base class extends :class:`gmplib.plot.GraphingBase <gmplib.plot.GraphingBase>`
provided by :mod:`GMPLib`; the other classes build on this.
Each is tailored to a particular category of GME problem,
such as single ray tracing or for tracking knickpoints.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`numpy`, :mod:`sympy`
  -  :mod:`matplotlib.pyplot`
  -  :mod:`gme.core.symbols`, :mod:`gme.plot.base`

---------------------------------------------------------------------

"""
import warnings

# Numpy
import numpy as np

# SymPy
from sympy import deg, tan

# MatPlotLib
import matplotlib.pyplot as plt

# GME
from gme.core.symbols import Ci, xiv_0, xih_0
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['RayProfiles']


class RayProfiles(Graphing):
    """
    Subclasses :class:`gme.plot.Graphing <plot.Graphing>`.
    """

    def profile_ray( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                        y_limits=None, eta_label_xy=None, n_points=101, aspect=None,
                        # do_direct=True,
                        do_schematic=False, do_ndim=True,
                        do_simple=False, do_t_sampling=True, do_etaxi_label=True,
                        do_pub_label=False, pub_label='', pub_label_xy=(0.15,0.50) ) \
                                -> None:
        r"""
        Plot a set of erosion rays for a time-invariant
        topographic profile solution of Hamilton's equations.

        Hamilton's equations are integrated once (from the left boundary to the divide)
        and a time-invariant profile is constructed by repeating
        the ray trajectory, with a suitable truncation and
        vertical initial offset, multiple times at the left boundary:
        the set of end of truncated rays constitutes
        the 'steady-state' topographic profile.
        The point of truncation of each trajectory corresponds to
        the effective time lag imposed by the choice of vertical initial
        offset (which is controlled by the vertical slip rate).

        Visualization of this profile includes: (i) plotting a
        subsampling of the terminated points of the ray truncations;
        (ii) plotting a continuous curve generated by integrating the
        surface gradient implied by the erosion-front normal
        covector values :math:`\mathbf{\widetilde{p}}` values generated
        by solving Hamilton's equations.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by
                :meth:`GMPLib create_figure <plot.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in
                    :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            sub (dict): dictionary of model parameter values
                        to be used for equation substitutions
            do_direct (bool): plot directly integrated ray trajectory
                            (from :math:`\mathbf{\widetilde{p}}` values) as a solid curve
            do_schematic (bool):
                optionally plot in more schematic form for expository purposes?
            do_simple (bool): optionally simplify?

        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()
        # pub_label_xy = [0.15,0.50] if pub_label_xy is None else pub_label_xy

        t_array  = gmes.t_array
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
        self.draw_rays_with_arrows_simple( axes, sub, xi_vh_ratio,
                                          t_rsmpld_array,
                                          rx_rsmpld_array, rz_rsmpld_array,
                                          v_rsmpld_array,
                                          n_rays=1, n_t=None,
                                          ls='-', sf=1, do_one_ray=True, color='0.5' )

        axes.set_aspect(aspect if aspect is not None else 1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=13)
        if eta_label_xy is None:
            eta_label_xy = (0.92,0.15)
        if not do_schematic and not do_simple:
            if do_etaxi_label:
                plt.text(*eta_label_xy,
                         rf'$\eta={gmeq.eta_}$'+r'$\quad\mathsf{Ci}=$'
                            +rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                         transform=axes.transAxes,
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=13, color='k')
            if do_pub_label:
                plt.text(*pub_label_xy, pub_label,
                         transform=axes.transAxes,
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=16, color='k')
        if y_limits is not None:
            plt.ylim(*y_limits)

    def profile_h_rays( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                        x_limits=None, y_limits=None, n_points=101,
                        do_direct=True, n_rays=4, #profile_subsetting=5,
                        do_schematic=False, do_legend=True, #do_profile_points=True,
                        do_fault_bdry=False, do_compute_xivh_ratio=False,
                        do_one_ray=False, do_t_sampling=True, do_etaxi_label=True,
                        do_pub_label=False, pub_label='',
                        pub_label_xy=(0.93,0.33), eta_label_xy=(0.5,0.8) ) -> None:
        r"""
        Plot a set of erosion rays for a time-invariant
        topographic profile solution of Hamilton's equations.

        Hamilton's equations are integrated once (from the left boundary to the divide)
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
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by
                :meth:`GMPLib create_figure <plot.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in
                    :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            sub (dict): dictionary of model parameter values to be used
                        for equation substitutions
            do_direct (bool): plot directly integrated ray trajectory
                             (from :math:`\mathbf{\widetilde{p}}` values) as a solid curve
            ray_subsetting (int): optional ray subsampling rate
                            (typically far more rays are computed than should be plotted)
            do_schematic (bool):
                optionally plot in more schematic form for expository purposes?
            do_simple (bool): optionally simplify?

        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # pub_label_xy = [0.93,0.33] if pub_label_xy is None else pub_label_xy
        # eta_label_xy = [0.5,0.8] if eta_label_xy is None else eta_label_xy
        axes = plt.gca()

        t_array  = gmes.t_array #[::ray_subsetting]
        rx_array = gmes.rx_array #[::ray_subsetting]
        # rz_array = gmes.rz_array #[::ray_subsetting]

        if do_t_sampling:
            t_begin, t_end = t_array[0], t_array[-1]
            t_rsmpld_array = np.linspace(t_begin, t_end, n_points)
        else:
            x_rsmpld_array = np.linspace(rx_array[0], rx_array[-1], n_points)
            t_rsmpld_array = gmes.t_interp_x(x_rsmpld_array)
        rx_rsmpld_array = gmes.rx_interp_t(t_rsmpld_array)
        rz_rsmpld_array = gmes.rz_interp_t(t_rsmpld_array)

        # Plot arrow-annotated rays
        xi_vh_ratio = float(-tan(Ci).subs(sub)) if do_compute_xivh_ratio \
                      else float(-(xiv_0/xih_0).subs(sub))
        self.draw_rays_with_arrows_simple( axes, sub, xi_vh_ratio,
                                           t_rsmpld_array,
                                           rx_rsmpld_array, rz_rsmpld_array,
                                           n_rays=n_rays, n_t=None,
                                           ls='-' if do_schematic else '-',
                                           sf=0.5 if do_schematic else 1,
                                           do_one_ray=do_one_ray )

        if do_schematic:
            # For schematic fig, also plot mirror-image topo profile
            #                     on opposite side of drainage divide
            self.draw_rays_with_arrows_simple( axes, sub, xi_vh_ratio,
                                               t_rsmpld_array,
                                               2-rx_rsmpld_array, rz_rsmpld_array,
                                               n_rays=n_rays, n_t=None,
                                               ls='-' if do_schematic else '-',
                                               sf=0.5 if do_schematic else 1,
                                               do_labels=False )

        # # Markers = topo profile from ray terminations
        # if not do_schematic and not do_one_ray:
        #     plt.plot( gmes.x_array[::profile_subsetting],
        #                             gmes.h_array[::profile_subsetting],
        #                 'k'+('s' if do_profile_points else '-'),
        #                 ms=3, label=r'$T(\mathbf{r})$ from rays $\mathbf{r}(t)$' )

        # Solid line = topo profile from direct integration of gradient array
        if (do_direct or do_schematic) and not do_one_ray:
            plt.plot(gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k',
                     label=r'$T(\mathbf{r})$' if do_schematic else r'$T(\mathbf{r})$' )
            if do_schematic:
                plt.plot(  gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.26, '0.75',
                           lw=1, ls='--')
                plt.plot(  gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.13, '0.5',
                           lw=1, ls='--')
                plt.plot(2-gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k')
                plt.plot(2-gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.13, '0.5',
                           lw=1, ls='--')
                plt.plot(2-gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.26, '0.75',
                           lw=1, ls='--')

        axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]',
                   fontsize=13 if do_schematic else 16)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]',
                   fontsize=13 if do_schematic else 16)
        if not do_schematic and not do_one_ray and do_legend:
            plt.legend(loc='upper right' if do_schematic else (0.065,0.45),
                       fontsize=9 if do_schematic else 11,
                       framealpha=0.95)
        if not do_schematic:
            if do_etaxi_label:
                plt.text(*eta_label_xy,
                         rf'$\eta={gmeq.eta_}$'
                            +r'$\quad\mathsf{Ci}=$'
                            +rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                         transform=axes.transAxes,
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=16, color='k')
            if do_pub_label:
                plt.text(*pub_label_xy, pub_label,
                         transform=axes.transAxes,
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=16, color='k')
        if x_limits is not None:
            plt.xlim(*x_limits)
        if y_limits is not None:
            plt.ylim(*y_limits)

        if do_schematic:
            for x_,align_ in [(0.03,'center'), (1.97,'center')]:
                plt.text(x_,0.45, 'rays initiated', #transform=axes.transAxes,
                         rotation=0,
                         horizontalalignment=align_, verticalalignment='center',
                         fontsize=12, color='r')
            plt.text(1,0.53, 'rays annihilated', #transform=axes.transAxes,
                     rotation=0,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, color='0.25')
            for x_,align_ in [(-0.03,'right'), (2.03,'left')]:
                plt.text(x_,0.17,
                         'fault slip b.c.' if do_fault_bdry else 'const. erosion rate',
                         rotation=90,
                         horizontalalignment=align_, verticalalignment='center',
                         fontsize=12, color='r', alpha=0.7)
            plt.text(0.46,0.38, r'surface isochrone $T(\mathbf{r})=\mathrm{past}$',
                     #transform=axes.transAxes,
                     rotation=12,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=10, color='0.2')
            plt.text(0.52,0.05, r'surface isochrone $T(\mathbf{r})=\mathrm{now}$',
                     #transform=axes.transAxes,
                     rotation=12,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=10, color='k')
            for x_,y_,dx_,dy_,shape_ in [(0,0.4, 0,-0.15,'left'),
                                         (0,0.25, 0,-0.15,'left'),
                                         (0,0.1, 0,-0.15,'left'),
                                         (2,0.4, 0,-0.15,'right'),
                                         (2,0.25, 0,-0.15,'right'),
                                         (2,0.1, 0,-0.15,'right')]:
                plt.arrow( x_,y_,dx_,dy_, head_length=0.04, head_width=0.03,
                           length_includes_head=True, shape=shape_,
                           facecolor='r', edgecolor='r' )

    def profile_h( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                    y_limits=None, eta_label_xy=None, # n_points=101,
                    # do_direct=True,
                    do_legend=True, do_profile_points=True,
                    profile_subsetting=5,
                    # do_t_sampling=True,
                    do_etaxi_label=True,
                    do_pub_label=False, pub_label='', pub_label_xy=(0.93,0.33) ) -> None:
        r"""
        Plot a time-invariant topographic profile solution of Hamilton's equations.
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # pub_label_xy = [0.93,0.33] if pub_label_xy is None else pub_label_xy
        axes = plt.gca()

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
        plt.plot( gmes.x_array[::profile_subsetting], gmes.h_array[::profile_subsetting],
                    'k'+('s' if do_profile_points else '-'),
                    ms=3, label=r'$T(\mathbf{r})$ from rays $\mathbf{r}(t)$' )

        # Solid line = topo profile from direct integration of gradient array
        plt.plot(gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k',
                 label=r'$T(\mathbf{r})$ by integration' )
        axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=15)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=15)
        if do_legend:
            plt.legend(loc=(0.38,0.75), fontsize=11, framealpha=0.95)
        if eta_label_xy is None:
            eta_label_xy = (0.92,0.15)
        if do_etaxi_label:
            plt.text(*eta_label_xy,
                     rf'$\eta={gmeq.eta_}$'+r'$\quad\mathsf{Ci}=$'
                        +rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=16, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=16, color='k')
        if y_limits is not None:
            plt.ylim(*y_limits)

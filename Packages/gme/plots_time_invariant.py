"""
---------------------------------------------------------------------

Visualization.

Provides classes to generate a range of graphics for GME visualization.
A base class extends :class:`gmplib.plot_utils.GraphingBase <plot_utils.GraphingBase>`
provided by :mod:`GMPLib`; the other classes build on this.
Each is tailored to a particular category of GME problem,
such as single ray tracing or for tracking knickpoints.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`gmplib.plot_utils <plot_utils>`
  -  :mod:`numpy`
  -  :mod:`sympy`
  -  :mod:`matplotlib.pyplot`
  -  :mod:`matplotlib.ticker`
  -  :mod:`matplotlib.patches`
  -  :mod:`mpl_toolkits.axes_grid1`

Imports symbols from :mod:`.symbols` module.

---------------------------------------------------------------------

"""

# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-few-public-methods, no-self-use
import warnings

# Numpy
import numpy as np

# SymPy
from sympy import N, Rational, deg

# GME
from gme.symbols import rx, x_h, Lc, Ci
from gme.plots_one_ray import OneRayPlots

# MatPlotLib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")

__all__ = ['TimeInvariantPlots']


class TimeInvariantPlots(OneRayPlots):
    """
    Subclasses :class:`gme.plots_oneray.OneRayPlots <plot.OneRayPlots>`.
    """

    # def profiles( self, gmes, pr_choices, name, fig_size=None, dpi=None,
    #                     # y_limits=None, n_points=101,
    #                     # do_direct=True, n_rays=4, profile_subsetting=5,
    #                     do_schematic=False
    #                     # do_legend=True, do_profile_points=True,
    #                     # do_simple=False, do_one_ray=False, do_t_sampling=True, do_etaxi_label=True,
    #                     # do_pub_label=False, pub_label='', pub_label_xy=None, eta_label_xy=None
    #                     ):
    #     r"""
    #     TBD
    #     """
    #     _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
    #     axes = plt.gca()
    #
    #     # pub_label_xy = [0.93,0.33] if pub_label_xy is None else pub_label_xy
    #     # eta_label_xy = [0.5,0.8] if eta_label_xy is None else eta_label_xy
    #
    #     # def plot_h_profile(eta_,Ci_,idx_,n_, lw=2,dashing='-'):
    #     #     sub_ = pr_choices[(eta_,Ci_)]
    #     #     mu_ = sub_[mu]
    #     #     gmes_ = gmes[(eta_,Ci_)]
    #     #     # t_array  = gmes_.t_array
    #     #     # rx_array = gmes_.rx_array
    #     #     # rz_array = gmes_.rz_array
    #     #     Ci_label = rf'{deg(Ci_)}' if deg(Ci_)>=1 else rf'{deg(Ci_).n():0.1}'
    #     #     color_ = self.mycolors(idx_, 1, n_, do_smooth=False, cmap_choice='brg')
    #     #     plt.plot(gmes_.h_x_array,(gmes_.h_z_array-gmes_.h_z_array[0]), dashing, lw=lw, color=color_,
    #     #              label=rf'$\eta=${eta_}, '+rf'$\mu=${mu_}, '+r'$\mathsf{Ci}=$'+Ci_label+r'$\degree$')
    #
    #     def make_eta_Ci_list(Ci_choice):
    #         eta_Ci_list = [(eta_,Ci_) for (eta_,Ci_) in gmes if eta_==Ci_choice]
    #         return sorted(eta_Ci_list, key=lambda Ci_: Ci_[1], reverse=True)
    #
    #     # eta_Ci_list_1p5 = make_eta_Ci_list(Rational(3,2))
    #     # eta_Ci_list_0p5 = make_eta_Ci_list(Rational(1,2))
    #
    #     axes.set_aspect(1)
    #     plt.grid(True, ls=':')
    #     plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13 if do_schematic else 16)
    #     plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=13 if do_schematic else 16)
    #
    #     plt.legend(loc='upper left', fontsize=11, framealpha=0.95)

    def profile_flow_model( self, gmeq, sub, name, fig_size=None, dpi=None,
                            n_points=26, subtitle='', do_subtitling=False, do_extra_annotations=False) -> None:
        """
        Plot the flow component of the erosion model.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
                    instance of time invariant solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            sub (dict): dictionary of model parameter values to be used for equation substitutions
            n_points (int): optional number of points to plot along curve
            subtitle (str): optional sub-title (likely 'ramp' or 'ramp-flat' or similar)
            do_subtitling (bool): annotate with subtitle and eta value
            do_extra_annotations (bool): annotate with hillslope, channel labels
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        x_array = np.linspace(0,float(Lc.subs(sub)),n_points)
        varphi_array = [gmeq.varphi_rx_eqn.rhs.subs(sub).subs({rx:x_}) for x_ in x_array]
        varphi_xh1p0_array = [gmeq.varphi_rx_eqn.rhs.subs({x_h:1}).subs(sub).subs({rx:x_}) for x_ in x_array]
        plt.plot( x_array, varphi_array, '-', color=self.colors[0], label='hillslope-channel model' )
        plt.plot( x_array, varphi_xh1p0_array, '--', color=self.colors[0], label='channel-only model' )

        # axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Dimensionless horizontal distance, $x/L_{\mathrm{c}}$  [-]')
        plt.ylabel(r'$\varphi(x)$  [-]')
        if do_subtitling:
            plt.text(0.1,0.15, rf'$\eta={gmeq.eta}$', transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center', fontsize=12, color='k')
            plt.text(0.05,0.22, subtitle, transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center', fontsize=12, color='k')
        if do_extra_annotations:
            plt.text(0.4,0.45, 'channel', transform=axes.transAxes,
                     rotation=-43, horizontalalignment='center', verticalalignment='center', fontsize=12, color='0.2')
            plt.text(0.83,0.16, 'hillslope', transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center', fontsize=11, color='0.2')

        y_limits = axes.get_ylim()
        x_h_ = float(x_h.subs(sub))
        varphi_h_ = float(gmeq.varphi_rx_eqn.rhs.subs({rx:x_h}).subs(sub))
        plt.plot([x_h_,x_h_],[varphi_h_-30,varphi_h_+70],'b:')
        plt.text(x_h_,varphi_h_+77, r'$x_h/L_{\mathrm{c}}$', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='bottom', fontsize=12, color='b')

        plt.legend(loc='upper right', fontsize=11, framealpha=0.95)
        plt.xlim(None,1.05)
        plt.ylim(*y_limits)

    def profile_aniso(self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                      y_limits=None, eta_label_xy=None, v_scale=0.4, v_exponent=1,
                      sf=None, n_points=51, n_arrows=26, xf_stop=0.995,
                      do_pub_label=False, pub_label='' ) -> None:
        r"""
        Plot time-invariant profile annotated with ray vectors, normal-slowness covector herringbones,
        and colorized for anisotropy (defined as the difference :math:`(\alpha-\beta)`).

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
                instance of time invariant solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                GME model equations class instance defined in :mod:`~.equations`
            sub (dict):
                dictionary of model parameter values to be used for equation substitutions
            sf ([float,float]):
                optional scale factor(s) for vertical axes (bottom and top)
            n_points (int):
                sample rate along each curve
            n_arrows (int):
                number of :math:`\mathbf{r}` vector arrows and :math:`\mathbf{\widetilde{p}}` covector herringbones to plot
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        eta_label_xy = [0.5,0.8] if eta_label_xy is None else eta_label_xy
        x_array = np.linspace(0,1*xf_stop,n_points)
        h_array = gmes.h_interp(x_array)

        profile_lw = 1
        # Solid line = topo profile from direct integration of gradient array
        plt.plot(gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k', lw=profile_lw,
                 label=rf'$h(x)\quad\eta={gmeq.eta}$')
        plt.plot(x_array, h_array, 'o', mec='k', mfc='gray', ms=3, fillstyle='full', markeredgewidth=0.5)

        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=16)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=16)
        plt.grid(True, ls=':')

        axes = plt.gca()
        ylim = axes.get_ylim()

        cmap_choice = 'viridis_r'
        #     cmap_choice = 'plasma_r'
        #     cmap_choice = 'magma_r'
        #     cmap_choice = 'inferno_r'
        #     cmap_choice = 'cividis_r'
        shape='full'

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min,x_max*xf_stop,n_arrows)
        h_array = gmes.h_interp(x_array)
        beta_array = gmes.beta_p_interp(x_array)
        alpha_array = gmes.alpha_interp(x_array)
        rdot_array = gmes.rdot_interp(x_array)
        p_array = gmes.p_interp(x_array)
        aniso_array = np.rad2deg(alpha_array-beta_array)

        aniso_span = np.array( (10*np.floor(min(aniso_array)/10), 90) )
        color_map = cm.get_cmap(cmap_choice)
        # stretch_aniso_array = (((aniso_array-min(aniso_array))/(max(aniso_array)-min(aniso_array)))**30) \
        #                             *(max(aniso_array)-min(aniso_array))+min(aniso_array)
        aniso_colors = [color_map((aniso_-aniso_span[0])/(aniso_span[1]-aniso_span[0]))
                         for aniso_ in aniso_array]
        for rp_idx in [0,1]:
            # Fix the range of p values to be represented: smaller values will be clipped
            p_range_max = 20
            p_max = max(p_array)
            p_min = min(p_array)
            p_min = p_max/p_range_max if p_max/p_min>p_range_max else p_min
            p_range = p_max-p_min
            np_scale = 15
            # Fix the range of rdot values to be represented: smaller values will be clipped
            rdot_range_max = 20
            rdot_max, rdot_min = max(rdot_array), min(rdot_array)
            rdot_min = rdot_max/rdot_range_max if rdot_max/rdot_min>rdot_range_max else rdot_min
            rdot_range = rdot_max-rdot_min
            nrdot_scale = 9
            for (x_,z_, aniso_color, alpha_, beta_, rdot_, p_) \
                    in zip(x_array, h_array,
                           aniso_colors,
                           alpha_array, beta_array,
                           rdot_array, p_array):
                if rp_idx==0:
                    # Ray vector
                    hw=0.01
                    hl=0.02
                    oh=0.1
                    len_arrow = (( v_scale*((rdot_-rdot_min)**v_exponent/rdot_range)
                                  if rdot_>=rdot_min else 0 ) + v_scale/nrdot_scale)
                    dx, dz = len_arrow*np.cos(alpha_-np.pi/2), len_arrow*np.sin(alpha_-np.pi/2)
                    plt.arrow(x_, z_, dx, dz,
                                head_width=hw, head_length=hl, lw=1,
                                shape=shape, overhang=oh,
                                length_includes_head=True,
                                ec=aniso_color,
                                fc=aniso_color)
                else:
                    # Slowness covector
                    len_stick = 0.08
                    hw=0.015
                    hl=0.0
                    oh=0.0
                    np_ = 1+int(0.5+np_scale*((p_-p_min)/p_range)) if p_>=p_min else 1
                    dx, dz = len_stick*np.sin(beta_), -len_stick*np.cos(beta_)
                    plt.arrow(x_, z_, -dx, -dz,
                                head_width=hw, head_length=-0.01, lw=1,
                                shape=shape, overhang=1,
                                length_includes_head=True,
                                head_starts_at_zero=True,
                                ec=aniso_color)
                    for i_head in list(range(1,np_)):
                        len_head = i_head/(np_)
                        plt.arrow(x_, z_, -dx*len_head, -dz*len_head,
                                head_width=hw, head_length=hl, lw=1,
                                shape=shape, overhang=oh,
                                length_includes_head=True,
                                ec=aniso_color)
        colorbar_im = axes.imshow(aniso_span[0]+(aniso_span[1]-aniso_span[0])*np.arange(9).reshape(3,3)/8,
                                  cmap=color_map, extent=(0,0,0,0))


        plt.text(*eta_label_xy,
                 rf'$\eta={gmeq.eta}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                 transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

        axes.set_aspect(1)
        plt.xlim(-0.05,1.08)
        if sf is None:
            if gmeq.eta <= Rational(1,2):
                sf = (3,6.)
            elif gmeq.eta < Rational(3,4):
                sf = (1.5,4.3)
            elif gmeq.eta >= Rational(3,2):
                sfy = float(N((9-4.3)*(gmeq.eta-0.5)+4.3))*0.3
                sf = (sfy,sfy)
            else:
                sfy = float(N((9-1.0)*(gmeq.eta-0.5)+1.0))*0.5
                sf = (sfy,sfy)
                plt.xlim(-0.03,1.05)
        if y_limits is None:
            plt.ylim(ylim[0]*sf[0],ylim[1]-ylim[0]*sf[1])
        else:
            plt.ylim(*y_limits)

        # Hand-made legend
        class ArrivalTime:
            """
            TBD
            """
            # pass

        class HandlerArrivalTime:
            """
            TBD
            """
            def legend_artist(self, handlebox) -> mpatches.Arrow:
                """
                TBD
                """
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                # width, height = handlebox.width, handlebox.height
                patch = mpatches.Arrow(4,4,20,0, width=0, lw=profile_lw, ec='k', fc='k')
                handlebox.add_artist(patch)
                return patch

        class RayPoint:
            """
            TBD
            """
            # pass

        class HandlerRayPoint:
            """
            TBD
            """
            def __init__(self,fc='gray'):
                """
                TBD
                """
                super().__init__()
                self.fc = fc

            def legend_artist(self, handlebox) -> mpatches.Circle:
                """
                TBD
                """
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                # width, height = handlebox.width, handlebox.height
                patch = mpatches.Circle((15, 4), radius=2.5, ec='k', fc=self.fc)
                handlebox.add_artist(patch)
                return patch

        class RayArrow:
            """
            TBD
            """
            # pass

        class HandlerRayArrow:
            """
            TBD
            """
            def legend_artist(self, handlebox) -> mpatches.FancyArrow:
                """
                TBD
                """
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                color_ = aniso_colors[3]
                patch = mpatches.FancyArrow(0, 0.5*height, width, 0,
                                            length_includes_head=True,
                                            head_width=0.75*height,
                                            overhang=0.1,
                                            fc=color_, ec=color_)
                handlebox.add_artist(patch)
                return patch

        class NormalStick:
            """
            TBD
            """
            # pass

        class HandlerNormalStick:
            """
            TBD
            """
            def legend_artist(self, handlebox) -> mpatches.FancyArrow:
                # legend, orig_handle, fontsize,
                """
                TBD
                """
                # x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                color_ = aniso_colors[5]
                patch = mpatches.FancyArrow(0.8*height, 0.5*height, 0.5, 0,
                                            length_includes_head=True,
                                            head_width=0.65*height,
                                            head_length=0.7*height,
                                            overhang=1,
                                            fc=color_, ec=color_)
                handlebox.add_artist(patch)
                for w in [width*0.25,width*0.6,width*0.8]:
                    patch = mpatches.FancyArrow(0.8*height+0.5, 0.5 * height, w, 0,
                                                length_includes_head=True,
                                                head_width=height if w<width*0.8 else 0,
                                                head_length=0,
                                                overhang=0, lw=1.5,
                                                fc=color_, ec=color_)
                    handlebox.add_artist(patch)
                return patch


        legend_fns1 = [ArrivalTime(), RayPoint(), RayArrow(), NormalStick()]
        legend_labels1 = [ r'$T(\mathbf{r})$', r'$\mathbf{r}$', r'$\mathbf{v}$', r'$\mathbf{\widetilde{p}}$']
        legend_handlers1 = { ArrivalTime: HandlerArrivalTime(),
                             RayPoint:HandlerRayPoint(),
                             RayArrow: HandlerRayArrow(),
                             NormalStick: HandlerNormalStick() }
        legend1 = plt.legend( legend_fns1, legend_labels1, handler_map=legend_handlers1, loc='upper left' )
        axes.add_artist(legend1)

        divider = make_axes_locatable(axes)
        colorbar_axes = divider.append_axes('right', size="5%", pad=0.2)
        colorbar_axes.set_aspect(5)
        colorbar = plt.colorbar(colorbar_im, cax=colorbar_axes)
        colorbar.set_label(r'Anisotropy  $\psi = \alpha-\beta+90$  [${\degree}$]', rotation=270, labelpad=20)

        if do_pub_label:
            plt.text(0.93,0.15, pub_label,
                     transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')

    def profile_beta( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=26, xf_stop=1,
                      legend_loc='upper left', do_etaxi_label=True, eta_label_xy=None,
                      do_pub_label=False, pub_label='', pub_label_xy=None ) -> None:
        r"""
        For a time-invariant (steady-state) topographic profile,
        plot the surface-normal covector angle :math:`\beta` from vertical,
        aka the surface tilt angle from horizontal, as a function of
        dimensionless horizontal distance :math:`x/L_{\mathrm{c}}`.

        This angle is named and calculated in three ways:
        (1) :math:`\beta_p`: directly from the series of :math:`\mathbf{\widetilde{p}}` values
        generated by ray ODE integration, since :math:`\tan(\beta_p)=-p_x/p_z`;
        (2) :math:`\beta_{ts}`: by differentiating the computed time-invariant
        topographic profile (itself constructed from the terminations of the ensemble of
        progressively truncated, identical rays);
        (3) :math:`\beta_{vt}`: from the velocity triangle
        :math:`\tan(\beta_{vt}) = \dfrac{v^z+\xi^{\perp}}{v^x}`: generated by the geometric
        constraint of balancing the ray velocities with surface-normal velocities.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
                    instance of time invariant solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): sample rate along each curve
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        eta_label_xy = [0.6,0.8] if eta_label_xy is None else eta_label_xy
        pub_label_xy = [0.88,0.7] if pub_label_xy is None else pub_label_xy

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min,x_max*xf_stop,n_points)
        x_dbl_array = np.linspace(x_min,x_max*xf_stop,n_points*2-1)
        plt.plot(x_dbl_array, np.rad2deg(gmes.beta_vt_interp(x_dbl_array)),
                 'bs', ls='-', ms=3, label=r'$\beta_{vt}$ from $(v^z+\xi^{\!\downarrow\!})/v^x$')
        plt.plot(x_array, np.rad2deg(gmes.beta_ts_interp(x_array)),
                 'go', ls='-', ms=4, label=r'$\beta_{ts}$ from topo gradient')
        plt.plot(x_dbl_array, np.rad2deg(gmes.beta_p_interp(x_dbl_array)),
                 'r', ls='-',  ms=3, label=r'$\beta_p$ from $p_x/p_z$')
        #
        # plt.plot(gmes.rx_array,np.rad2deg(gmes.beta_array),
        #          'ks', ls='-', ms=3, label=r'$\beta_p$ from $p_x/p_z$')
        # plt.plot(gmes.rx_array,np.rad2deg(gmes.beta_ts_array),
        #          'bo', ls='-', ms=4, label=r'$\beta_{ts}$ from topo gradient')
        # plt.plot(gmes.rx_array,np.rad2deg(gmes.beta_vt_array),
        #          'r', ls='-', label=r'$\beta_{vt}$ from $(v^z+\xi^{\!\downarrow\!})/v^x$')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Angle $\beta$  [$\degree$]', fontsize=13)
        plt.grid(True, ls=':')
        plt.ylim(1e-9,)

        axes = plt.gca()
        plt.legend(loc=legend_loc, fontsize=11, framealpha=0.95)
        if do_etaxi_label:
            plt.text(*eta_label_xy,
                    rf'$\eta={gmeq.eta}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                    transform=axes.transAxes,
                    horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
            transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')

    def profile_beta_error(self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=101,
                           pub_label_xy=None, eta_label_xy=None, xf_stop=0.995) -> None:
        r"""
        For a time-invariant (steady-state) topographic profile,
        plot the error in the estimated surface-normal covector angle :math:`\beta`
        as a function of dimensionless horizontal distance :math:`x/L_{\mathrm{c}}`.

        This error, expressed as a percentage, is defined as one of the following normalized differences:
        (1) :math:`100(\beta_{ts}-\beta_{p})/\beta_{p}`, or
        (2) :math:`100(\beta_{vt}-\beta_{p})/\beta_{p}`.
        The error in :math:`\beta_{vt}` can be non-trivial for $x/L_{\mathrm{c}} \rightarrow 0$ and :math:`\eta < 1`
        if the number of rays used to construct the topographic profile is insufficient.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
                    instance of time invariant solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): sample rate along each curve
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        eta_label_xy = [0.5,0.8] if eta_label_xy is None else eta_label_xy
        pub_label_xy = [0.5,0.2] if pub_label_xy is None else pub_label_xy

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min,x_max*xf_stop,n_points)
        plt.plot(x_array, gmes.beta_vt_error_interp(x_array),
                 'b', ls='-', label=r'$\dfrac{\beta_{vt}-\beta_{p}}{\beta_{p}}$')
        plt.plot(x_array, gmes.beta_ts_error_interp(x_array),
                 'g', ls='-', label=r'$\dfrac{\beta_{ts}-\beta_{p}}{\beta_{p}}$')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Error  [%]', fontsize=13)
        plt.grid(True, ls=':')

        axes = plt.gca()
        ylim = axes.get_ylim()
        plt.ylim(ylim[0]*1.0,ylim[1]*1.3)
        plt.legend(loc='upper left', fontsize=9, framealpha=0.95)
        plt.text(*eta_label_xy,
                 rf'$\eta={gmeq.eta}'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                 transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

    def profile_xi( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, xf_stop=1, n_points=201,
                    pub_label_xy=None, eta_label_xy=None, var_label_xy=None,
                    do_etaxi_label=True, do_pub_label=False, pub_label='(a)', xi_norm=None ) -> None:
        r"""
        Plot surface-normal erosion speed :math:`\xi^{\perp}`  along a time-invariant profile.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
                    instance of time invariant solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): optional sample rate along each curve
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        eta_label_xy = [0.5,0.5] if eta_label_xy is None else eta_label_xy
        pub_label_xy = [0.5,0.2] if pub_label_xy is None else pub_label_xy
        var_label_xy = [0.8,0.5] if var_label_xy is None else var_label_xy

        if xi_norm is None:
            xi_norm = 1
            rate_label = r'$\xi^{\!\perp\!}$'
        else:
            xi_norm = float(N(xi_norm))
            rate_label = r'$\xi^{\!\perp\!}/\xi^{\!\rightarrow_{\!\!0}}$  [-]'
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min,x_max*xf_stop,n_points)
        u_array = gmes.u_interp(x_array)
        u_from_rdot_array = gmes.u_from_rdot_interp(x_array)
        dashes = [1, 2.]
        if not do_pub_label:
            plt.plot(x_array,u_from_rdot_array/xi_norm, 'g', dashes=dashes, lw=3, label=r'$\xi^{\!\perp\!}(x)$ from $v$')
            plt.plot(x_array,u_array/xi_norm, 'b', ls='-', lw=1.5, label=r'$\xi^{\!\perp\!}(x)$ from $1/p$')
        else:
            plt.plot(x_array,u_from_rdot_array/xi_norm, 'g', dashes=dashes, lw=3, label=r'from $v$')
            plt.plot(x_array,u_array/xi_norm, 'b', ls='-', lw=1.5, label=r'from $1/p$')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Normal erosion rate  '+rate_label, fontsize=12)
        plt.grid(True, ls=':')

        axes = plt.gca()
        ylim = plt.ylim()
        axes.set_ylim( -(ylim[1]-ylim[0])/20,ylim[1]*1.05 )
        plt.legend(loc='lower left', fontsize=11, framealpha=0.95)
        plt.text(*eta_label_xy,
                 rf'$\eta={gmeq.eta}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$'
                    if do_etaxi_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')
        plt.text(*var_label_xy, r'$\xi^{\perp}(x)$',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=18, color='k')
        plt.text(*pub_label_xy, pub_label if do_pub_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')

    def profile_xihorizontal( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, xf_stop=1, n_points=201,
                              pub_label_xy=None, eta_label_xy=None, var_label_xy=None,
                              do_etaxi_label=True, do_pub_label=False, pub_label='(d)', xi_norm=None ) -> None:
        r"""
        Plot horizontal erosion speed :math:`\xi^{\rightarrow}` along a time-invariant profile.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
                    instance of time invariant solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): sample rate along each curve
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        eta_label_xy = [0.5,0.2] if eta_label_xy is None else eta_label_xy
        pub_label_xy = [0.55,0.81] if pub_label_xy is None else pub_label_xy
        var_label_xy = [0.85,0.81] if var_label_xy is None else var_label_xy

        if xi_norm is None:
            xi_norm = 1
            rate_label = r'$\xi^{\!\rightarrow}$'
        else:
            xi_norm = float(N(xi_norm))
            rate_label = r'$\xi^{\!\rightarrow\!\!}/\xi^{\!\rightarrow_{\!\!0}}$  [-]'
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min,x_max*xf_stop,n_points)
        uhorizontal_p_array = gmes.uhorizontal_p_interp(x_array)/xi_norm
        uhorizontal_v_array = gmes.uhorizontal_v_interp(x_array)/xi_norm
        dashes = [1, 2.]
        if not do_pub_label:
            plt.plot(x_array,uhorizontal_v_array, 'g', dashes=dashes, lw=3, label=r'$\xi^{\!\rightarrow\!}(x)$ from $v$')
            plt.plot(x_array,uhorizontal_p_array, 'b', ls='-', lw=1.5, label=r'$\xi^{\!\rightarrow\!}(x)$ from $1/p$')
        else:
            plt.plot(x_array,uhorizontal_p_array, 'g', dashes=dashes, lw=3, label=r'from $v$')
            plt.plot(x_array,uhorizontal_v_array, 'b', ls='-', lw=1.5, label=r'from $1/p$')
        axes = plt.gca()
        ylim = plt.ylim()
        axes.set_ylim( -(ylim[1]-ylim[0])/20,ylim[1] )
        plt.grid(True, ls=':')

        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Horiz erosion rate  '+rate_label, fontsize=12)
        # axes.set_ylim(ylim[0]*1.1,-0)
        plt.legend(loc='lower left', fontsize=11, framealpha=0.95)
        plt.text(*eta_label_xy,
                 rf'$\eta={gmeq.eta}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$'
                    if do_etaxi_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')
        plt.text(*var_label_xy, r'$\xi^{\rightarrow}(x)$',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=18, color='k')
        plt.text(*pub_label_xy, pub_label if do_pub_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')

    def profile_xivertical( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, xf_stop=1,
                            n_points=201, y_limits=None,
                            pub_label_xy=None, eta_label_xy=None, var_label_xy=None,
                            do_etaxi_label=True, do_pub_label=False, pub_label='(e)', xi_norm=None ) -> None:
        r"""
        Plot vertical erosion speed :math:`\xi^{\downarrow}` along a time-invariant profile.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
                    instance of time invariant solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): sample rate along each curve
            y_limits (list of float):
                optional [z_min, z_max] vertical plot range
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        eta_label_xy = [0.5,0.2] if eta_label_xy is None else eta_label_xy
        pub_label_xy = [0.5,0.81] if pub_label_xy is None else pub_label_xy
        var_label_xy = [0.85,0.81] if var_label_xy is None else var_label_xy

        if xi_norm is None:
            xi_norm = 1
            rate_label = r'$\xi^{\!\downarrow}$'
        else:
            xi_norm = float(N(xi_norm))
            rate_label = r'$\xi^{\!\downarrow}\!\!/\xi^{\!\rightarrow_{\!\!0}}$  [-]'
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min,x_max*xf_stop,n_points)
        xiv_p_array = gmes.xiv_p_interp(x_array)/xi_norm
        xiv_v_array = gmes.xiv_v_interp(x_array)/xi_norm
        if not do_pub_label:
            plt.plot(x_array,xiv_v_array, 'g', ls='-', label=r'$\xi^{\!\downarrow\!}(x)$ from $v$')
            plt.plot(x_array,xiv_p_array, 'b', ls='-', label=r'$\xi^{\!\downarrow\!}(x)$ from $1/p$')
        else:
            plt.plot(x_array,xiv_v_array, 'g', ls=':', lw=3, label=r'from $v$')
            plt.plot(x_array,xiv_p_array, 'b', ls='-', label=r'from $1/p$')
        axes = plt.gca()
        # ylim = plt.ylim()
        plt.grid(True, ls=':')

        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Vertical erosion rate  '+rate_label, fontsize=12)
        if y_limits is not None:
            axes.set_ylim(*y_limits)
        else:
            xiv_mean = np.mean(xiv_p_array)
            xiv_deviation = np.max([xiv_mean-np.min(xiv_p_array),
                                    np.max(xiv_p_array)-xiv_mean]) * 1.1
            axes.set_ylim([xiv_mean-xiv_deviation, xiv_mean+xiv_deviation])
        plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
        plt.text(*eta_label_xy,
                 rf'$\eta={gmeq.eta}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$'
                    if do_etaxi_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')
        plt.text(*var_label_xy, r'$\xi^{\downarrow}(x)$',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=18, color='k')
        plt.text(*pub_label_xy, pub_label if do_pub_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')



    # def profile_slope_area( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
    #                         n_points=26, subtitle='', x_min=0.01,
    #                         do_subtitling=False, do_extra_annotations=False, do_simple=False ):
    #     r"""
    #     Generate a log-log slope-area plot for a time-invariant topographic profile solution
    #     of Hamilton's equations.
    #
    #     Args:
    #         fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
    #             reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
    #         gmes (:class:`~.ode_raytracing.TimeInvariantSolution`):
    #                 instance of time invariant solution class defined in :mod:`~.ode_raytracing`
    #         gmeq (:class:`~.equations.Equations`):
    #                 GME model equations class instance defined in :mod:`~.equations`
    #         sub (dict): dictionary of model parameter values to be used for equation substitutions
    #         n_points (int): optional number of points to plot along curve
    #         subtitle (str): optional sub-title (likely 'ramp' or 'ramp-flat' or similar)
    #         x_min (float): optional x offset
    #         do_subtitling (bool): optionally annotate with subtitle and eta value?
    #         do_extra_annotations (bool): optionally annotate with hillslope, channel labels?
    #         do_simple (bool): optionally avoid hillslope-channel extras?
    #     """
    #     _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
    #     axes = plt.gca()
    #
    #     eta_ = float(gmeq.eta)
    #     x_array = np.linspace(0,float(Lc.subs(sub)-x_min),n_points)
    #     x_dbl_array = np.linspace(0,float(Lc.subs(sub)-x_min),n_points*2-1)
    #     area_array = ((Lc.subs(sub)-x_dbl_array)/Lc.subs(sub))**2
    #     slope_array = gmes.beta_vt_interp(x_dbl_array)
    #     if eta_!=1.0 or do_simple:
    #         plt.loglog( area_array, slope_array, '-' )
    #     # plt.loglog( area_array, gmes.beta_vt_interp(x_dbl_array), '-' )
    #     from matplotlib.ticker import ScalarFormatter
    #     for axis in [axes.yaxis]:
    #         axis.set_major_formatter(ScalarFormatter())
    #
    #     eta0p5_slope_array = [\
    #     0.0094, 0.0096, 0.0098, 0.0100, 0.0102, 0.0104, 0.0107, 0.0109, 0.0112, 0.0115, 0.0117, 0.0120, 0.0123, 0.0126, 0.0129, 0.0133, 0.0136, 0.0140, 0.0143, 0.0147,
    #     0.0151, 0.0155, 0.0160, 0.0164, 0.0169, 0.0174, 0.0179, 0.0184, 0.0190, 0.0196, 0.0202, 0.0209, 0.0215, 0.0222, 0.0230, 0.0238, 0.0246, 0.0255, 0.0264, 0.0274,
    #     0.0284, 0.0295, 0.0306, 0.0318, 0.0331, 0.0345, 0.0359, 0.0374, 0.0391, 0.0408, 0.0427, 0.0447, 0.0469, 0.0492, 0.0516, 0.0543, 0.0572, 0.0603, 0.0636, 0.0673,
    #     0.0713, 0.0756, 0.0803, 0.0855, 0.0912, 0.0974, 0.1043, 0.1120, 0.1205, 0.1299, 0.1405, 0.1523, 0.1655, 0.1804, 0.1973, 0.2163, 0.2379, 0.2625, 0.2902, 0.3217,
    #     0.3573, 0.3974, 0.4423, 0.4923, 0.5475, 0.6081, 0.6740, 0.7450, 0.8196, 0.8959, 0.9673, 1.0240, 1.0601, 1.0780, 1.0858, 1.0887, 1.0899, 1.0903, 1.0905, 1.0905,
    #     1.0905]
    #
    #     eta0p5_area_array = [1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083,
    #     0.6917, 0.6754, 0.6592, 0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142,
    #     0.4016, 0.3891, 0.3769, 0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986,
    #     0.1898, 0.1813, 0.1730, 0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613,
    #     0.0565, 0.0519, 0.0475, 0.0433, 0.0392, 0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025,
    #     0.0016, 0.0009, 0.0004, 0.0001]
    #
    #     eta1p0_slope_array = [\
    #     0.0960, 0.0970, 0.0981, 0.0991, 0.1002, 0.1013, 0.1025, 0.1036, 0.1048, 0.1060, 0.1073, 0.1085, 0.1098, 0.1112, 0.1125, 0.1139, 0.1153, 0.1168, 0.1183, 0.1199,
    #     0.1214, 0.1231, 0.1247, 0.1264, 0.1282, 0.1300, 0.1319, 0.1338, 0.1358, 0.1378, 0.1399, 0.1420, 0.1442, 0.1465, 0.1489, 0.1513, 0.1538, 0.1564, 0.1591, 0.1619,
    #     0.1647, 0.1677, 0.1708, 0.1740, 0.1773, 0.1807, 0.1843, 0.1880, 0.1919, 0.1959, 0.2001, 0.2044, 0.2090, 0.2137, 0.2187, 0.2238, 0.2292, 0.2349, 0.2409, 0.2471,
    #     0.2537, 0.2606, 0.2679, 0.2756, 0.2837, 0.2922, 0.3013, 0.3109, 0.3211, 0.3319, 0.3434, 0.3557, 0.3688, 0.3828, 0.3979, 0.4141, 0.4314, 0.4501, 0.4703, 0.4922,
    #     0.5159, 0.5419, 0.5701, 0.6009, 0.6347, 0.6720, 0.7132, 0.7588, 0.8084, 0.8615, 0.9137, 0.9573, 0.9857, 1.0003, 1.0066, 1.0091, 1.0100, 1.0104, 1.0105, 1.0105,
    #     1.0106]
    #
    #     eta1p0_area_array = [1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083, 0.6917,
    #     0.6754, 0.6592, 0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142, 0.4016,
    #     0.3891, 0.3769, 0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986, 0.1898,
    #     0.1813, 0.1730, 0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613, 0.0565,
    #     0.0519, 0.0475, 0.0433, 0.0392,  0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025, 0.0016,
    #     0.0009, 0.0004, 0.0001]
    #
    #     eta1p0_xh1p0_slope_array = [\
    #     0.0960, 0.0969, 0.0979, 0.0988, 0.0998, 0.1008, 0.1018, 0.1028, 0.1039, 0.1049, 0.1060, 0.1071, 0.1083, 0.1094, 0.1106, 0.1118, 0.1131, 0.1143, 0.1156, 0.1170,
    #     0.1183, 0.1197, 0.1211, 0.1226, 0.1241, 0.1256, 0.1271, 0.1287, 0.1304, 0.1321, 0.1338, 0.1356, 0.1374, 0.1392, 0.1412, 0.1431, 0.1451, 0.1472, 0.1494, 0.1516,
    #     0.1538, 0.1562, 0.1586, 0.1610, 0.1636, 0.1662, 0.1689, 0.1717, 0.1746, 0.1776, 0.1807, 0.1839, 0.1873, 0.1907, 0.1943, 0.1980, 0.2018, 0.2058, 0.2099, 0.2142,
    #     0.2187, 0.2233, 0.2281, 0.2332, 0.2385, 0.2439, 0.2497, 0.2557, 0.2620, 0.2686, 0.2756, 0.2828, 0.2905, 0.2985, 0.3070, 0.3159, 0.3253, 0.3352, 0.3458, 0.3569,
    #     0.3688, 0.3814, 0.3948, 0.4091, 0.4243, 0.4406, 0.4580, 0.4767, 0.4968, 0.5185, 0.5419, 0.5672, 0.5945, 0.6243, 0.6568, 0.6922, 0.7310, 0.7736, 0.8202, 0.8694,
    #     0.9179]
    #
    #     eta1p0_xh1p0_area_array = [\
    #     1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083, 0.6917, 0.6754, 0.6592,
    #     0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142, 0.4016, 0.3891, 0.3769,
    #     0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986, 0.1898, 0.1813, 0.1730,
    #     0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613, 0.0565, 0.0519, 0.0475,
    #     0.0433, 0.0392, 0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025, 0.0016, 0.0009, 0.0004,
    #     0.0001]
    #
    #     eta1p5_slope_array = [\
    #     0.2049, 0.2063, 0.2077, 0.2092, 0.2106, 0.2121, 0.2136, 0.2151, 0.2167, 0.2183, 0.2199, 0.2216, 0.2233, 0.2250, 0.2267, 0.2285, 0.2303, 0.2322, 0.2340, 0.2360,
    #     0.2379, 0.2400, 0.2420, 0.2441, 0.2463, 0.2485, 0.2507, 0.2530, 0.2554, 0.2578, 0.2602, 0.2628, 0.2653, 0.2680, 0.2707, 0.2735, 0.2763, 0.2792, 0.2822, 0.2853,
    #     0.2884, 0.2917, 0.2950, 0.2984, 0.3019, 0.3056, 0.3093, 0.3131, 0.3171, 0.3212, 0.3254, 0.3298, 0.3343, 0.3390, 0.3439, 0.3489, 0.3541, 0.3595, 0.3651, 0.3709,
    #     0.3769, 0.3832, 0.3898, 0.3966, 0.4038, 0.4112, 0.4190, 0.4272, 0.4358, 0.4448, 0.4543, 0.4643, 0.4748, 0.4860, 0.4977, 0.5102, 0.5236, 0.5380, 0.5533, 0.5697,
    #     0.5873, 0.6064, 0.6270, 0.6495, 0.6742, 0.7015, 0.7318, 0.7656, 0.8027, 0.8432, 0.8837, 0.9186, 0.9414, 0.9534, 0.9584, 0.9605, 0.9613, 0.9615, 0.9616, 0.9617,
    #     0.9617]
    #
    #     eta1p5_area_array = [1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083,
    #     0.6917, 0.6754, 0.6592, 0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142,
    #     0.4016, 0.3891, 0.3769, 0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986,
    #     0.1898, 0.1813, 0.1730, 0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613,
    #     0.0565, 0.0519, 0.0475, 0.0433, 0.0392, 0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025,
    #     0.0016, 0.0009, 0.0004, 0.0001]
    #
    #     if eta_==1.0 and not do_simple:
    #         plt.loglog(eta1p5_area_array,eta1p5_slope_array,'-', label=r'$\eta=3/2\,,\,\,x_h/L_{\mathrm{c}}=0.9$', color=self.colors[1])
    #         plt.loglog(eta0p5_area_array,eta0p5_slope_array,'-', label=r'$\eta=1/2\,,\,\,x_h/L_{\mathrm{c}}=0.9$', color=self.colors[2])
    #         plt.loglog(eta1p0_area_array,eta1p0_slope_array,'-', label=r'$\eta=1\,,\,\,x_h/L_{\mathrm{c}}=0.9$', color=self.colors[0])
    #         plt.loglog(eta1p0_xh1p0_area_array,eta1p0_xh1p0_slope_array,'--', label=r'channel-only model', color=self.colors[0])
    #
    #     # axes.set_aspect(1)
    #     plt.grid(True, ls=':')
    #     plt.xlabel(r'Dimensionless area, $(L_{\mathrm{c}}-x)^2/L_{\mathrm{c}}^2$  [-]')
    #     plt.ylabel(r'Slope, $|\tan\beta|$  [-]')
    #     # plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
    #
    #     if do_subtitling:
    #         plt.text(0.1,0.15, r'$\eta={}$'.format(gmeq.eta), transform=axes.transAxes,
    #                  horizontalalignment='left', verticalalignment='center', fontsize=12, color='k')
    #         plt.text(0.05,0.25, subtitle, transform=axes.transAxes,
    #                  horizontalalignment='left', verticalalignment='center', fontsize=12, color='k')
    #     if do_extra_annotations:
    #         plt.text(0.29,0.83, 'hillslope', transform=axes.transAxes,
    #                  horizontalalignment='center', verticalalignment='center', fontsize=12, color='0.2')
    #         plt.text(0.735,0.49, 'channel', transform=axes.transAxes,
    #                  rotation=-58.5, horizontalalignment='center', verticalalignment='center', fontsize=12, color='0.2')
    #
    #     y_limits = axes.get_ylim()
    #     x_h_ = x_h.subs(sub)
    #     area_h_ = float(( ((Lc-x_h_)/Lc)**2 ).subs(sub))
    #     slope_h_ = gmes.beta_vt_interp(float(x_h_))
    #     plt.plot([area_h_,area_h_],[slope_h_*0.45,slope_h_*1.5],'b:')
    #     if area_h_>0.0:
    #         plt.text(area_h_,slope_h_*0.4, r'$\dfrac{(L_{\mathrm{c}}-x_h)^2}{L_{\mathrm{c}}^2}$', #transform=axes.transAxes,
    #                      horizontalalignment='center', verticalalignment='top', fontsize=12, color='b')
    #
    #     if not do_simple:
    #         plt.legend(loc='lower left', fontsize=11, framealpha=0.95)
    #     plt.xlim(None,1.15)
    #     plt.ylim(*y_limits)
    #
    #     return area_array, slope_array

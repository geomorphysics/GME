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
  -  :mod:`numpy`, :mod:`sympy`
  -  :mod:`matplotlib.pyplot`
  -  :mod:`gme.core.symbols`, :mod:`gme.core.equations`, :mod:`gme.plot.base`

---------------------------------------------------------------------

"""
import warnings

# Numpy
import numpy as np

# SymPy
from sympy import deg, atan

# MatPlotLib
import matplotlib.pyplot as plt

# GME
from gme.core.symbols import px, pz, Ci
from gme.core.equations import px_value
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['RayAngles']


class RayAngles(Graphing):
    """
    Subclasses :class:`gme.plot.Graphing <plot.Graphing>`.
    """

    def alpha_beta( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, aspect=1,
                    n_points=201, x_limits=None, y_limits=None, do_legend=True,
                    do_etaxi_label=True, eta_label_xy=(0.5,0.85),
                    do_pub_label=False, pub_label='', pub_label_xy=(0.88,0.7) ) -> None:
        r"""
        Plot ray vector angle :math:`\alpha` versus normal-slowness covector angle :math:`\beta`
        generated by a time-invariant solution.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): optional sample rate along each curve
            x_limits (list of float):
                optional [x_min, x_max] horizontal plot range
            y_limits (list of float):
                optional [z_min, z_max] vertical plot range
            do_legend (bool): optional plot legend?
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # eta_label_xy = [0.5,0.85] if eta_label_xy is None else eta_label_xy
        # pub_label_xy = [0.88,0.7] if pub_label_xy is None else pub_label_xy

        x_array = np.linspace(0,1,n_points)
        alpha_array = np.rad2deg(gmes.alpha_interp(x_array))
        beta_p_array = np.rad2deg(gmes.beta_p_interp(x_array))
        plt.plot( beta_p_array, alpha_array-90, 'b', ls='-', label=r'$\alpha(\beta)-90$')
        plt.xlabel(r'Surface normal angle  $\beta$  [${\degree}$ from vertical]')
        plt.ylabel(r'Ray angle  $\alpha\,$  [${\degree}$ from horiz]')
        plt.grid(True, ls=':')

        axes = plt.gca()
        axes.set_aspect(aspect)
        xlim = axes.get_xlim()
        # ylim = axes.get_ylim()
        axes.set_yticks([-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90])
        if x_limits is None:
            axes.set_xlim( -(xlim[1]-xlim[0])/30,xlim[1] )
        else:
            axes.set_xlim(*x_limits)
        if y_limits is not None:
            axes.set_ylim(*y_limits)
        if do_legend:
            plt.legend()
        if do_etaxi_label:
            # plt.text(0.5,0.85, r'$\eta={}$'.format(gmeq.eta_),
            plt.text(*eta_label_xy,
                     rf'$\eta={gmeq.eta_}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
            transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')

    def angular_disparity( self, gmes, gmeq, name, fig_size=None, dpi=None,
                           n_points=201, x_limits=None, y_limits=None, do_legend=True,
                           aspect=0.75,
                           pub_label_xy=(0.5,0.2), eta_label_xy=(0.5,0.81), #var_label_xy=(0.85,0.81),
                           do_pub_label=False, pub_label='' ) -> None:
        r"""
        Plot ray vector angular disparity :math:`\alpha-\beta`
        generated by a time-invariant solution.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): optional sample rate along each curve
            x_limits (list of float):
                optional [x_min, x_max] horizontal plot range
            y_limits (list of float):
                optional [z_min, z_max] vertical plot range
            do_legend (bool): optional plot legend?
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # pub_label_xy = [0.5,0.2] if pub_label_xy is None else pub_label_xy
        # eta_label_xy = [0.5,0.81] if eta_label_xy is None else eta_label_xy
        # var_label_xy = [0.85,0.81] if var_label_xy is None else var_label_xy

        x_array = np.linspace(0,1,n_points)
        alpha_array = np.rad2deg(gmes.alpha_interp(x_array))
        beta_p_array = np.rad2deg(gmes.beta_p_interp(x_array))
        plt.plot( beta_p_array, alpha_array-beta_p_array, 'DarkBlue', ls='-', label=r'$\alpha(\beta)-90$')
        plt.xlabel(r'Surface normal angle  $\beta$  [${\degree}$ from vertical]')
        plt.ylabel(r'Angular disparity  $(\alpha-\beta\!)+90\,$  [${\degree}$]')
        plt.grid(True, ls=':')

        axes = plt.gca()
        axes.set_aspect(aspect)
        xlim = axes.get_xlim()
        # ylim = axes.get_ylim()
        axes.set_yticks([-20,-10,0,10,20,30,40,50,60,70,80,90])
        if x_limits is None:
            axes.set_xlim( -(xlim[1]-xlim[0])/30,xlim[1] )
        else:
            axes.set_xlim(*x_limits)
        if y_limits is not None:
            axes.set_ylim(*y_limits)
        if do_legend:
            plt.legend()
        plt.text(*eta_label_xy, rf'$\eta={gmeq.eta_}$', transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
            transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')

    def profile_angular_disparity( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=201,
                                   pub_label_xy=(0.5,0.2), eta_label_xy=(0.25,0.5), var_label_xy=(0.8,0.35),
                                   do_pub_label=False, pub_label='(a)' ) -> None:
        r"""
        Plot horizontal erosion speed :math:`\xi^{\rightarrow}` along a time-invariant profile.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): sample rate along each curve
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # pub_label_xy = [0.5,0.2] if pub_label_xy is None else pub_label_xy
        # eta_label_xy = [0.25,0.5] if eta_label_xy is None else eta_label_xy
        # var_label_xy = [0.8,0.35] if var_label_xy is None else var_label_xy

        x_array = np.linspace(0,1,n_points)
        # x_dbl_array = np.linspace(0,1,n_points*2-1)
        angular_diff_array = np.rad2deg(gmes.alpha_interp(x_array)-gmes.beta_p_interp(x_array))
        plt.plot(x_array,angular_diff_array, 'DarkBlue', ls='-', lw=1.5, label=r'$\alpha(x)-\beta(x)$')
        axes = plt.gca()
        # ylim = plt.ylim()
        axes.set_yticks([-30,0,30,60,90])
        axes.set_ylim( -5,95 )
        plt.grid(True, ls=':')

        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Anisotropy,  $\psi = \alpha-\beta+90$  [${\degree}$]', fontsize=12)
        if not do_pub_label:
            plt.legend(loc='lower left', fontsize=11, framealpha=0.95)
        plt.text(*pub_label_xy, pub_label if do_pub_label else rf'$\eta={gmeq.eta_}$',
                 transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')
        plt.text(*var_label_xy, r'$\psi(x)$' if do_pub_label else '',
                 transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=18, color='k')
        plt.text(*eta_label_xy,
                 rf'$\eta={gmeq.eta_}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                 transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

    def profile_alpha( self, gmes, gmeq, name, fig_size=None, dpi=None, n_points=201, do_legend=True) -> None:
        r"""
        Plot ray vector angle :math:`\alpha` along a time-invariant profile.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            n_points (int): optional sample rate along each curve
            do_legend (bool): optional plot legend?
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        x_array = np.linspace(0,1,n_points)
        alpha_array = np.rad2deg(gmes.alpha_interp(x_array))
        plt.plot(x_array,alpha_array-90, 'DarkBlue', ls='-', label=r'$\alpha(x)$')
        x_array = np.linspace(0,1,11)
        pz0_ = gmes.pz0
        # TBD
        alpha_array = [(np.mod(180+np.rad2deg(float(
            atan(gmeq.tanalpha_pxpz_eqn.rhs.subs({px:px_value(x_,pz0_),pz:pz0_})))),180))
            for x_ in x_array]
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]')
        plt.ylabel(r'Ray dip  $\alpha\!\,$  [${\degree}$ from horiz]')
        plt.grid(True, ls=':')

        if do_legend:
            plt.legend()
        axes = plt.gca()
        plt.text(0.5,0.7, rf'$\eta={gmeq.eta_}$', transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
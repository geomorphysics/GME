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

# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name
import warnings

# Numpy
import numpy as np

# SymPy
from sympy import N, deg

# GME
from gme.core.symbols import Ci, varepsilon
from gme.plot.base import Graphing

# MatPlotLib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

__all__ = ['RayVelocities']


class RayVelocities(Graphing):
    """
    Subclasses :class:`gme.plot.Graphing <plot.Graphing>`.
    """

    def profile_v( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=201,
                   pub_label_xy=(0.5,0.5), eta_label_xy=(0.5,0.81), var_label_xy=(0.8,0.5),
                   do_pub_label=False, pub_label='', do_etaxi_label=True,
                   xi_norm=None, legend_loc='lower right', do_mod_v=False ) -> None:
        r"""
        Plot velocity :math:`\dot{r}` along a ray.

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
        # pub_label_xy = [0.5,0.5] if pub_label_xy is None else pub_label_xy
        # eta_label_xy = [0.5,0.81] if eta_label_xy is None else eta_label_xy
        # var_label_xy = [0.8,0.5] if var_label_xy is None else var_label_xy

        if xi_norm is None:
            xi_norm = 1
            rate_label = '${v}$'
        else:
            xi_norm = float(N(xi_norm))
            rate_label = r'${v}/\xi^{\!\rightarrow_{\!\!0}}$  [-]'
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max,n_points)
        # t_array  = gmes.t_interp_x(x_array)
        vx_array = gmes.rdotx_interp(x_array)/xi_norm
        vz_array = gmes.rdotz_interp(x_array)/xi_norm
        v_array = np.sqrt(vx_array**2+vz_array**2)
        # vx_max = np.max(np.abs(vx_array))
        vz_max = np.max(np.abs(vz_array))
        v_max = np.max((v_array))

        if do_mod_v:
            plt.plot( x_array, v_array, 'DarkBlue', ls='-', lw=1.5, label=r'${v}(x)$')
            plt.ylabel(r'Ray speed  '+rate_label, fontsize=13)
            legend_loc = 'lower left'
        else:
            sfx = np.power(10,np.round(np.log10(vz_max/v_max),0))
            label_suffix = '' if sfx==1 else r'$\,\times\,$'+f'{sfx}'
            plt.plot( x_array, vx_array*sfx, 'r', ls='-', lw=1.5,
                                label=r'${v}^x(x)$'+label_suffix)
            plt.plot( x_array, vz_array, 'b', ls='-', lw=1.5,
                                label=r'${v}^z(x)$')
            plt.ylabel(r'Ray velocity  '+rate_label, fontsize=13)

        axes = plt.gca()
        ylim = plt.ylim()
        if ylim[1]<0: axes.set_ylim(ylim[0],0)
        if ylim[0]>0: axes.set_ylim(0,ylim[1])
        # axes.set_ylim( -(ylim[1]-ylim[0])/20,ylim[1] )
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.legend(loc=legend_loc, fontsize=14, framealpha=0.95)
        if do_etaxi_label:
            plt.text(*eta_label_xy,
                     rf'$\eta={gmeq.eta}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=16, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
                     transform=axes.transAxes, horizontalalignment='center', verticalalignment='center',
                     fontsize=16, color='k')
        plt.text(*var_label_xy, r'${v}(x)$' if do_mod_v else r'$\mathbf{v}(x)$',
                 transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=18, color='k')

    def profile_vdot( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                      n_points=201, do_pub_label=False, pub_label='', xi_norm=None, do_etaxi_label=True,
                      legend_loc='lower right', do_legend=True, do_mod_vdot=False, do_geodesic=False ) -> None:
        r"""
        Plot acceleration :math:`\ddot{r}` along a ray.

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

        # Use an erosion rate (likely vertical xi) to renormalize velocities and accelns (up to T)
        if xi_norm is None:
            xi_norm = 1
            rate_label = r'$\dot{v}$'
        else:
            xi_norm = float(N(xi_norm))
            rate_label = r'$\dot{v}/\xi^{\!\rightarrow_{\!\!0}}$  [T$^{-1}$]'

        # Specify sampling in x and t
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max,n_points)
        t_array  = gmes.t_interp_x(x_array)

        # Get ray velocities
        vdotx_array = gmes.rddotx_interp_t(t_array)/xi_norm
        vdotz_array = gmes.rddotz_interp_t(t_array)/xi_norm
        vdot_array = np.sqrt(vdotx_array**2+vdotz_array**2)

        # Prep to set vertical axis to span vdotz and thus scale vdotx to fit
        # vdotx_max = np.max(np.abs(vdotx_array))
        vdotz_max = np.max(np.abs(vdotz_array))
        vdot_max = np.max((vdot_array))

        # Start doing some plotting
        sfx = 1 if np.abs(vdotz_max)<1e-20 else np.power(10,np.round(np.log10(vdotz_max/vdot_max),0))
        label_suffix = '' if sfx==1 else rf'$\,\times\,${sfx}'
        vdotx_label = r'$\dot{v}^x_\mathrm{hmltn}(x)$'+label_suffix
        vdotz_label = r'$\dot{v}^z_\mathrm{hmltn}(x)$'
        if do_mod_vdot:
            plt.plot( x_array, vdot_array, 'DarkBlue', ls='-', lw=1.5,
                        label=r'$\dot{v}_\mathrm{hmltn}(x)$')
            plt.ylabel(r'Ray acceleration  '+rate_label, fontsize=13)
            legend_loc = 'lower left'
        else:
            plt.plot( x_array, vdotx_array*sfx, 'r', ls='-', lw=1.5, label=vdotx_label)
            plt.plot( x_array, vdotz_array, 'b', ls='-', lw=1.5, label=vdotz_label)
            plt.ylabel(r'Ray acceleration  '+rate_label, fontsize=14)

        ylim = plt.ylim()

        # Geodesic computation of acceln using Christoffel symbols
        if do_geodesic and hasattr(gmeq,'vdotx_lambdified') and hasattr(gmeq,'vdotz_lambdified')\
                        and gmeq.vdotx_lambdified is not None and gmeq.vdotz_lambdified is not None:
            vx_array = gmes.rdotx_interp(x_array)
            vz_array = gmes.rdotz_interp(x_array)
            vdotx_gdsc_array = np.array([gmeq.vdotx_lambdified(float(x), float(vx), float(vz), varepsilon.subs(sub))/xi_norm
                                        for x,vx,vz in zip(x_array, vx_array, vz_array)])
            vdotz_gdsc_array = np.array([gmeq.vdotz_lambdified(float(x), float(vx), float(vz), varepsilon.subs(sub))/xi_norm
                                        for x,vx,vz in zip(x_array, vx_array, vz_array)])
            vdot_gdsc_array = np.sqrt(vdotx_gdsc_array**2+vdotz_gdsc_array**2)
            vdotx_label = r'$\dot{v}^x_\mathrm{gdsc}(x)$'+label_suffix
            vdotz_label = r'$\dot{v}^z_\mathrm{gdsc}(x)$'
            if do_mod_vdot:
                plt.plot( x_array, vdot_gdsc_array, 'DarkBlue', ls=':', lw=3,
                            label=r'$\dot{v}_\mathrm{gdsc}(x)$')
                plt.ylabel(r'Ray acceleration  '+rate_label, fontsize=13)
                legend_loc = 'lower left'
            else:
                plt.plot( x_array, vdotx_gdsc_array*sfx, 'DarkRed', ls=':', lw=3, label=vdotx_label)
                plt.plot( x_array, vdotz_gdsc_array, 'DarkBlue', ls=':', lw=3, label=vdotz_label)

        # Misc pretty stuff
        axes = plt.gca()
        axes.set_ylim(*ylim)
        plt.grid(True, ls=':')

        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=14)
        if do_legend: plt.legend(loc=legend_loc, fontsize=13, framealpha=0.95)
        if do_etaxi_label:
            plt.text(0.6,0.8, pub_label if do_pub_label else rf'$\eta={gmeq.eta}$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

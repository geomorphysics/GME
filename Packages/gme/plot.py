"""
---------------------------------------------------------------------

Visualization.

Provides classes to generate a range of graphics for GME visualization.
A base class extends :class:`gmplib.plot_utils.GraphingBase <plot_utils.GraphingBase>` provided by :mod:`GMPLib`;
the other classes build on this.
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

from gmplib.plot_utils import GraphingBase

from sympy import Eq, factor, N, Abs, lambdify, Rational, Matrix, poly, \
                    simplify, diff, sign, sin, tan, deg, solve, sqrt, rad, numer, denom, im, re
from gme.symbols import *
from gme.equations import px_value
from gmplib.utils import e2d, omitdict, round as gmround, convert
import numpy as np

# Scipy utils
from scipy.linalg import eig, eigh, det, norm
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, FancyArrow, FancyArrowPatch, Arrow, Rectangle, Circle, RegularPolygon,\
                                ArrowStyle, ConnectionPatch, Arc
from matplotlib.spines import Spine
from matplotlib.legend_handler import HandlerPatch
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings
warnings.filterwarnings("ignore")

__all__ = ['Graphing', 'OneRayPlots', 'TimeInvariantPlots',
           'TimeDependentPlots', 'TheoryPlots', 'SlicingPlots', 'ManuscriptPlots']

class Graphing(GraphingBase):
    """
    Subclasses :class:`gmplib.plot_utils.GraphingBase <plot_utils.GraphingBase>`.
    """
    def __init__(self, dpi=100, font_size=11):
        """
        Constructor method.

        Args:
            dpi (int): resolution for rasterized images
            font_size (int): general font size
        """
        super().__init__(dpi, font_size)

    def mycolors(self, i, r, n, do_smooth=False, cmap_choice='brg'):
        if not do_smooth:
            return self.colors[(i//r)%self.n_colors]
        else:
            cmap = cm.get_cmap(cmap_choice)
            return cmap(i/(n-1))

    @staticmethod
    def gray_color(i_isochrone=0, n_isochrones=1):
        return '{}'.format((n_isochrones-1-i_isochrone)/(n_isochrones-1)*0.75)

    @staticmethod
    def correct_quadrant(angle):
        r"""
        If angle :math:`|\theta|\approx 0`, set :math:`\theta=0`;
        otherwise, if angle :math:`\theta<0`, map :math:`\theta \rightarrow \pi-\theta`.

        Args:
            angle (float): angle in radians

        Returns:
            Modified value of angle.
        """
        if abs(angle)<=1e-10:
            return 0
        elif angle>0:
            return angle
        else:
            return np.pi+angle

    def draw_rays_with_arrows_simple( self, axes, sub, t_array, rx_array, rz_array, v_array=None,
                                      n_t=None, n_rays=4,
                                      ls='-', sf=1, color=None, do_labels=True, do_one_ray=False ):
        """
        Plot ray and arrowheads along the ray to visualize the direction of motion.

        Args:
            axes (:class:`Matplotlib axes <matplotlib.axes.Axes>`): 'axes' instance for current figure
            sub (dict): dictionary of model parameter values to be used for equation substitutions
            t_array (numpy.ndarray): sample times along the ray
            rx_array (numpy.ndarray): x coordinates along the sampled ray
            rz_array (numpy.ndarray): z coordinates along the sampled ray
            ls (str): optional line style
        """
        i_max = len(t_array) if n_t is None else n_t
        i_step = i_max//n_rays
        i_off = i_step*(1+i_max//i_step)-i_max + 1
        my_arrow_style = mpatches.ArrowStyle.Fancy(head_length=.99*sf, head_width=.6*sf,
                                                   tail_width=0.01*sf)
        if v_array is not None:
            v_max, v_min = max(v_array), min(v_array)
            color_map = plt.get_cmap('plasma')
        t_ref = t_array[i_max-1]
        for i in range(i_max-1,0,-1):
            color = color if do_one_ray else self.colors[(i//i_step)%self.n_colors]
            if (i+i_off)//i_step==(i+i_off)/i_step:
                t_offset = 0 if do_one_ray else float(t_array[i]*(-(xiv_0/xih_0).subs(sub)))
                # print(t_ref, round(t_ref-t_array[i],1), t_offset)
                t_label = f'$\hat{t}_0={round(t_ref-t_array[i],1)}$'  #$t_0={}$'.format(i_max-i-1)
                plt.plot( rx_array[:i+1], rz_array[:i+1]-t_offset, ls,
                          label=t_label if do_labels else None,
                          color=color)
            # if (i+i_off)//i_step==(i+i_off)/i_step:
                for q in range(1,i-1,3):
                    if do_one_ray:
                        v_rel = ( (v_array[q]-v_min)/(v_max-v_min) )**0.5
                        rgba = color_map(v_rel*0.8)
                    else:
                        rgba = color
                    axes.annotate('', xy=((rx_array[q+1]),(rz_array[q+1]-t_offset)),
                                      xytext=(rx_array[q], rz_array[q]-t_offset),
                                  arrowprops=dict(arrowstyle=my_arrow_style, color=rgba) )
        if v_array is not None:
            color_map = plt.get_cmap('plasma')
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = plt.colorbar(sm, ticks=[], shrink=0.4, aspect=5, pad=0.03)
            for idx, tick_label in enumerate([r'$v_\mathrm{min}$',r'$v_\mathrm{max}$']):
                cbar.ax.text(0.45, idx*1.3-0.15, tick_label, ha='center', va='center')
            # cbar.ax.get_yaxis().labelpad = -100
            # cbar.ax.set_ylabel(r'ray speed  $v$', rotation=90)

    def arrow_annotate_ray_custom( self, rx_array, rz_array, axes, sub, i_ray, i_ray_step, n_rays, n_arrows,
                                   arrow_sf=0.7, arrow_offset=1, x_limits=None, y_limits=None,
                                   line_style='dashed', line_width=1, ray_label=None, do_smooth_colors=False ):
        """
        Add arrowheads to a ray trajectory to visualize the direction of motion.

        Args:
            rx_array (numpy.ndarray): x coordinates along the sampled ray
            rz_array (numpy.ndarray): z coordinates along the sampled ray
            axes (:class:`Matplotlib axes <matplotlib.axes.Axes>`): 'axes' instance for current figure
            sub (dict): dictionary of model parameter values to be used for equation substitutions
            i_ray (int): index of this ray among the set currently being plotted
            i_ray_step (int): ray index step, used as divisor when assigning a color to match the parent ray color
            n_arrows (int): number of arrows to plot along the ray
            arrow_sf (float): optional scale factor for arrowhead size
            arrow_offset (int): optional offset from ray initial point to start adding arrowheads
            x_limits (list): optional horizontal axis range
            y_limits (list): optional vertical axis range
            line_style (str): optional line style
            line_width (str): optional line width
            ray_label (str): optional ray label for legend
        """
        # Drop repeated points on vb
        rxz_array = np.unique( np.vstack([rx_array,rz_array]), axis=1 )
        n_pts = rxz_array.shape[1]
        my_arrow_style = mpatches.ArrowStyle.Fancy(head_length=0.99*arrow_sf, head_width=0.6*arrow_sf,
                                                   tail_width=0.01*arrow_sf)
        # color = self.colors[(i_ray//i_ray_step)%self.n_colors]
        color = self.mycolors(i_ray, i_ray_step, n_rays, do_smooth=do_smooth_colors)
        if n_pts>2:
            for q in range(n_pts//(n_arrows)//arrow_offset+n_pts//(n_arrows), n_pts-1, n_pts//(n_arrows)):
                if (y_limits is None or
                        ( (rxz_array[1][q+1]>y_limits[0] and rxz_array[1][q+1]<y_limits[1]) and
                          (rxz_array[1][q]>y_limits[0] and rxz_array[1][q]<y_limits[1]) )) \
                    and \
                    (x_limits is None or
                            ( (rxz_array[0][q+1]>x_limits[0] and rxz_array[0][q+1]<x_limits[1]) and
                              (rxz_array[0][q]>x_limits[0] and rxz_array[0][q]<x_limits[1]) ) ):
                    axes.annotate('', xy=((rxz_array[0][q+1]),(rxz_array[1][q+1])),
                                  xytext=(rxz_array[0][q], rxz_array[1][q]-0),
                                  arrowprops=dict(arrowstyle=my_arrow_style, color=color) )
        plt.plot(rxz_array[0],rxz_array[1], color=color, linestyle=line_style, lw=line_width, label=ray_label)


class OneRayPlots(Graphing):

    def profile_ray( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                        y_limits=None, eta_label_xy=None, n_points=101, aspect=None,
                        do_direct=True, do_schematic=False,
                        do_simple=False, do_t_sampling=True, do_etaxi_label=True,
                        do_pub_label=False, pub_label='', pub_label_xy=[0.15,0.50] ):
        r"""
        Plot a set of erosion rays for a time-invariant topographic profile solution of Hamilton's equations.

        Hamilton's equations are integrated once (from the left boundary to the divide)
        and a time-invariant profile is constructed by repeating
        the ray trajectory, with a suitable truncation and vertical initial offset, multiple times at the left boundary:
        the set of end of truncated rays constitutes the 'steady-state' topographic profile.
        The point of truncation of each trajectory corresponds to the effective time lag imposed by the choice of vertical initial
        offset (which is controlled by the vertical slip rate).

        Visualization of this profile includes: (i) plotting a subsampling of the terminated points of the ray truncations;
        (ii) plotting a continuous curve generated by integrating the surface gradient implied by the erosion-front normal
        covector values :math:`\mathbf{\widetilde{p}}` values generated by solving Hamilton's equations.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            sub (dict): dictionary of model parameter values to be used for equation substitutions
            do_direct (bool): plot directly integrated ray trajectory (from :math:`\mathbf{\widetilde{p}}` values) as a solid curve
            do_schematic (bool):
                optionally plot in more schematic form for expository purposes?
            do_simple (bool): optionally simplify?

        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

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
        self.draw_rays_with_arrows_simple( axes, sub,
                                          t_rsmpld_array, rx_rsmpld_array, rz_rsmpld_array, v_rsmpld_array,
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
                plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                # plt.text(*eta_label_xy, rf'$\eta={gmeq.eta}$',
                         transform=axes.transAxes,
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=13, color='k')
            if do_pub_label:
                plt.text(*pub_label_xy, pub_label,
                         transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')
        if y_limits is not None:
            plt.ylim(*y_limits)

    def profile_h_rays( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                        x_limits=None, y_limits=None, n_points=101,
                        do_direct=True, n_rays=4, profile_subsetting=5,
                        do_schematic=False, do_legend=True, do_profile_points=True, do_fault_bdry=False,
                        do_one_ray=False, do_t_sampling=True, do_etaxi_label=True,
                        do_pub_label=False, pub_label='', pub_label_xy=[0.93,0.33], eta_label_xy=[0.5,0.8] ):
        r"""
        Plot a set of erosion rays for a time-invariant topographic profile solution of Hamilton's equations.

        Hamilton's equations are integrated once (from the left boundary to the divide)
        and a time-invariant profile is constructed by repeating
        the ray trajectory, with a suitable truncation and vertical initial offset, multiple times at the left boundary:
        the set of end of truncated rays constitutes the 'steady-state' topographic profile.
        The point of truncation of each trajectory corresponds to the effective time lag imposed by the choice of vertical initial
        offset (which is controlled by the vertical slip rate).

        Visualization of this profile includes: (i) plotting a subsampling of the terminated points of the ray truncations;
        (ii) plotting a continuous curve generated by integrating the surface gradient implied by the erosion-front normal
        covector values :math:`\mathbf{\widetilde{p}}` values generated by solving Hamilton's equations.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.OneRaySolution`):
                    instance of single ray solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            sub (dict): dictionary of model parameter values to be used for equation substitutions
            do_direct (bool): plot directly integrated ray trajectory (from :math:`\mathbf{\widetilde{p}}` values) as a solid curve
            ray_subsetting (int): optional ray subsampling rate (typically far more rays are computed than should be plotted)
            do_schematic (bool):
                optionally plot in more schematic form for expository purposes?
            do_simple (bool): optionally simplify?

        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        t_array  = gmes.t_array #[::ray_subsetting]
        rx_array = gmes.rx_array #[::ray_subsetting]
        rz_array = gmes.rz_array #[::ray_subsetting]

        if do_t_sampling:
            t_begin, t_end = t_array[0], t_array[-1]
            t_rsmpld_array = np.linspace(t_begin, t_end, n_points)
        else:
            x_rsmpld_array = np.linspace(rx_array[0], rx_array[-1], n_points)
            t_rsmpld_array = gmes.t_interp_x(x_rsmpld_array)
        rx_rsmpld_array = gmes.rx_interp_t(t_rsmpld_array)
        rz_rsmpld_array = gmes.rz_interp_t(t_rsmpld_array)

        # Plot arrow-annotated rays
        self.draw_rays_with_arrows_simple( axes, sub,
                                           t_rsmpld_array, rx_rsmpld_array, rz_rsmpld_array,
                                           n_rays=n_rays, n_t=None,
                                           ls='-' if do_schematic else '-',
                                           sf=0.5 if do_schematic else 1, do_one_ray=do_one_ray )

        if do_schematic:
            # For schematic fig, also plot mirror-image topo profile on opposite side of drainage divide
            self.draw_rays_with_arrows_simple( axes, sub,
                                               t_rsmpld_array, 2-rx_rsmpld_array, rz_rsmpld_array,
                                               n_rays=n_rays, n_t=None,
                                               ls='-' if do_schematic else '-',
                                               sf=0.5 if do_schematic else 1, do_labels=False )

        # # Markers = topo profile from ray terminations
        # if not do_schematic and not do_one_ray:
        #     plt.plot( gmes.x_array[::profile_subsetting], gmes.h_array[::profile_subsetting],
        #                 'k'+('s' if do_profile_points else '-'),
        #                 ms=3, label=r'$T(\mathbf{r})$ from rays $\mathbf{r}(t)$' )

        # Solid line = topo profile from direct integration of gradient array
        if (do_direct or do_schematic) and not do_one_ray:
            plt.plot(gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k',
                     label='$T(\mathbf{r})$' if do_schematic else ('$T(\mathbf{r})$') )
            if do_schematic:
                plt.plot(  gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.26, '0.75', lw=1, ls='--')
                plt.plot(  gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.13, '0.5', lw=1, ls='--')
                plt.plot(2-gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k')
                plt.plot(2-gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.13, '0.5', lw=1, ls='--')
                plt.plot(2-gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0])+0.26, '0.75', lw=1, ls='--')

        axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13 if do_schematic else 16)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=13 if do_schematic else 16)
        if not do_schematic and not do_one_ray and do_legend:
            plt.legend(loc='upper right' if do_schematic else (0.065,0.45),
                       fontsize=9 if do_schematic else 11,
                       framealpha=0.95)
        if not do_schematic:
            if do_etaxi_label:
                plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                         transform=axes.transAxes,
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=16, color='k')
            if do_pub_label:
                plt.text(*pub_label_xy, pub_label,
                         transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')
        if x_limits is not None:
            plt.xlim(*x_limits)
        if y_limits is not None:
            plt.ylim(*y_limits)

        if do_schematic:
            annotate_color = 'k'
            for x_,align_ in [(0.03,'center'), (1.97,'center')]:
                plt.text(x_,0.45, 'rays initiated', #transform=axes.transAxes,
                         rotation=0, horizontalalignment=align_, verticalalignment='center',
                         fontsize=12, color='r')
            plt.text(1,0.53, 'rays annihilated', #transform=axes.transAxes,
                     rotation=0, horizontalalignment='center', verticalalignment='center',
                     fontsize=12, color='0.25')
            for x_,align_ in [(-0.03,'right'), (2.03,'left')]:
                plt.text(x_,0.17, 'fault slip b.c.' if do_fault_bdry else 'const. erosion rate',
                         rotation=90, horizontalalignment=align_, verticalalignment='center',
                         fontsize=12, color='r', alpha=0.7)
            plt.text(0.46,0.38, r'surface isochrone $T(\mathbf{r})=\mathrm{past}$',
                     #transform=axes.transAxes,
                     rotation=12, horizontalalignment='center', verticalalignment='center',
                     fontsize=10, color='0.2')
            plt.text(0.52,0.05, r'surface isochrone $T(\mathbf{r})=\mathrm{now}$',
                     #transform=axes.transAxes,
                     rotation=12, horizontalalignment='center', verticalalignment='center',
                     fontsize=10, color='k')
            for x_,y_,dx_,dy_,shape_ in [(0,0.4, 0,-0.15,'left'), (0,0.25, 0,-0.15,'left'),
                                         (0,0.1, 0,-0.15,'left'), (2,0.4, 0,-0.15,'right'),
                                         (2,0.25, 0,-0.15,'right'),(2,0.1, 0,-0.15,'right')]:
                plt.arrow( x_,y_,dx_,dy_, head_length=0.04, head_width=0.03,
                           length_includes_head=True, shape=shape_,
                           facecolor='r', edgecolor='r' )

    def profile_h( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                    y_limits=None, eta_label_xy=None, n_points=101,
                    do_direct=True,  do_legend=True, do_profile_points=True,
                    profile_subsetting=5,
                    do_t_sampling=True, do_etaxi_label=True,
                    do_pub_label=False, pub_label='', pub_label_xy=[0.93,0.33] ):
        r"""
        Plot a time-invariant topographic profile solution of Hamilton's equations.


        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        t_array  = gmes.t_array #[::ray_subsetting]
        rx_array = gmes.rx_array #[::ray_subsetting]
        rz_array = gmes.rz_array #[::ray_subsetting]
        print()

        if do_t_sampling:
            t_begin, t_end = t_array[0], t_array[-1]
            t_rsmpld_array = np.linspace(t_begin, t_end, n_points)
        else:
            x_rsmpld_array = np.linspace(rx_array[0], rx_array[-1], n_points)
            t_rsmpld_array = gmes.t_interp_x(x_rsmpld_array)
        rx_rsmpld_array = gmes.rx_interp_t(t_rsmpld_array)
        rz_rsmpld_array = gmes.rz_interp_t(t_rsmpld_array)

        # Markers = topo profile from ray terminations
        plt.plot( gmes.x_array[::profile_subsetting], gmes.h_array[::profile_subsetting],
                    'k'+('s' if do_profile_points else '-'),
                    ms=3, label=r'$T(\mathbf{r})$ from rays $\mathbf{r}(t)$' )

        # Solid line = topo profile from direct integration of gradient array
        plt.plot(gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k',
                 label='$T(\mathbf{r})$ by integration' )
        axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=15)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=15)
        if do_legend:
            plt.legend(loc=(0.38,0.75),
                       fontsize=9 if do_schematic else 11,
                       framealpha=0.95)
        if eta_label_xy is None:
            eta_label_xy = (0.92,0.15)
        if do_etaxi_label:
            plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=16, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
                     transform=axes.transAxes, horizontalalignment='center', verticalalignment='center',
                     fontsize=16, color='k')
        if y_limits is not None:
            plt.ylim(*y_limits)

    def alpha_beta( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, aspect=1,
                    n_points=201, x_limits=None, y_limits=None, do_legend=True,
                    do_etaxi_label=True, eta_label_xy=[0.5,0.85],
                    do_pub_label=False, pub_label='', pub_label_xy=[0.88,0.7] ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

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
        ylim = axes.get_ylim()
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
            # plt.text(0.5,0.85, r'$\eta={}$'.format(gmeq.eta),
            plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
            transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')

    def angular_disparity( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                           n_points=201, x_limits=None, y_limits=None, do_legend=True,
                           aspect=0.75,
                           pub_label_xy=[0.5,0.2], eta_label_xy=[0.5,0.81], var_label_xy=[0.85,0.81],
                           do_pub_label=False, pub_label='' ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

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
        plt.text(*eta_label_xy, r'$\eta={}$'.format(gmeq.eta), transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
            transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')

    def profile_angular_disparity( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=201,
                                   pub_label_xy=[0.5,0.2], eta_label_xy=[0.25,0.5], var_label_xy=[0.8,0.35],
                                   do_pub_label=False, pub_label='(a)' ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        x_array = np.linspace(0,1,n_points)
        # x_dbl_array = np.linspace(0,1,n_points*2-1)
        angular_diff_array = np.rad2deg(gmes.alpha_interp(x_array)-gmes.beta_p_interp(x_array))
        plt.plot(x_array,angular_diff_array, 'DarkBlue', ls='-', lw=1.5, label=r'$\alpha(x)-\beta(x)$')
        axes = plt.gca()
        ylim = plt.ylim()
        axes.set_yticks([-30,0,30,60,90])
        axes.set_ylim( -5,95 )
        plt.grid(True, ls=':')

        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13)
        plt.ylabel(r'Anisotropy,  $\psi = \alpha-\beta+90$  [${\degree}$]', fontsize=12)
        if not do_pub_label:
            plt.legend(loc='lower left', fontsize=11, framealpha=0.95)
        plt.text(*pub_label_xy, pub_label if do_pub_label else r'$\eta={}$'.format(gmeq.eta),
                 transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')
        plt.text(*var_label_xy, r'$\psi(x)$' if do_pub_label else '',
                 transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=18, color='k')
        plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                 transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

    def profile_alpha( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=201, do_legend=True):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        x_array = np.linspace(0,1,n_points)
        alpha_array = np.rad2deg(gmes.alpha_interp(x_array))
        plt.plot(x_array,alpha_array-90, 'DarkBlue', ls='-', label=r'$\alpha(x)$')
        x_array = np.linspace(0,1,11)
        pz0_ = gmes.pz0
        alpha_array = [(np.mod(180+np.rad2deg(float(
            sy.atan(gmeq.tanalpha_pxpz_eqn.rhs.subs({px:gmes.px_value(x_,pz0_),pz:pz0_})))),180))
            for x_ in x_array]
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]')
        plt.ylabel(r'Ray dip  $\alpha\!\,$  [${\degree}$ from horiz]')
        plt.grid(True, ls=':')

        if do_legend:
            plt.legend()
        axes = plt.gca()
        plt.text(0.5,0.7, r'$\eta={}$'.format(gmeq.eta), transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

    def profile_v( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=201,
                   pub_label_xy=[0.5,0.5], eta_label_xy=[0.5,0.81], var_label_xy=[0.8,0.5],
                   do_pub_label=False, pub_label='', do_etaxi_label=True,
                   xi_norm=None, legend_loc='lower right', do_mod_v=False ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        if xi_norm is None:
            xi_norm = 1
            rate_label = '${v}$'
        else:
            xi_norm = float(sy.N(xi_norm))
            rate_label = r'${v}/\xi^{\!\rightarrow_{\!\!0}}$  [-]'
        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        x_array = np.linspace(x_min, x_max,n_points)
        t_array  = gmes.t_interp_x(x_array)
        vx_array = gmes.rdotx_interp(x_array)/xi_norm
        vz_array = gmes.rdotz_interp(x_array)/xi_norm
        v_array = np.sqrt(vx_array**2+vz_array**2)
        vx_max = np.max(np.abs(vx_array))
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
            plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
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
                      legend_loc='lower right', do_legend=True, do_mod_vdot=False, do_geodesic=False ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Use an erosion rate (likely vertical xi) to renormalize velocities and accelns (up to T)
        if xi_norm is None:
            xi_norm = 1
            rate_label = '$\dot{v}$'
        else:
            xi_norm = float(sy.N(xi_norm))
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
        vdotx_max = np.max(np.abs(vdotx_array))
        vdotz_max = np.max(np.abs(vdotz_array))
        vdot_max = np.max((vdot_array))

        # Start doing some plotting
        sfx = 1 if np.abs(vdotz_max)<1e-20 else np.power(10,np.round(np.log10(vdotz_max/vdot_max),0))
        label_suffix = '' if sfx==1 else r'$\,\times\,$'+f'{sfx}'
        choice = '_\mathrm{hmltn}'
        vdotx_label = r'$\dot{v}^x'+choice+'(x)$'+label_suffix
        vdotz_label = r'$\dot{v}^z'+choice+'(x)$'
        if do_mod_vdot:
            plt.plot( x_array, vdot_array, 'DarkBlue', ls='-', lw=1.5, label=r'$\dot{v}'+choice+'(x)$')
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
        # if ylim[1]<0: axes.set_ylim(ylim[0],0)
        # if ylim[0]>0: axes.set_ylim(0,ylim[1])
        # axes.set_ylim( -(ylim[1]-ylim[0])/20,ylim[1] )
        axes.set_ylim(*ylim)
        plt.grid(True, ls=':')

        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=14)
        # axes.set_ylim(ylim[0]*1.1,-0)
        if do_legend: plt.legend(loc=legend_loc, fontsize=13, framealpha=0.95)
        if do_etaxi_label:
            plt.text(0.6,0.8, pub_label if do_pub_label else r'$\eta={}$'.format(gmeq.eta),
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

    def prep_g_arrays( self, gmes, gmeq, n_points, do_recompute=False ):
        if not hasattr(self,'x_array') or n_points!=len(self.x_array):
            do_recompute=True
            print('(Re)computing g matrices')

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        self.x_array = np.linspace(x_min, x_max, n_points) \
                        if do_recompute or not hasattr(self,'x_array') else self.x_array
        self.t_array  = gmes.t_interp_x(self.x_array) \
                        if do_recompute or not hasattr(self,'t_array') else self.t_array
        self.rz_array = gmes.rz_interp(self.x_array)  \
                        if do_recompute or not hasattr(self,'rz_array') else self.rz_array
        self.vx_array = gmes.rdotx_interp(self.x_array)  \
                        if do_recompute or not hasattr(self,'vx_array') else self.vx_array
        self.vz_array = gmes.rdotz_interp(self.x_array)  \
                        if do_recompute or not hasattr(self,'vz_array') else self.vz_array
        x_array = self.x_array
        t_array  = self.t_array
        rz_array = self.rz_array
        vx_array = self.vx_array
        vz_array = self.vz_array

        if do_recompute:
            self.gstar_matrices_list = None
            self.gstar_matrices_array = None
            self.g_matrices_list = None
            self.g_matrices_array = None
        if not hasattr(gmeq, 'gstar_ij_mat'): return
        try:
            self.gstar_matrices_list = self.gstar_matrices_list \
                                        if not do_recompute and self.gstar_matrices_list is not None \
                                        else [gmeq.gstar_ij_mat.subs({rx:x_,rdotx:vx_,rdotz:vz_})
                                              for x_,vx_,vz_ in zip(x_array,vx_array,vz_array)]
        except Exception as e:
            print(f'Failed to (re)generate gstar_matrices_list: "{e}"')

        try:
            self.gstar_matrices_array = self.gstar_matrices_array \
                                        if not do_recompute and self.gstar_matrices_array is not None \
                                        else [np.array([float(re(elem_)) for elem_ in g_]).reshape(2,2)
                                              for g_ in self.gstar_matrices_list]
        except Exception as e:
            print(f'Failed to (re)generate gstar_matrices_array: "{e}"')

        try:
            self.g_matrices_list = self.g_matrices_list \
                                        if not do_recompute and self.g_matrices_list is not None \
                                        else [gmeq.g_ij_mat.subs({rx:x_,rdotx:vx_,rdotz:vz_})
                                              for x_,vx_,vz_ in zip(x_array,vx_array,vz_array)]
        except Exception as e:
            print(f'Failed to (re)generate g_matrices_list: "{e}"')

        try:
            self.g_matrices_array = self.g_matrices_array \
                                        if not do_recompute and self.g_matrices_array is not None \
                                        else [np.array([float(re(elem_)) for elem_ in g_]).reshape(2,2)
                                              for g_ in self.g_matrices_list]
        except Exception as e:
            print(f'Failed to (re)generate g_matrices_array: "{e}"')

    def profile_g_properties( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                              y_limits=[None,None],
                              n_points=121, do_pub_label=False, pub_label='',
                              do_gstar=False, do_det=False, do_eigenvectors=False,
                              eta_label_xy=None, do_etaxi_label=True,
                              legend_loc='lower left', do_mod_v=False, do_pv=False,
                              do_recompute=False ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        self.prep_g_arrays(gmes, gmeq, n_points, do_recompute)
        if do_gstar:
            g_matrices_array = self.gstar_matrices_array
        else:
            g_matrices_array = self.g_matrices_array
        x_array = self.x_array
        t_array  = self.t_array
        rz_array = self.rz_array
        vx_array = self.vx_array
        vz_array = self.vz_array

        if do_gstar:
            # Use of lambdified g matrix here fails for eta=1/4, sin(beta) for some reason
            # g_matrices_list = [gmeq.gstar_ij_mat_lambdified(x_,vx_,vz_)
            #                         for x_,vx_,vz_ in zip(x_array,vx_array,vz_array)]
            g_label = '{g^*}'
            m_label = 'co-metric'
            h_label = 'H'
            eta_label_xy = [0.5,0.2] if eta_label_xy is None else eta_label_xy
        else:
            # Use of lambdified g* matrix here fails for eta=1/4, sin(beta) for some reason
            # g_matrices_list = [gmeq.g_ij_mat_lambdified(x_,vx_,vz_)
            #                     for x_,vx_,vz_ in zip(x_array,vx_array,vz_array)]
            g_label = '{g}'
            m_label = 'metric'
            h_label = 'L'
            eta_label_xy = [0.5,0.85] if eta_label_xy is None else eta_label_xy
        # g_eigenvalues_array = np.array([np.real(eig(g_)[0]) for g_ in g_matrices_array])
        # The metric tensor matrices are symmetric therefore Hermitian so we can use 'eigh'
        # print(f'g_matrices_array = {g_matrices_array}')
        if g_matrices_array is not None:
            g_eigh_array = [eigh(g_) for g_ in g_matrices_array]
            g_det_array = np.array([det(g_) for g_ in g_matrices_array])
        else:
            g_eigh_array = None
            g_det_array = None
        if g_eigh_array is not None:
            g_eigenvalues_array = np.real(np.array( [g_eigh_[0] for g_eigh_ in g_eigh_array] ))
            g_eigenvectors_array = np.real(np.array( [g_eigh_[1] for g_eigh_ in g_eigh_array] ))
        else:
            g_eigenvalues_array = None
            g_eigenvectors_array = None
        if do_eigenvectors and g_eigenvectors_array is not None:
            plt.plot( x_array, rz_array, '0.6', ls='-', lw=3, label=r'ray')
            plt.ylabel(r'Eigenvectors of $'+g_label+'$', fontsize=14)
            arrow_sf = 0.5
            my_arrow_style = mpatches.ArrowStyle.Fancy(head_length=0.99*arrow_sf, head_width=0.6*arrow_sf,
                                                       tail_width=0.01*arrow_sf)
            step = 8
            off = 0*step//2
            ev_sf = 0.04
            zipped_arrays = zip(x_array[off::step], rz_array[off::step], g_eigenvectors_array[off::step])
            for x_,rz_,evs_ in zipped_arrays:
                xy_ = np.array([x_,rz_])
                [axes.annotate('', xy=xy_+pm*evs_[0]*ev_sf, xytext=xy_,
                        arrowprops={'arrowstyle':my_arrow_style, 'color':'magenta'} ) for pm in (-1,+1)]
                [axes.annotate('', xy=xy_+pm*evs_[1]*ev_sf, xytext=xy_,
                        arrowprops={'arrowstyle':my_arrow_style, 'color':'DarkGreen'} ) for pm in (-1,+1)]
            plt.plot( 0, 0, 'DarkGreen', ls='-', lw=1.5, label='eigenvector 0')
            plt.plot( 0, 0, 'magenta', ls='-', lw=1.5, label=r'eigenvector 1')
            axes.set_aspect(1)
        elif do_det and g_det_array is not None:
            plt.plot( x_array, g_det_array, 'DarkBlue', ls='-', lw=1.5,
                            label=r'$\det('+g_label+')$')
            plt.ylabel(r'Det of $'+g_label+'$ (Hessian of $'+h_label+'$)', fontsize=14)
        elif do_pv:
            px_array = gmes.px_interp(x_array)
            pz_array = gmes.pz_interp(x_array)
            pv_array = px_array*vx_array + pz_array*vz_array
            plt.plot( x_array, pv_array, 'r', ls='-', lw=2, label=r'$p_i v^i$')
            if self.gstar_matrices_array is not None:
                gstarpp_array = [np.dot(np.dot(gstar_, np.array([px_, pz_])),np.array([px_, pz_]))
                             for gstar_, px_, pz_ in zip(self.gstar_matrices_array, px_array, pz_array)]
                plt.plot( x_array, gstarpp_array, '0.5', ls='--', lw=3, label=r'$g^j p_j p_j$')
            if self.g_matrices_array is not None:
                gvv_array = [np.dot(np.dot(g_, np.array([vx_, vz_])),np.array([vx_, vz_]))
                             for g_, vx_, vz_ in zip(self.g_matrices_array, vx_array, vz_array)]
                plt.plot( x_array, gvv_array, 'k', ls=':', lw=4, label=r'$g_i v^iv^i$')
            plt.ylabel(r'Inner product of $\mathbf{\widetilde{p}}$ and $\mathbf{{v}}$', fontsize=14)
            legend_loc = 'upper left'
        elif g_eigenvalues_array is not None:
            sign_ev0,label_ev0 = (-1,'negative  ') if g_eigenvalues_array[0,0]<0 else (1,'positive  ')
            sign_ev1,label_ev1 = (-1,'negative  ') if g_eigenvalues_array[0,1]<0 else (1,'positive  ')
            plt.yscale('log')
            plt.plot( x_array, sign_ev1*(g_eigenvalues_array[:,1]), 'DarkGreen', ls='-', lw=1.5,
                            label=r''+label_ev1+'$\lambda_'+g_label+'(1)$')
            plt.plot( x_array, sign_ev0*(g_eigenvalues_array[:,0]), 'magenta', ls='-', lw=1.5,
                            label=r''+label_ev0+'$\lambda_'+g_label+'(0)$')
            plt.ylabel(r'Eigenvalues of '+m_label+' tensor $'+g_label+'$', fontsize=12)
        else:
            return

        if do_eigenvectors:
            axes.set_ylim(*y_limits)
        elif do_det:
            ylim = plt.ylim()
            if ylim[1]<0: axes.set_ylim(ylim[0],0)
            if ylim[0]>0: axes.set_ylim(0,ylim[1])
            # axes.set_ylim( -(ylim[1]-ylim[0])/20,ylim[1] )
        elif do_pv:
            axes.set_ylim(0,2)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=14)
        # axes.set_ylim(ylim[0]*1.1,-0)
        plt.legend(loc=legend_loc, fontsize=12, framealpha=0.95)
        if do_etaxi_label:
            plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
            # plt.text(*eta_label_xy, pub_label if do_pub_label else r'$\eta={}$'.format(gmeq.eta),
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')


class TimeInvariantPlots(OneRayPlots):

    def profiles( self, gmes, gmeq, pr_choices, name, fig_size=None, dpi=None,
                        y_limits=None, n_points=101,
                        do_direct=True, n_rays=4, profile_subsetting=5,
                        do_schematic=False, do_legend=True, do_profile_points=True,
                        do_simple=False, do_one_ray=False, do_t_sampling=True, do_etaxi_label=True,
                        do_pub_label=False, pub_label='', pub_label_xy=[0.93,0.33], eta_label_xy=[0.5,0.8] ):
        r"""
        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        def plot_h_profile(eta_,Ci_,idx_,n_, lw=2,dashing='-'):
            sub_ = pr_choices[(eta_,Ci_)]
            mu_ = sub_[mu]
            gmeq_ = gmeq[eta_]
            gmes_ = gmes[(eta_,Ci_)]
            t_array  = gmes_.t_array
            rx_array = gmes_.rx_array
            rz_array = gmes_.rz_array
            Ci_label = rf'{sy.deg(Ci_)}' if sy.deg(Ci_)>=1 else rf'{sy.deg(Ci_).n():0.1}'
            color_ = self.mycolors(idx_, 1, n_, do_smooth=False, cmap_choice='brg')
            plt.plot(gmes_.h_x_array,(gmes_.h_z_array-gmes_.h_z_array[0]), dashing, lw=lw, color=color_,
                     label=rf'$\eta=${eta_}, '+rf'$\mu=${mu_}, '+r'$\mathsf{Ci}=$'+Ci_label+r'$\degree$')

        def make_eta_Ci_list(Ci_choice):
            eta_Ci_list = [(eta_,Ci_) for (eta_,Ci_) in gmes if eta_==Ci_choice]
            return sorted(eta_Ci_list, key=lambda Ci_: Ci_[1], reverse=True)

        eta_Ci_list_1p5 = make_eta_Ci_list(Rational(3,2))
        eta_Ci_list_0p5 = make_eta_Ci_list(Rational(1,2))
        [plot_h_profile(eta_,Ci_,i_,len(eta_Ci_list_1p5), lw=2) for i_,(eta_,Ci_) in enumerate(eta_Ci_list_1p5)]
        [plot_h_profile(eta_,Ci_,i_,len(eta_Ci_list_0p5), lw=3, dashing=':') for i_,(eta_,Ci_) in enumerate(eta_Ci_list_0p5)]

        axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=13 if do_schematic else 16)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=13 if do_schematic else 16)

        plt.legend(loc='upper left', fontsize=11, framealpha=0.95)


        # if not do_schematic and not do_simple:
        #     if do_etaxi_label:
        #         plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
        #                  transform=axes.transAxes,
        #                  horizontalalignment='center', verticalalignment='center',
        #                  fontsize=16, color='k')
        #     if do_pub_label:
        #         plt.text(*pub_label_xy, pub_label,
        #                  transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')
        # if do_schematic:
        #     plt.xlim(-0.15,2.15)
        #     plt.ylim(-0.07,0.62)
        # elif do_simple:
        #     plt.ylim(-0.025,0.38)
        # elif y_limits is not None:
        #     plt.ylim(*y_limits)


    def profile_flow_model( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                            n_points=26, subtitle='', do_subtitling=False, do_extra_annotations=False):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        x_array = np.linspace(0,float(x_1.subs(sub)),n_points)
        varphi_array = [gmeq.varphi_rx_eqn.rhs.subs(sub).subs({rx:x_}) for x_ in x_array]
        varphi_xh1p0_array = [gmeq.varphi_rx_eqn.rhs.subs({x_h:1}).subs(sub).subs({rx:x_}) for x_ in x_array]
        # print(area_array)
        plt.plot( x_array, varphi_array, '-', color=self.colors[0], label='hillslope-channel model' )
        plt.plot( x_array, varphi_xh1p0_array, '--', color=self.colors[0], label='channel-only model' )
        # plt.loglog( area_array, gmes.beta_vt_interp(x_dbl_array), 'ko' )

        # axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Dimensionless horizontal distance, $x/L_{\mathrm{c}}$  [-]')
        plt.ylabel(r'$\varphi(x)$  [-]')
        if do_subtitling:
            # plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
            plt.text(0.1,0.15, r'$\eta={}$'.format(gmeq.eta), transform=axes.transAxes,
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
        # plt.plot(x_h_,varphi_h_,'bo', label='$x_h$')
        plt.plot([x_h_,x_h_],[varphi_h_-30,varphi_h_+70],'b:')
        plt.text(x_h_,varphi_h_+77, '$x_h/L_{\mathrm{c}}$', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='bottom', fontsize=12, color='b')

        plt.legend(loc='upper right', fontsize=11, framealpha=0.95)
        plt.xlim(None,1.05)
        plt.ylim(*y_limits)

    def profile_slope_area( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                            n_points=26, subtitle='', x_min=0.01,
                            do_subtitling=False, do_extra_annotations=False, do_simple=False ):
        """
        Generate a log-log slope-area plot for a time-invariant topographic profile solution
        of Hamilton's equations.

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
            x_min (float): optional x offset
            do_subtitling (bool): optionally annotate with subtitle and eta value?
            do_extra_annotations (bool): optionally annotate with hillslope, channel labels?
            do_simple (bool): optionally avoid hillslope-channel extras?
        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        eta_ = float(gmeq.eta)
        x_array = np.linspace(0,float(x_1.subs(sub)-x_min),n_points)
        x_dbl_array = np.linspace(0,float(x_1.subs(sub)-x_min),n_points*2-1)
        area_array = ((x_1.subs(sub)-x_dbl_array)/x_1.subs(sub))**2
        slope_array = gmes.beta_vt_interp(x_dbl_array)
        if eta_!=1.0 or do_simple:
            plt.loglog( area_array, slope_array, '-' )
        # plt.loglog( area_array, gmes.beta_vt_interp(x_dbl_array), '-' )
        from matplotlib.ticker import ScalarFormatter
        for axis in [axes.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

        eta0p5_slope_array = [\
        0.0094, 0.0096, 0.0098, 0.0100, 0.0102, 0.0104, 0.0107, 0.0109, 0.0112, 0.0115, 0.0117, 0.0120, 0.0123, 0.0126, 0.0129, 0.0133, 0.0136, 0.0140, 0.0143, 0.0147,
        0.0151, 0.0155, 0.0160, 0.0164, 0.0169, 0.0174, 0.0179, 0.0184, 0.0190, 0.0196, 0.0202, 0.0209, 0.0215, 0.0222, 0.0230, 0.0238, 0.0246, 0.0255, 0.0264, 0.0274,
        0.0284, 0.0295, 0.0306, 0.0318, 0.0331, 0.0345, 0.0359, 0.0374, 0.0391, 0.0408, 0.0427, 0.0447, 0.0469, 0.0492, 0.0516, 0.0543, 0.0572, 0.0603, 0.0636, 0.0673,
        0.0713, 0.0756, 0.0803, 0.0855, 0.0912, 0.0974, 0.1043, 0.1120, 0.1205, 0.1299, 0.1405, 0.1523, 0.1655, 0.1804, 0.1973, 0.2163, 0.2379, 0.2625, 0.2902, 0.3217,
        0.3573, 0.3974, 0.4423, 0.4923, 0.5475, 0.6081, 0.6740, 0.7450, 0.8196, 0.8959, 0.9673, 1.0240, 1.0601, 1.0780, 1.0858, 1.0887, 1.0899, 1.0903, 1.0905, 1.0905,
        1.0905]

        eta0p5_area_array = [1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083,
        0.6917, 0.6754, 0.6592, 0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142,
        0.4016, 0.3891, 0.3769, 0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986,
        0.1898, 0.1813, 0.1730, 0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613,
        0.0565, 0.0519, 0.0475, 0.0433, 0.0392, 0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025,
        0.0016, 0.0009, 0.0004, 0.0001]

        eta1p0_slope_array = [\
        0.0960, 0.0970, 0.0981, 0.0991, 0.1002, 0.1013, 0.1025, 0.1036, 0.1048, 0.1060, 0.1073, 0.1085, 0.1098, 0.1112, 0.1125, 0.1139, 0.1153, 0.1168, 0.1183, 0.1199,
        0.1214, 0.1231, 0.1247, 0.1264, 0.1282, 0.1300, 0.1319, 0.1338, 0.1358, 0.1378, 0.1399, 0.1420, 0.1442, 0.1465, 0.1489, 0.1513, 0.1538, 0.1564, 0.1591, 0.1619,
        0.1647, 0.1677, 0.1708, 0.1740, 0.1773, 0.1807, 0.1843, 0.1880, 0.1919, 0.1959, 0.2001, 0.2044, 0.2090, 0.2137, 0.2187, 0.2238, 0.2292, 0.2349, 0.2409, 0.2471,
        0.2537, 0.2606, 0.2679, 0.2756, 0.2837, 0.2922, 0.3013, 0.3109, 0.3211, 0.3319, 0.3434, 0.3557, 0.3688, 0.3828, 0.3979, 0.4141, 0.4314, 0.4501, 0.4703, 0.4922,
        0.5159, 0.5419, 0.5701, 0.6009, 0.6347, 0.6720, 0.7132, 0.7588, 0.8084, 0.8615, 0.9137, 0.9573, 0.9857, 1.0003, 1.0066, 1.0091, 1.0100, 1.0104, 1.0105, 1.0105,
        1.0106]

        eta1p0_area_array = [1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083, 0.6917,
        0.6754, 0.6592, 0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142, 0.4016,
        0.3891, 0.3769, 0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986, 0.1898,
        0.1813, 0.1730, 0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613, 0.0565,
        0.0519, 0.0475, 0.0433, 0.0392,  0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025, 0.0016,
        0.0009, 0.0004, 0.0001]

        eta1p0_xh1p0_slope_array = [\
        0.0960, 0.0969, 0.0979, 0.0988, 0.0998, 0.1008, 0.1018, 0.1028, 0.1039, 0.1049, 0.1060, 0.1071, 0.1083, 0.1094, 0.1106, 0.1118, 0.1131, 0.1143, 0.1156, 0.1170,
        0.1183, 0.1197, 0.1211, 0.1226, 0.1241, 0.1256, 0.1271, 0.1287, 0.1304, 0.1321, 0.1338, 0.1356, 0.1374, 0.1392, 0.1412, 0.1431, 0.1451, 0.1472, 0.1494, 0.1516,
        0.1538, 0.1562, 0.1586, 0.1610, 0.1636, 0.1662, 0.1689, 0.1717, 0.1746, 0.1776, 0.1807, 0.1839, 0.1873, 0.1907, 0.1943, 0.1980, 0.2018, 0.2058, 0.2099, 0.2142,
        0.2187, 0.2233, 0.2281, 0.2332, 0.2385, 0.2439, 0.2497, 0.2557, 0.2620, 0.2686, 0.2756, 0.2828, 0.2905, 0.2985, 0.3070, 0.3159, 0.3253, 0.3352, 0.3458, 0.3569,
        0.3688, 0.3814, 0.3948, 0.4091, 0.4243, 0.4406, 0.4580, 0.4767, 0.4968, 0.5185, 0.5419, 0.5672, 0.5945, 0.6243, 0.6568, 0.6922, 0.7310, 0.7736, 0.8202, 0.8694,
        0.9179]

        eta1p0_xh1p0_area_array = [\
        1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083, 0.6917, 0.6754, 0.6592,
        0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142, 0.4016, 0.3891, 0.3769,
        0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986, 0.1898, 0.1813, 0.1730,
        0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613, 0.0565, 0.0519, 0.0475,
        0.0433, 0.0392, 0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025, 0.0016, 0.0009, 0.0004,
        0.0001]

        eta1p5_slope_array = [\
        0.2049, 0.2063, 0.2077, 0.2092, 0.2106, 0.2121, 0.2136, 0.2151, 0.2167, 0.2183, 0.2199, 0.2216, 0.2233, 0.2250, 0.2267, 0.2285, 0.2303, 0.2322, 0.2340, 0.2360,
        0.2379, 0.2400, 0.2420, 0.2441, 0.2463, 0.2485, 0.2507, 0.2530, 0.2554, 0.2578, 0.2602, 0.2628, 0.2653, 0.2680, 0.2707, 0.2735, 0.2763, 0.2792, 0.2822, 0.2853,
        0.2884, 0.2917, 0.2950, 0.2984, 0.3019, 0.3056, 0.3093, 0.3131, 0.3171, 0.3212, 0.3254, 0.3298, 0.3343, 0.3390, 0.3439, 0.3489, 0.3541, 0.3595, 0.3651, 0.3709,
        0.3769, 0.3832, 0.3898, 0.3966, 0.4038, 0.4112, 0.4190, 0.4272, 0.4358, 0.4448, 0.4543, 0.4643, 0.4748, 0.4860, 0.4977, 0.5102, 0.5236, 0.5380, 0.5533, 0.5697,
        0.5873, 0.6064, 0.6270, 0.6495, 0.6742, 0.7015, 0.7318, 0.7656, 0.8027, 0.8432, 0.8837, 0.9186, 0.9414, 0.9534, 0.9584, 0.9605, 0.9613, 0.9615, 0.9616, 0.9617,
        0.9617]

        eta1p5_area_array = [1.0000, 0.9803, 0.9608, 0.9415, 0.9224, 0.9035, 0.8847, 0.8662, 0.8479, 0.8297, 0.8118, 0.7941, 0.7765, 0.7592, 0.7420, 0.7251, 0.7083,
        0.6917, 0.6754, 0.6592, 0.6432, 0.6274, 0.6118, 0.5964, 0.5813, 0.5663, 0.5515, 0.5368, 0.5224, 0.5082, 0.4942, 0.4804, 0.4668, 0.4533, 0.4401, 0.4271, 0.4142,
        0.4016, 0.3891, 0.3769, 0.3648, 0.3530, 0.3413, 0.3298, 0.3185, 0.3075, 0.2966, 0.2859, 0.2754, 0.2651, 0.2550, 0.2451, 0.2354, 0.2259, 0.2166, 0.2075, 0.1986,
        0.1898, 0.1813, 0.1730, 0.1648, 0.1569, 0.1492, 0.1416, 0.1342, 0.1271, 0.1201, 0.1134, 0.1068, 0.1004, 0.0942, 0.0883, 0.0825, 0.0769, 0.0715, 0.0663, 0.0613,
        0.0565, 0.0519, 0.0475, 0.0433, 0.0392, 0.0354, 0.0318, 0.0284, 0.0251, 0.0221, 0.0192, 0.0166, 0.0141, 0.0119, 0.0098, 0.0080, 0.0063, 0.0048, 0.0035, 0.0025,
        0.0016, 0.0009, 0.0004, 0.0001]

        if eta_==1.0 and not do_simple:
            plt.loglog(eta1p5_area_array,eta1p5_slope_array,'-', label=r'$\eta=3/2\,,\,\,x_h/L_{\mathrm{c}}=0.9$', color=self.colors[1])
            plt.loglog(eta0p5_area_array,eta0p5_slope_array,'-', label=r'$\eta=1/2\,,\,\,x_h/L_{\mathrm{c}}=0.9$', color=self.colors[2])
            plt.loglog(eta1p0_area_array,eta1p0_slope_array,'-', label=r'$\eta=1\,,\,\,x_h/L_{\mathrm{c}}=0.9$', color=self.colors[0])
            plt.loglog(eta1p0_xh1p0_area_array,eta1p0_xh1p0_slope_array,'--', label=r'channel-only model', color=self.colors[0])

        # axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(r'Dimensionless area, $(L_{\mathrm{c}}-x)^2/L_{\mathrm{c}}^2$  [-]')
        plt.ylabel(r'Slope, $|\tan\beta|$  [-]')
        # plt.legend(loc='upper left', fontsize=11, framealpha=0.95)

        if do_subtitling:
            plt.text(0.1,0.15, r'$\eta={}$'.format(gmeq.eta), transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center', fontsize=12, color='k')
            plt.text(0.05,0.25, subtitle, transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center', fontsize=12, color='k')
        if do_extra_annotations:
            plt.text(0.29,0.83, 'hillslope', transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=12, color='0.2')
            plt.text(0.735,0.49, 'channel', transform=axes.transAxes,
                     rotation=-58.5, horizontalalignment='center', verticalalignment='center', fontsize=12, color='0.2')

        y_limits = axes.get_ylim()
        x_h_ = x_h.subs(sub)
        area_h_ = float(( ((x_1-x_h_)/x_1)**2 ).subs(sub))
        slope_h_ = gmes.beta_vt_interp(float(x_h_))
        plt.plot([area_h_,area_h_],[slope_h_*0.45,slope_h_*1.5],'b:')
        if area_h_>0.0:
            plt.text(area_h_,slope_h_*0.4, r'$\dfrac{(L_{\mathrm{c}}-x_h)^2}{L_{\mathrm{c}}^2}$', #transform=axes.transAxes,
                         horizontalalignment='center', verticalalignment='top', fontsize=12, color='b')

        if not do_simple:
            plt.legend(loc='lower left', fontsize=11, framealpha=0.95)
        plt.xlim(None,1.15)
        plt.ylim(*y_limits)

        return area_array, slope_array

    def profile_aniso(self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                      y_limits=None, eta_label_xy=[0.5,0.8], v_scale=0.4, v_exponent=1,
                      sf=None, n_points=51, n_arrows=26, xf_stop=0.995,
                      do_pub_label=False, pub_label='' ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        x_array = np.linspace(0,1*xf_stop,n_points)
        h_array = gmes.h_interp(x_array)

        profile_lw = 1
        # Solid line = topo profile from direct integration of gradient array
        plt.plot(gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k', lw=profile_lw, label='$h(x)'+'\quad\eta={}$'.format(gmeq.eta))
        plt.plot(x_array, h_array, 'o', mec='k', mfc='gray', ms=3, fillstyle='full', markeredgewidth=0.5)
        # plt.plot(gmes.h_x_array,(gmes.h_z_array-gmes.h_z_array[0]), 'k' )

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


        plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
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
                sfy = float(sy.N((9-4.3)*(gmeq.eta-0.5)+4.3))*0.3
                sf = (sfy,sfy)
            else:
                sfy = float(sy.N((9-1.0)*(gmeq.eta-0.5)+1.0))*0.5
                sf = (sfy,sfy)
                plt.xlim(-0.03,1.05)
        if y_limits is None:
            plt.ylim(ylim[0]*sf[0],ylim[1]-ylim[0]*sf[1])
        else:
            plt.ylim(*y_limits)

        # Hand-made legend

        class ArrivalTime(object):
            pass

        class HandlerArrivalTime(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch = mpatches.Arrow(4,4,20,0, width=0, lw=profile_lw, ec='k', fc='k')
                handlebox.add_artist(patch)
                return patch

        class RayPoint(object):
            pass

        class HandlerRayPoint(object):
            def __init__(self,fc='gray'):
                super().__init__()
                self.fc = fc

            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                patch = mpatches.Circle((15, 4), radius=2.5, ec='k', fc=self.fc)
                handlebox.add_artist(patch)
                return patch

        class RayArrow(object):
            pass

        class HandlerRayArrow(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                color_ = aniso_colors[3]
                patch = mpatches.FancyArrow(0, 0.5*height, width, 0,
                                            length_includes_head=True,
                                            head_width=0.75*height,
                                            overhang=0.1,
                                            fc=color_, ec=color_)
                handlebox.add_artist(patch)
                return patch

        class NormalStick(object):
            pass

        class HandlerNormalStick(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
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
        legend_labels1 = [ r'$T(\mathbf{r})$', '$\mathbf{r}$', '$\mathbf{v}$', '$\mathbf{\widetilde{p}}$']
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
                      legend_loc='upper left', do_etaxi_label=True, eta_label_xy=[0.6,0.8],
                      do_pub_label=False, pub_label='', pub_label_xy=[0.88,0.7] ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

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
            plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                 transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label,
            transform=axes.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=16, color='k')

    def profile_beta_error(self, gmes, gmeq, sub, name, fig_size=None, dpi=None, n_points=101,
                           pub_label_xy=[0.5,0.2], eta_label_xy=[0.5,0.8], xf_stop=0.995):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

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
        plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                 transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

    def profile_xi( self, gmes, gmeq, sub, name, fig_size=None, dpi=None, xf_stop=1, n_points=201,
                    pub_label_xy=[0.5,0.2], eta_label_xy=[0.5,0.5], var_label_xy=[0.8,0.5],
                    do_etaxi_label=True, do_pub_label=False, pub_label='(a)', xi_norm=None ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        if xi_norm is None:
            xi_norm = 1
            rate_label = r'$\xi^{\!\perp\!}$'
        else:
            xi_norm = float(sy.N(xi_norm))
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
                 r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$'
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
                              pub_label_xy=[0.5,0.2], eta_label_xy=[0.55,0.81], var_label_xy=[0.85,0.81],
                              do_etaxi_label=True, do_pub_label=False, pub_label='(d)', xi_norm=None ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        if xi_norm is None:
            xi_norm = 1
            rate_label = r'$\xi^{\!\rightarrow}$'
        else:
            xi_norm = float(sy.N(xi_norm))
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
                 r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$'
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
                            pub_label_xy=[0.5,0.2], eta_label_xy=[0.5,0.81], var_label_xy=[0.85,0.81],
                            do_etaxi_label=True, do_pub_label=False, pub_label='(e)', xi_norm=None ):
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
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        if xi_norm is None:
            xi_norm = 1
            rate_label = r'$\xi^{\!\downarrow}$'
        else:
            xi_norm = float(sy.N(xi_norm))
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
        ylim = plt.ylim()
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
                 r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$'
                    if do_etaxi_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')
        plt.text(*var_label_xy, r'$\xi^{\downarrow}(x)$',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=18, color='k')
        plt.text(*pub_label_xy, pub_label if do_pub_label else '',
                 transform=axes.transAxes, horizontalalignment='center',
                 verticalalignment='center', fontsize=16, color='k')


class TimeDependentPlots(Graphing):

    rp_list = ['rx','rz','px','pz']
    rpt_list = rp_list+['t']

    def profile_isochrones( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                            do_zero_isochrone=True, do_overlay=False, fig=None,
                            do_rays=True, ray_subsetting=5, ray_lw=0.5, ray_ls='-', ray_label='ray',
                            do_isochrones=True, isochrone_subsetting=1, do_isochrone_p=False,
                            isochrone_lw=0.5, isochrone_ls='-',
                            do_annotate_rays=False, n_arrows=10, arrow_sf=0.7, arrow_offset=4,
                            do_annotate_cusps=False, cusp_lw=1.5, do_smooth_colors=False,
                            x_limits=(-0.001,1.001), y_limits=(-0.025,0.525),
                            aspect=None,
                            do_legend=True, do_alt_legend=False, do_grid=True,
                            do_infer_initiation=True,
                            do_etaxi_label=True, eta_label_xy=[0.65,0.85],
                            do_pub_label=False, pub_label=None, pub_label_xy=[0.5,0.92] ):
        """
        Plot xxxx.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.VelocityBoundarySolution`):
                    instance of velocity boundary solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            sub (dict):
                dictionary of model parameter values to be used for equation substitutions
            do_zero_isochrone (bool):
                optional plot initial surface?
            do_rays (bool):
                optional plot rays?
            ray_subsetting (int):
                optional rate of ray subsetting
            ray_lw (float):
                optional ray line width
            ray_ls (float):
                optional ray line style
            ray_label (float):
                optional ray line label
            do_isochrones (bool):
                optional plot isochrones?
            isochrone_subsetting (int):
                optional rate of isochrone subsetting
            do_isochrone_p (bool):
                optional plot isochrone herringbones?
            isochrone_lw (float):
                optional isochrone line width
            isochrone_ls (float):
                optional isochrone line style
            do_annotate_rays (bool):
                optional plot arrowheads along rays?
            n_arrows (int):
                optional number of arrowheads to annotate along rays or cusp-line
            arrow_sf (float):
                optional scale factor for arrowhead sizes
            arrow_offset (int):
                optional offset to start annotating arrowheads
            do_annotate_cusps (bool):
                optional plot line to visualize cusp initiation and propagation
            cusp_lw (float):
                optional cusp propagation curve line width
            x_limits (list of float):
                optional [x_min, x_max] horizontal plot range
            y_limits (list of float):
                optional [z_min, z_max] vertical plot range
            aspect (float):
                optional figure aspect ratio
            do_legend (bool):
                optional plot legend
            do_alt_legend (bool):
                optional plot slightly different legend
            do_grid (bool):
                optional plot dotted grey gridlines
            do_infer_initiation (bool):
                optional draw dotted line inferring cusp initiation at the left boundary
        """
        if do_overlay:
            fig = fig
        else:
            fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Unpack for brevity
        if hasattr(gmes,'rpt_isochrones'):
            rx_isochrones, rz_isochrones, px_isochrones, pz_isochrones, t_isochrones \
                = [gmes.rpt_isochrones[rpt_] for rpt_ in self.rpt_list]

        # Initial boundary
        if hasattr(gmes,'rpt_isochrones') and do_zero_isochrone:
            n_isochrones = len(rx_isochrones)
            plt.plot(rx_isochrones[0],rz_isochrones[0], '-', color=self.gray_color(0, n_isochrones), lw=2,
                      label=('zero isochrone' if do_legend else None) )

        # Rays
        axes = plt.gca()
        if do_rays:
            n_rays = len(gmes.rpt_arrays['rx'])
            for i_ray,(rx_array,rz_array,t_array) in enumerate(zip(reversed(gmes.rpt_arrays['rx']),
                                                                   reversed(gmes.rpt_arrays['rz']),
                                                                   reversed(gmes.rpt_arrays['t']))):
                if (i_ray//ray_subsetting-i_ray/ray_subsetting)==0:
                    this_ray_label=(ray_label+' ($t_{\mathrm{oldest}}$)' if i_ray==0 else
                                    ray_label+' ($t_{\mathrm{newest}}$)' if i_ray==n_rays-1 else
                                    None)
                    if do_annotate_rays:
                        self.arrow_annotate_ray_custom(rx_array, rz_array, axes, sub, i_ray, ray_subsetting, n_rays,
                                                       n_arrows, arrow_sf, arrow_offset,
                                                       x_limits=x_limits, y_limits=y_limits,
                                                       line_style=ray_ls, line_width=ray_lw,
                                                       ray_label=this_ray_label,
                                                       do_smooth_colors=do_smooth_colors)
                    else:
                        plt.plot(rx_array,rz_array, lw=ray_lw,
                                 color=self.mycolors(i_ray, ray_subsetting, n_rays, do_smooth=do_smooth_colors), linestyle=ray_ls,
                                 label=this_ray_label)

        # Time slices or isochrones of erosion front
        if hasattr(gmes,'rpt_isochrones') and do_isochrones:
            n_isochrones = len(rx_isochrones)
            delta_t = t_isochrones[1]
            for i_isochrone,(rx_isochrone,rz_isochrone,t_) in enumerate(zip(rx_isochrones,rz_isochrones,
                                                                                            t_isochrones)):
                i_subsetted = (i_isochrone//isochrone_subsetting-i_isochrone/isochrone_subsetting)
                i_subsubsetted = (i_isochrone//(isochrone_subsetting*10)-i_isochrone/(isochrone_subsetting*10))
                if (i_isochrone>0 and i_subsetted==0 and rx_isochrone is not None):
                    plt.plot(rx_isochrone, rz_isochrone, self.gray_color(i_isochrone, n_isochrones),
                             linestyle=isochrone_ls, lw=1.3*isochrone_lw if i_subsubsetted==0 else 0.5*isochrone_lw)
            # Hack legend items
            if (rx_isochrone is not None):
                plt.plot(rx_isochrone, rz_isochrone, self.gray_color(i_isochrone, n_isochrones),
                         linestyle=isochrone_ls, lw=1.3*isochrone_lw, label=r'isochrone $\Delta{\hat{t}}='+f'{int(10*delta_t)}'+'$')
                plt.plot(rx_isochrone, rz_isochrone, self.gray_color(i_isochrone, n_isochrones),
                         linestyle=isochrone_ls, lw=0.5*isochrone_lw, label=r'isochrone $\Delta{\hat{t}}='+f'{round(delta_t,1)}'+'$')
                if do_isochrone_p:
                    for (rx_lowres,rz_lowres,px_lowres,pz_lowres) \
                            in zip(rx_isochrones_lowres,rz_isochrones_lowres,
                                        px_isochrones_lowres,pz_isochrones_lowres):
                        [plt.arrow(rx_,rz_,0.02*px_/np.sqrt(px_**2+pz_**2),0.02*pz_/np.sqrt(px_**2+pz_**2),
                               ec=color_, fc=color_, lw=0.5*isochrone_lw,
                               head_width=0.015, head_length=0, overhang=0)
                     for (rx_,rz_,px_,pz_) in zip(rx_lowres, rz_lowres, px_lowres, pz_lowres)]

        # Knickpoint aka cusp propagation
        if do_annotate_cusps:
            rxz_array =  np.array([rxz for (t,rxz),_,_ in gmes.trxz_cusps if rxz!=[] and rxz[0]<=1.01])
            if (n_cusps := len(rxz_array)) > 0:
                # Plot locations of cusps as a propagation curve
                plt.plot(rxz_array.T[0][:-1],rxz_array.T[1][:-1], lw=cusp_lw, color='r', alpha=0.4,
                                label='cusp propagation')

                # Plot inferred initiation of cusp propagation from LHS boundary
                if do_infer_initiation:
                    ((x1_,y1_),(x2_,y2_)) = rxz_array[0:2]
                    dx_, dy_ = (x2_-x1_)/100, (y2_-y1_)/100
                    x0_ = 0
                    y0_ = y1_+(dy_/dx_)*(x0_-x1_)
                    plt.plot( (x0_,x1_), (y0_,y1_), ':', lw=cusp_lw, color='r', alpha=0.4,
                                    label='inferred initiation' )

                # Plot arrow annotations using subset of cusps
                cusp_subsetting = max(1,n_cusps//n_arrows)
                rxz_array =  rxz_array[::cusp_subsetting].T
                sf = 0.7
                my_arrow_style = mpatches.ArrowStyle.Fancy(head_length=.99*sf,
                                                            head_width=.6*sf, tail_width=0.0001)
                for ((x1_,y1_),(x0_,y0_)) in zip(rxz_array.T[:-1],rxz_array.T[1:]):
                    dx_, dy_ = (x1_-x0_)/100, (y1_-y0_)/100
                    if (x_limits is None or  (x0_>=x_limits[0] if x_limits[0] is not None else True) \
                                         and (x0_<=x_limits[1] if x_limits[1] is not None else True) ):
                        axes.annotate('', xy=(x0_,y0_),  xytext=(x0_+dx_,y0_+dy_),
                                  arrowprops=dict(arrowstyle=my_arrow_style, color='r', alpha=0.4 ))

        # Label axes
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=16)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=16)

        if do_legend or do_alt_legend:
            plt.legend(loc='upper left', fontsize=10, framealpha=0.95)

        # Tidy axes etc
        plt.xlim(*x_limits)
        plt.ylim(*y_limits)
        if do_grid: plt.grid(True, ls=':')
        axes.set_aspect(1 if aspect is None else aspect)

        if do_etaxi_label:
            plt.text(*eta_label_xy, r'$\eta='+rf'{gmeq.eta}'+r'\quad\mathsf{Ci}='+rf'{round(float(sy.deg(Ci.subs(sub))))}'+'{\degree}$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label, transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

        return fig


    def profile_cusp_speed( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                            sample_spacing=10, x_limits=[-0.05,1.05], t_limits=[0,None], y_limits=[-5,None],
                            legend_loc='lower right', do_x=True, do_infer_initiation=True ):
        """
        Plot horizontal speed of cusp propagation

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmes (:class:`~.ode_raytracing.VelocityBoundarySolution`):
                    instance of velocity boundary solution class defined in :mod:`~.ode_raytracing`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            sample_spacing (int): sample interval between cusps over which to compute the speed
            x_limits (list of float):
                optional [x_min, x_max] horizontal plot range
            y_limits (list of float):
                optional [z_min, z_max] vertical plot range
            do_x (bool):
                optional plot x-axis as dimensionless horizontal distance :math:`x/L_{\mathrm{c}}`;
                otherwise plot as time :math:`t`
            do_infer_initiation (bool):
                optional draw dotted line inferring cusp initiation at the left boundary

        Todo:
            implement `do_infer_initiation`
        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Drop last cusp because it's sometimes bogus
        trxz_cusps =  gmes.trxz_cusps
        x_or_t_array = gmes.cusps['rxz'][:-1][:,0] if do_x else gmes.cusps['t'][:-1]
        vc_array = np.array( [(x1_-x0_)/(t1_-t0_) for (x0_,z0),(x1_,z1),t0_,t1_
                            in zip(gmes.cusps['rxz'][:-1], gmes.cusps['rxz'][1:],
                                   gmes.cusps['t'][:-1], gmes.cusps['t'][1:])] )

        plt.plot(x_or_t_array, vc_array, '.', ms=7, label='measured')

        if do_x:
            plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]')
        else:
            plt.xlabel(r'Time, $t$')
        plt.ylabel(r'Cusp horiz propagation speed,  $c^x$')

        axes = plt.gca()
        plt.text(0.15,0.2, r'$\eta={}$'.format(gmeq.eta), transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

        x_array = np.linspace(0.001 if do_infer_initiation else x_or_t_array[0],1,num=101)
        # color_cx, color_bounds = 'DarkGreen', 'Green'
        color_cx, color_bounds = 'Red', 'DarkRed'
        plt.plot(x_array, gmes.cx_pz_lambda(x_array), color=color_cx, alpha=0.8, lw=2, label=r'$c^x$ model ($p_z$)' )
        # plt.plot(x_array, gmes.cx_pz_tanbeta_lambda(x_array), ':', color='g', alpha=0.8, lw=2, label=r'$c^x$ model ($\tan\beta$)' )
        plt.plot(x_array, gmes.cx_v_lambda(x_array), ':', color='k', alpha=0.8, lw=2, label=r'$c^x$ model ($\mathbf{v}$)' )
        plt.plot(x_array, gmes.vx_interp_fast(x_array), '--', color=color_bounds, alpha=0.8, lw=1,
                        label='fast ray $v^x$ bound' )
        plt.plot(x_array, gmes.vx_interp_slow(x_array), '-.', color=color_bounds, alpha=0.8, lw=1,
                        label='slow ray $v^x$ bound' )

        xlim = plt.xlim(*x_limits) if do_x else plt.xlim(*t_limits)
        ylim = plt.ylim(*y_limits) if y_limits is not None else None
        plt.grid(True, ls=':')

        plt.legend(loc=legend_loc, fontsize=12, framealpha=0.95)


class TheoryPlots(Graphing):

    def comparison_logpolar(self, gmeq, name, fig_size=None, dpi=None,
                            varphi_=1, n_points=100,
                            idtx_pz_min=1e-3, idtx_pz_max=1000,
                            fgtx_pz_min=1e-3, fgtx_pz_max=1000,
                            y_limits=None, do_beta_crit=True):
        """
        Plot both indicatrix and figuratrix on one log-polar graph.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        plt.title('Erosion indicatrix & figuratrix', va='bottom', pad=15)
        plt.grid(False, ls=':')

        # px_pz_lambda = lambdify( [pz], gmeq.fgtx_px_pz_varphi_eqn.rhs.subs({varphi:varphi_}) )
        # fgtx_pz_array = -np.power(10,np.linspace(np.log10(1000),np.log10(1e-6),n_points))
        # fgtx_px_array = np.array([float(re(sy.N(px_pz_lambda(pz_)))) for pz_ in fgtx_pz_array])

        # Compute figuratrix
        fgtx_px_array, fgtx_pz_array, _ \
            = self.figuratrix(gmeq, varphi_, n_points, fgtx_pz_min, fgtx_pz_max)
        fgtx_p_array = np.log10(np.sqrt(fgtx_px_array**2+fgtx_pz_array**2))
        fgtx_tanbeta_array = -fgtx_px_array/fgtx_pz_array
        fgtx_beta_array = np.arctan(fgtx_tanbeta_array)
        fgtx_theta_array = np.concatenate([fgtx_beta_array, (2*np.pi-fgtx_beta_array)[::-1]])
        fgtx_p_array = np.concatenate([fgtx_p_array,fgtx_p_array[::-1]])

        # Compute indicatrix
        idtx_rdotx_array,idtx_rdotz_array, _,_ \
            = self.indicatrix(gmeq, varphi_, n_points, idtx_pz_min, idtx_pz_max)
        idtx_rdot_array = np.log10(np.sqrt(idtx_rdotx_array**2+idtx_rdotz_array**2))
        idtx_alpha_array = np.arctan(-idtx_rdotx_array/idtx_rdotz_array)
        idtx_theta_negrdot_array = idtx_alpha_array[idtx_rdotz_array<0]
        idtx_rdot_negrdot_array = idtx_rdot_array[idtx_rdotz_array<0]
        idtx_theta_posrdot_array = np.pi+idtx_alpha_array[idtx_rdotz_array>0]
        idtx_rdot_posrdot_array = idtx_rdot_array[idtx_rdotz_array>0]
        idtx_theta_array = np.concatenate([idtx_theta_negrdot_array, idtx_theta_posrdot_array])
        idtx_rdot_array = np.concatenate([idtx_rdot_negrdot_array,idtx_rdot_posrdot_array])

        # Compute reference unit circle
        unit_circle_beta_array = np.linspace(0, 2*np.pi)

        # Do the basic plotting
        # idtx_label = r'indicatrix $F(\mathbf{v},\alpha)$'
        # fgtx_label = r'figuratrix $F^*(\mathbf{\widetilde{p}},\beta)$'
        idtx_label = r'ray velocity'
        fgtx_label = r'normal slowness'
        plt.polar( fgtx_theta_array, fgtx_p_array, 'DarkBlue', '-', lw=1.5, label=fgtx_label )
        plt.polar( idtx_theta_array, idtx_rdot_array, 'DarkRed', ls='-', lw=1.5, label=idtx_label)
        plt.polar( 2*np.pi-idtx_theta_array, idtx_rdot_array, 'DarkRed', ls='-', lw=1.5)
        # plt.polar( idtx_theta_posrdot_array, idtx_rdot_posrdot_array, 'magenta', ls='-', lw=3)
        # plt.polar( idtx_theta_negrdot_array, idtx_rdot_negrdot_array, 'k', ls='-', lw=3)
        plt.polar(unit_circle_beta_array, unit_circle_beta_array*0, 'g', ':', lw=1, label='unit circle')

        # Critical angles
        beta_crit = np.arctan(float(gmeq.tanbeta_crit))
        alpha_crit = np.pi/2+np.arctan(float(gmeq.tanalpha_crit))
        plt.polar([beta_crit,beta_crit], [-2,3], ':', color='DarkBlue', label=fr'$\beta=\beta_c$')
        plt.polar([alpha_crit,alpha_crit], [-2,3], ':', color='DarkRed', label=fr'$\alpha=\alpha_c$')

        # Labelling etc
        posn_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
        xtick_posns = [(-np.pi/i_ if i_ != 0 else np.pi) for i_ in posn_list]
        xtick_posns = plt.xticks()[0]
        xtick_labels = [(r'$\frac{\pi}{' + '{}'.format(int(x_)) + '}$' if x_ > 0
                         else r'$-\frac{\pi}{' + '{}'.format(int(-x_)) + '}$' if x_ != 0
                         else r'$0$')
                       for x_ in posn_list]
        xtick_labels = [
            r'$\beta=0$',
            r'$\beta=\frac{\pi}{4}$',
            r'$\qquad\alpha=0,\beta=\frac{\pi}{2}$',
            r'$\alpha=\frac{\pi}{4}$',
            r'$\alpha=\pm\frac{\pi}{2}$',
            r'$\alpha=-\frac{\pi}{4}$',
            r'$\alpha=0,\beta=-\frac{\pi}{2}\quad$',
            r'$\beta=-\frac{\pi}{4}$'
        ]
        plt.xticks(xtick_posns, xtick_labels)

        if y_limits is None:
            y_limits = [-2,3]
            ytick_posns = [-1,0,1,2,3,4]
        else:
            ytick_posns = list(range(int(y_limits[0]),int(y_limits[1])+2))
        plt.ylim(*y_limits)
        ytick_labels = [r'$10^{' + '{}'.format(int(y_)) + '}$' for y_ in ytick_posns]
        plt.yticks(ytick_posns, ytick_labels)

        axes = plt.gca()
        plt.text(np.pi/1.1,(1+y_limits[1]-y_limits[0])*2/3+y_limits[0], '$\eta={}$'.format(gmeq.eta),fontsize=16)

        axes.set_theta_zero_location("S")

        handles, labels = axes.get_legend_handles_labels()
        subset = [3,2,0,5,6]
        handles = [handles[idx] for idx in subset]
        labels = [labels[idx] for idx in subset]
        # Hacked fix to bug in Matplotlib that adds a bogus entry here
        axes.legend(handles, labels, loc='upper left', framealpha=1)

    @staticmethod
    def text_labels(gmeq, varphi_, px_, pz_, rdotx_, rdotz_, zoom_factor, do_text_labels):
        xy_ = (2.5,-2) if zoom_factor>=1 else (1.25,0.9)
        plt.text(*xy_, r'$\varphi={}\quad\eta={}$'.format(varphi_,gmeq.eta),
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')
        if do_text_labels:
            plt.text(rdotx_*0.97, rdotz_+0.2, r'$\mathbf{v}$',
                     horizontalalignment='center', verticalalignment='center', fontsize=15, color='r')
            sf = 1.2 if varphi_>=0.5 else 0.85
            plt.text(px_+0.25,pz_*sf, r'$\mathbf{\widetilde{p}}$',
                     horizontalalignment='center', verticalalignment='center', fontsize=15, color='b')

    @staticmethod
    def arrows(px_, pz_, rdotx_, rdotz_):
        p_ = np.sqrt(px_**2+pz_**2)
        # Arrow for rdot direction
        plt.arrow(0, 0, rdotx_,rdotz_, head_width=0.08, head_length=0.16, lw=1,
                  length_includes_head=True, ec='r', fc='r')
        # Fishbone for p direction
        np_ = int(0.5+((p_)*5)) if p_>=0.2 else 1
        hw_ = 0.2 #if do_half else 0.2
        plt.arrow(px_,pz_,-px_,-pz_, color='b',
                head_width=hw_, head_length=-0.1, lw=1,
                shape='full', overhang=1,
                length_includes_head=True,
                head_starts_at_zero=False,
                ec='b')
        for i_head in list(range(1,np_)):
            len_head = i_head/(np_)
            plt.arrow(0, 0, px_*len_head, pz_*len_head,
                    head_width=hw_, head_length=0, lw=1,
                    shape='full', overhang=0,
                    length_includes_head=True,
                    ec='b')
        len_head = 1
        plt.arrow(0, 0, px_*len_head, pz_*len_head,
                head_width=hw_*2.5, head_length=0, lw=2,
                shape='full', overhang=0,
                length_includes_head=True,
                ec='b')

    @staticmethod
    def lines_and_points(pd, axes, zoomx, do_pz, do_shapes):
        # Unpack
        varphi_ = pd['varphi']
        n_vertices_, ls_ = pd['n_vertices'], pd['ls']
        px_, pz_, rdotx_, rdotz_ = pd['px'], pd['pz'], pd['rdotx'], pd['rdotz']
        pz_min_, pz_max_ = pd['pz_min'], pd['pz_max']
        psqrd_ = (px_**2+pz_**2)
        plt.plot([px_/psqrd_,rdotx_], [pz_/psqrd_, rdotz_], c='r', ls=ls_, lw=1)
        plt.plot([0, px_], [0, pz_], c='b', ls=ls_, lw=1)
        if do_pz:
            plt.plot([zoomx[0], zoomx[1]], [pz_, pz_], c='b', ls=':', lw=1)
        if psqrd_<1:
            plt.plot([px_,px_/psqrd_], [pz_,pz_/psqrd_], c='b', ls=ls_, lw=1)
        if do_shapes:
            axes.add_patch( RegularPolygon( (px_,pz_), n_vertices_, radius=0.08,
                            lw=1, ec='k', fc='b') )
            axes.add_patch( RegularPolygon( (rdotx_,rdotz_), n_vertices_, radius=0.08,
                            lw=1, ec='k', fc='r') )
        else:
            axes.add_patch( Circle( (px_,pz_), radius=0.04, lw=1, ec='k', fc='b') )
            axes.add_patch( Circle( (rdotx_,rdotz_), radius=0.04, lw=1, ec='k', fc='r') )

    @staticmethod
    def annotations(beta_, tanalpha_):
        # alpha arc
        plt.text(0.45, -0.05, r'$\alpha$', color='DarkRed', #transform=axes.transAxes,
                 fontsize=12, horizontalalignment='center', verticalalignment='center')
        axes.add_patch( mpatches.Arc((0, 0), 1.2, 1.2, color='DarkRed',
                            linewidth=0.5, fill=False, zorder=2,
                            theta1=270, theta2= 90+np.rad2deg(np.arctan(tanalpha_))) )
        # beta arc
        plt.text(0.08, -0.30, r'$\beta$', color='DarkBlue', #transform=axes.transAxes,
                 fontsize=10, horizontalalignment='center', verticalalignment='center')
        axes.add_patch( mpatches.Arc((0, 0), 0.9, 0.9, color='DarkBlue',
                            linewidth=0.5, fill=False, zorder=2,
                            theta1=270, theta2=270+np.rad2deg(beta_)) )
        # "faster" direction
        plt.text(3.05,1.6, 'faster', color='r', #transform=axes.transAxes,
                 fontsize=12, rotation=55, horizontalalignment='center', verticalalignment='center')
        # "faster" direction
        plt.text(1.03,-2, 'slower', color='b', #transform=axes.transAxes,
                 fontsize=12, rotation=-85, horizontalalignment='center', verticalalignment='center')

    @staticmethod
    def legend(gmeq, axes, do_legend, do_half, do_ray_slowness=False):
        if not do_legend: return
        handles, labels = axes.get_legend_handles_labels()
        if gmeq.eta>=1:
            loc_='center right' if do_half else 'upper left' if do_ray_slowness  else 'lower left'
        else:
            loc_='upper right' if do_half else 'upper left' if do_ray_slowness else 'lower left'
        axes.legend(handles[::-1], labels[::-1], loc=loc_)

    def figuratrix(self, gmeq, varphi_, n_points, pz_min=1e-5, pz_max=50):
        px_pz_eqn = Eq( px, factor(gmeq.fgtx_px_pz_varphi_eqn.rhs.subs({varphi:varphi_})) )
        px_pz_lambda = lambdify( [pz], re(sy.N(px_pz_eqn.rhs)) )
        fgtx_pz_array = -np.power(10,np.linspace(np.log10(pz_max),np.log10(pz_min), n_points))
        # fgtx_px_array = np.array([float(re(sy.N(px_pz_eqn.rhs.subs({pz:pz_})))) for pz_ in fgtx_pz_array])
        fgtx_px_array = np.array([float(px_pz_lambda(pz_)) for pz_ in fgtx_pz_array])
        return fgtx_px_array, fgtx_pz_array, px_pz_eqn

    def indicatrix(self, gmeq, varphi_, n_points, pz_min=1e-5, pz_max=300):
        rdotx_pz_eqn = gmeq.idtx_rdotx_pz_varphi_eqn.subs({varphi:varphi_})
        rdotz_pz_eqn = gmeq.idtx_rdotz_pz_varphi_eqn.subs({varphi:varphi_})
        rdotx_pz_lambda = lambdify( [pz], re(sy.N(rdotx_pz_eqn.rhs)) )
        rdotz_pz_lambda = lambdify( [pz], re(sy.N(rdotz_pz_eqn.rhs)) )
        fgtx_pz_array = -np.power(10,np.linspace(np.log10(pz_max),np.log10(pz_min), n_points))
        # idtx_rdotx_array = np.array([float(re(sy.N(rdotx_pz_eqn.rhs.subs({pz:pz_}))))
        #                                 for pz_ in fgtx_pz_array])
        # idtx_rdotz_array = np.array([float(re(sy.N(rdotz_pz_eqn.rhs.subs({pz:pz_}))))
        #                                 for pz_ in fgtx_pz_array])
        idtx_rdotx_array = np.array([float(rdotx_pz_lambda(pz_)) for pz_ in fgtx_pz_array])
        idtx_rdotz_array = np.array([float(rdotz_pz_lambda(pz_)) for pz_ in fgtx_pz_array])
        return idtx_rdotx_array, idtx_rdotz_array, rdotx_pz_eqn, rdotz_pz_eqn

    @staticmethod
    def plot_figuratrix(fgtx_px_array, fgtx_pz_array, maybe_recip_fn, do_ray_slowness=False):
        # label = r'normal speed' if do_ray_slowness else r'figuratrix $F^*$'
        label = r'normal velocity' if do_ray_slowness else r'normal slowness'
        plt.plot( *maybe_recip_fn(-fgtx_px_array, fgtx_pz_array), lw=2, c='DarkBlue', ls='-',
                                        label=label)
        plt.plot( *maybe_recip_fn( fgtx_px_array, fgtx_pz_array), lw=2, c='DarkBlue', ls='-')

    @staticmethod
    def plot_indicatrix(idtx_rdotx_array,idtx_rdotz_array, maybe_recip_fn, do_ray_slowness=False):
        # label = r'ray slowness' if do_ray_slowness else r'indicatrix $F$'
        label = r'ray slowness' if do_ray_slowness else r'ray velocity'
        plt.plot( *maybe_recip_fn( idtx_rdotx_array, idtx_rdotz_array), lw=2, c='DarkRed', ls='-',
                                        label=label)
        plt.plot( *maybe_recip_fn(-idtx_rdotx_array, idtx_rdotz_array), lw=2, c='DarkRed', ls='-')

    @staticmethod
    def plot_unit_circle(do_varphi_circle):
        unit_circle_beta_array = np.linspace(0, np.pi)
        plt.plot(np.cos(unit_circle_beta_array*2), np.sin(unit_circle_beta_array*2),
                 lw=1, c='g', ls='-', label='unit circle')
        if do_varphi_circle:
            plt.plot(varphi_*np.cos(unit_circle_beta_array*2), varphi_*np.sin(unit_circle_beta_array*2),
                     lw=1, c='g', ls=':', label=r'$\varphi={}$'.format(varphi_))

    def relative_geometry( self, gmeq, name, fig_size=None, dpi=None,
                           varphi_=1, zoom_factor=1,
                           do_half=False, do_annotations=True, do_legend=True,
                           do_text_labels=True, do_arrows=True, do_lines_points=True,
                           do_shapes=False, do_varphi_circle=False, do_pz=False,
                           do_ray_slowness=False,
                           x_max=3.4, n_points=100, pz_min=1e-1 ):
        r"""
        Plot the loci of :math:`\mathbf{\widetilde{p}}` and :math:`\mathbf{r}` and
        their behavior defined by :math:`F` relative to the :math:`\xi` circle.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            varphi_choice (int): the value of :math:`\varphi` to use
            zoom_factor (float): fractional zoom factor relative to preassigned graph limits
            do_half (bool): plot only for :math:`x\geq0`?
            do_annotations (bool): plot annotated arcs, faster/slower arrows?
            do_legend (bool): display the legend?
            do_text_labels (bool): display text annotations?
            do_arrows (bool): display vector/covector arrows?
            do_lines_points (bool): display dashed/dotted tangent lines etc?
            do_shapes (bool): plot key points using polygon symbols rather than simple circles
            do_varphi_circle (bool): plot the :math:`\xi` (typically unit) circle?
            do_pz (bool): plot horizontal dotted line indicating magnitude of :math:`p_z`
            do_ray_slowness (bool): invert ray speed (but keep ray direction) in indicatrix?
        """

        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Adjust plot scale, limits
        axes = plt.gca()
        axes.set_aspect(1)
        plt.grid(True, ls=':')
        x_min = -0.1 if do_half else -x_max
        y_minmax = 2.4 if gmeq.eta >= 1 else 2.4
        zoomx = np.array([x_min, x_max]) * zoom_factor
        zoomz = np.array([-y_minmax, y_minmax]) * zoom_factor
        plt.xlim(zoomx)
        plt.ylim(zoomz)
        plt.xlabel(r'Horizontal component')
        plt.ylabel(r'Vertical component')

        # Prep
        recip_fn = lambda x_, z_: [x_/(x_**2+z_**2), z_/(x_**2+z_**2)]
        null_fn  = lambda x_, z_: [x_, z_]
        maybe_recip_fn = recip_fn if do_ray_slowness else null_fn
        points_tangents_dicts = {
            0.1  : {'n_vertices' : 3, 'ls' : ':',  'pz_min' : pz_min, 'pz_max' : 1000},
            0.15 : {'n_vertices' : 3, 'ls' : ':',  'pz_min' : pz_min, 'pz_max' : 1000},
            0.5  : {'n_vertices' : 4, 'ls' : '--', 'pz_min' : pz_min, 'pz_max' : 100},
            1    : {'n_vertices' : 4, 'ls' : '--', 'pz_min' : pz_min, 'pz_max' : 100},
            1.3  : {'n_vertices' : 5, 'ls' : '-.', 'pz_min' : pz_min, 'pz_max' : 10},
            2    : {'n_vertices' : 5, 'ls' : '-.', 'pz_min' : pz_min, 'pz_max' : 10},
            3    : {'n_vertices' : 5, 'ls' : '-.', 'pz_min' : pz_min, 'pz_max' : 3}
        }

        # Compute some stuff
        pdict = points_tangents_dicts[varphi_]
        fgtx_px_array, fgtx_pz_array, px_pz_eqn \
            = self.figuratrix(gmeq, varphi_, n_points)
        idtx_rdotx_array,idtx_rdotz_array, rdotx_pz_eqn, rdotz_pz_eqn \
            = self.indicatrix(gmeq, varphi_, n_points, pz_min=pdict['pz_min'], pz_max=pdict['pz_max'])
        pz_ = -np.cos(np.pi/4)
        px_ = float(sy.N(re(px_pz_eqn.rhs.subs({pz:pz_}))))
        rdotx_ = float(re(rdotx_pz_eqn.rhs.subs({pz:pz_})))
        rdotz_ = float(re(rdotz_pz_eqn.rhs.subs({pz:pz_})))
        tanalpha_ = (-rdotx_/rdotz_)
        pdict.update({'varphi': varphi_, 'px' : px_, 'pz' : pz_, 'rdotx' : rdotx_, 'rdotz' : rdotz_})

        # Do the plotting
        self.plot_indicatrix(idtx_rdotx_array, idtx_rdotz_array, maybe_recip_fn, do_ray_slowness)
        self.plot_figuratrix(fgtx_px_array, fgtx_pz_array, maybe_recip_fn, do_ray_slowness)
        self.plot_unit_circle(do_varphi_circle)
        if do_lines_points: self.lines_and_points(pdict, axes, zoomx, do_pz, do_shapes)
        self.text_labels(gmeq, varphi_, px_, pz_, rdotx_, rdotz_, zoom_factor, do_text_labels)
        if do_arrows: self.arrows(px_, pz_, rdotx_, rdotz_)
        if do_annotations: self.annotations(beta_, tanalpha_)
        self.legend(gmeq, axes, do_legend, do_half, do_ray_slowness)

    def alpha_beta(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ):
        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # fig = gr.create_figure(job_name+'_alpha_beta', fig_size=(6,3))
        plt.plot(beta_array, alpha_array, 'b');
        x_,y_ = np.rad2deg(np.arctan(float(tanbeta_crit_))), np.rad2deg(np.arctan(float(tanalpha_crit_)))
        plt.plot( x_,y_, 'ob' )
        plt.text( x_,y_-y_/9, r'$\beta_c, \,\alpha_c$', color='b', horizontalalignment='center', fontsize=14)
        eta_label = fr'$\eta = ${gmeq.eta}'
        plt.text( 40,y_/4, fr'$\eta = ${gmeq.eta}', color='k', horizontalalignment='center', fontsize=14)
        plt.text( 87,y_*0.67, '(a)' if gmeq.eta==Rational(3,2) else ('(b)' if gmeq.eta==Rational(1,2) else ''),
                  color='k', horizontalalignment='center', fontsize=16 )
        plt.grid('on')
        plt.xlabel(r'Surface tilt  $\beta$   [${\degree}\!$ from horiz]')
        plt.ylabel(r'Ray angle  $\alpha$   [${\degree}\!$ from horiz]');
        np.rad2deg(np.arctan(float(tanbeta_crit_))), np.rad2deg(np.arctan(float(tanalpha_crit_)));

    def beta_anisotropy(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ):
        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(beta_array, alpha_array-beta_array+90, 'b');
        x_,y_ = np.rad2deg(np.arctan(float(tanbeta_crit_))), np.rad2deg(np.arctan(float(tanalpha_crit_)))
        plt.plot( x_,y_-x_+90, 'ob' )
        if gmeq.eta<1:
            plt.text( x_*(1.0 if gmeq.eta<Rational(1,2) else 1.0),
                     (y_-x_+90)*(1.15), r'$\beta_c$', color='b', horizontalalignment='center', fontsize=14)
        else:
            plt.text( x_*1,(y_-x_+90)*(0.85), r'$\beta_c$', color='b', horizontalalignment='center', fontsize=14)
        eta_label = fr'$\eta = ${gmeq.eta}'
        plt.text( 75,55, fr'$\eta = ${gmeq.eta}', color='k',
                 horizontalalignment='center', verticalalignment='center', fontsize=15)
        if gmeq.eta==Rational(3,2):
            plt.text( 30,15, '(a)', color='k', horizontalalignment='center', verticalalignment='center', fontsize=17 )
        elif gmeq.eta==Rational(1,2):
            plt.text( 30,15, '(b)', color='k', horizontalalignment='center', verticalalignment='center', fontsize=17 )
        plt.grid('on')
        plt.xlabel(r'Surface normal angle  $\beta$   [${\degree}\!$ from vertical]')
        plt.ylabel(r'Anisotropy   $\psi = \alpha-\beta+90{\degree}$   [${\degree}\!$]');
        axes = plt.gca()
        axes.set_aspect(1)
        plt.xlim(0,90)
        plt.ylim(0,90);
        plt.plot(beta_array, beta_array, ':');

    def alpha_anisotropy(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ):
        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(alpha_array-beta_array+90, alpha_array, 'b');
        eta_label = fr'$\eta = ${gmeq.eta}'
        if gmeq.eta>1:
            plt.text( 40,5, fr'$\eta = ${gmeq.eta}', color='k',
                     horizontalalignment='center', verticalalignment='center', fontsize=15)
        else:
            plt.text( 40,-5, fr'$\eta = ${gmeq.eta}', color='k',
                     horizontalalignment='center', verticalalignment='center', fontsize=15)
        if gmeq.eta==Rational(3,2):
            plt.text( 10,7.5, '(a)', color='k', horizontalalignment='center', verticalalignment='center', fontsize=17 )
        elif gmeq.eta==Rational(1,2):
            plt.text( 7,-16.5, '(b)', color='k', horizontalalignment='center', verticalalignment='center', fontsize=17 )
        plt.grid('on')
        plt.ylabel(r'Ray angle  $\alpha$   [${\degree}\!$]')
        plt.xlabel(r'Anisotropy   $\psi = \alpha-\beta+90{\degree}$   [${\degree}\!$]');
        axes = plt.gca()
        axes.invert_xaxis()
        axes.set_aspect(2)

    def alpha_image(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ):
        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(beta_array, beta_array-(alpha_array-beta_array+90), 'b');
        x_,y_ = np.rad2deg(np.arctan(float(tanbeta_crit_))), np.rad2deg(np.arctan(float(tanalpha_crit_)))
        plt.plot( x_,x_-(y_-x_+90), 'ob' )
        plt.text( x_,-15, r'$\beta_c$', color='b', horizontalalignment='center', fontsize=14)
        eta_label = fr'$\eta = ${gmeq.eta}'
        plt.text( 40,62.5, fr'$\eta = ${gmeq.eta}', color='k',
                 horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.text( 70,-62.5, '(a)' if gmeq.eta==Rational(3,2) else '(b)', color='k', horizontalalignment='center', verticalalignment='center', fontsize=17 )
        plt.grid('on')
        plt.ylabel(r'Image ray angle  $\beta-\psi$   [${\degree}\!$ from vertical]')
        plt.xlabel(r'Surface normal angle  $\beta$   [${\degree}\!$ from vertical]')
        axes = plt.gca()

    def indicatrix_prep(self, gmeq, pr, sub_, varphi_=1):
        self.H_parametric_eqn = Eq((2*gmeq.H_eqn.rhs)**2,1).subs({varphi_r:varphi_, xiv:xiv_0}).subs(sub_)

        if pr.model.eta==Rational(3,2):
            pz_min_eqn = Eq(pz_min,
                (solve(Eq( ((solve(Eq(4*gmeq.H_eqn.rhs**2,1)
                                   .subs({varphi_r:varphi}),px**2)[2]).args[0].args[0].args[0])**2, 0)
                  ,pz**4)[0])**Rational(1,4))
            px_min_eqn = Eq(px_min,
                    solve(simplify(gmeq.H_eqn.subs({varphi_r:varphi})
                                   .subs({pz:pz_min_eqn.rhs})).subs({H:Rational(1,2)}),px)[0] )
            tanbeta_max_eqn = Eq(tan(beta_max), ((px_min/pz_min).subs(e2d(px_min_eqn))).subs(e2d(pz_min_eqn)))
            self.tanbeta_max = float(N(tanbeta_max_eqn.rhs))
        else:
            pz_min_eqn = Eq(pz_min, 0)
            px_min_eqn = Eq(px_min,
                            sqrt(solve(Eq((
                            solve(Eq(4*gmeq.H_eqn.rhs**2,1).subs({varphi_r:varphi}),pz**2)[:])[0],0)
                                           ,px**2)[1]))
            tanbeta_max_eqn = Eq(tan(beta_max),sy.oo)
            self.tanbeta_max = None

        px_min_ = round(float(N(px_min_eqn.rhs.subs({varphi:varphi_}))),4)
        pz_min_ = round(float(N(pz_min_eqn.rhs.subs({varphi:varphi_}))),8)
        px_min_, -pz_min_, np.rad2deg(np.arctan(px_min_/pz_min_)) if pz_min_>0 else None

        # px_H_solns = [simplify(soln) for soln in solve(Eq((4*H_parametric_eqn.lhs)**2, (4*H_parametric_eqn.rhs)**2),px**2)]
        # x,z = symbols('x, z')
        px_H_solns = [simplify(sqrt(soln)) for soln in solve( self.H_parametric_eqn ,px**2)]
        pz_H_solns = [simplify(sqrt(soln)).subs({Abs(px):px})
                      for soln in solve( self.H_parametric_eqn ,pz**2)]
        px_H_soln_ = [soln for soln in px_H_solns if Abs(sy.im(N(soln.subs({pz:1}))))<1e-10][0]
        self.px_H_lambda = lambdify( [pz], simplify(px_H_soln_) )

        if pr.model.eta==Rational(3,2):
            pz_max_ = 10**4
        else:
            pz_max_ = 10**2
        pz_array = -10**np.linspace(np.log10(pz_min_ if pz_min_>0 else 1e-6), np.log10(pz_max_), 1000)
        px_array = self.px_H_lambda(pz_array)
        p_array = np.vstack([px_array,pz_array]).T
        modp_array = norm(p_array,axis=0)
        # np.rad2deg(np.arctan(-px_array[0]/pz_array[0])), np.rad2deg(np.arctan(gmeq.tanbeta_crit)), \
        #     Eq(beta_crit, round(N(sy.deg(atan(gmeq.tanbeta_crit_eqn.rhs))),2))
        tanbeta_crit = float(N(gmeq.tanbeta_crit_eqn.rhs))

        self.p_infc_array = p_array[np.abs(p_array[:,0]/p_array[:,1])<tanbeta_crit]
        self.p_supc_array = p_array[np.abs(p_array[:,0]/p_array[:,1])>=tanbeta_crit]

        px_poly_eqn = sy.poly(self.H_parametric_eqn)
        px_poly_lambda = lambdify( [px,pz], px_poly_eqn.as_expr() )
        dpx_poly_lambda = lambdify( [px,pz], diff(px_poly_eqn.as_expr(),px) )

        px_solutions = lambda px_guess_: np.array([
            (root_scalar( px_poly_lambda, args=(pz_),fprime=dpx_poly_lambda,
                         method='newton', x0=px_guess_ )).root for pz_ in p_array[:,1]])
        px_newton_array = px_solutions(10)

        pz_ = 2
        px_ = self.px_H_lambda(pz_)

        v_from_gstar_lambda_tmp = lambdify((px,pz),
                        N(gmeq.gstar_varphi_pxpz_eqn.subs({varphi_r:varphi_}).rhs*Matrix([px,pz])))
        self.v_from_gstar_lambda = lambda px_,pz_: (v_from_gstar_lambda_tmp(px_,pz_)).flatten()

        v_ = (self.v_from_gstar_lambda(px_,pz_))
        dp_supc_array = p_array[(-p_array[:,0]/p_array[:,1])>=gmeq.tanbeta_crit]
        v_lambda = lambda pa: np.array([(self.v_from_gstar_lambda(px_,pz_)) for px_,pz_ in pa])
        self.v_infc_array = v_lambda(self.p_infc_array)
        self.v_supc_array = v_lambda(self.p_supc_array)
        v_array = v_lambda(p_array)

    def Fstar_F_rectlinear(self, gmeq, job_name, pr, do_zoom=False, fig_size=None):
        name = job_name+'_Fstar_F_rectlinear'+('_zoom' if do_zoom else '')
        fig = self.create_figure(name, fig_size)

        eta_ = pr.model.eta
        if do_zoom:
            if eta_==Rational(3,2):
                plt.xlim(0.98,1.07)
                plt.ylim(0.15,0.23)
                eta_xy_label = [0.2,0.85]
            else:
                plt.xlim(0.7,1.2)
                plt.ylim(-0.4,0)
                eta_xy_label = [0.8,0.8]
        else:
            if eta_==Rational(3,2):
                plt.xlim(0,2)
                plt.ylim(-4,0.6)
                eta_xy_label = [0.7,0.8]
            else:
                plt.xlim(0,2.5)
                plt.ylim(-2,0)
                eta_xy_label = [0.8,0.7]

        # Critical, bounding angles
        if eta_==Rational(3,2):
            pz_max_ = -1.5
        else:
            pz_max_ = -1.5
        px_abmax_ = -pz_max_*(self.tanbeta_max if self.tanbeta_max is not None else 1)
        pz_abmax_ = pz_max_
        vx_abmax_,vz_abmax_ = self.v_from_gstar_lambda(px_abmax_,pz_abmax_)
        px_abcrit_ = -pz_max_*gmeq.tanbeta_crit
        pz_abcrit_ = pz_max_
        vx_abcrit_,vz_abcrit_ = self.v_from_gstar_lambda(px_abcrit_,pz_abcrit_)

        # Lines visualizing critical, bounding angles: ray velocity
        if eta_>1:
            plt.plot([0,vx_abmax_],[0,vz_abmax_],
                     '-', color='r', alpha=0.4, lw=2, label=r'$\alpha_{\mathrm{lim}}$')

        # Indicatrix aka F=1 for rays
        plt.plot(self.v_supc_array[:,0],self.v_supc_array[:,1],
                 'r' if eta_>1 else 'DarkRed',
                 lw=2, ls='-',
                 label=r'$F=1$,  $\beta\geq\beta_\mathrm{c}$')
        plt.plot([0,vx_abcrit_],[0,vz_abcrit_], '-.',
                 color='DarkRed' if eta_>1 else 'r',
                 lw=1, label=r'$\alpha_{\mathrm{c}}$')
        plt.plot(self.v_infc_array[:,0],self.v_infc_array[:,1],
                 'DarkRed' if eta_>1 else 'r',
                 lw=1 if eta_==Rational(3,2) and not do_zoom else 2, ls='-',
                 label=r'$F=1$,  $\beta<\beta_\mathrm{c}$')

        # Lines visualizing critical, bounding angles: normal slowness
        if eta_==Rational(3,2) and not do_zoom:
            plt.plot(np.array([0,px_abmax_]),[0,pz_abmax_],
                     '-b', alpha=0.4, lw=1.5, label=r'$\beta_{\mathrm{max}}$')

        # Figuratrix aka F*=1 for surfaces
        if not do_zoom:
            plt.plot(self.p_supc_array[:,0],self.p_supc_array[:,1],
                     'b' if eta_>1 else 'DarkBlue', lw=2, ls='-',
                     label=r'$F^*\!\!=1$,  $\beta\geq\beta_\mathrm{c}$')
            plt.plot([0,px_abcrit_],[0,pz_abcrit_],
                     '--',
                     color='DarkBlue' if eta_>1 else 'b',
                     lw=1, label=r'$\beta_{\mathrm{c}}$')
            plt.plot(self.p_infc_array[:,0],self.p_infc_array[:,1],
                     'DarkBlue' if eta_>1 else 'b', lw=2, ls='-',
                     label=r'$F^*\!\!=1$,  $\beta<\beta_\mathrm{c}$')

        pz_ = -float(solve(self.H_parametric_eqn.subs({px:pz*(gmeq.tanbeta_crit)}),pz)[0])
        px_ = self.px_H_lambda(pz_)
        vx_,vz_ = self.v_from_gstar_lambda(px_,pz_)
        if eta_!=Rational(3,2):
            plt.plot([vx_],[-vz_], 'o', color='r', ms=5)
        if not do_zoom:
            plt.plot([px_],[-pz_], 'o', color='DarkBlue' if eta_>1 else 'b', ms=5)

        plt.xlabel(r'$p_x$ (for $F^*$)  or  $v^x$ (for $F$)', fontsize=14)
        plt.ylabel(r'$p_z$ (for $F^*$)  or  $v^z$ (for $F$)', fontsize=14)

        axes = plt.gca()
        axes.set_aspect(1)
        plt.text(*eta_xy_label, rf'$\eta={gmeq.eta}$', transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=15, color='k')

        if eta_==Rational(3,2):
            if do_zoom:
                plt.legend(loc='lower right')
            else:
                plt.legend(loc='lower left')
        else:
            if do_zoom:
                plt.legend(loc='upper left')
            else:
                plt.legend(loc='lower right')

        if do_zoom:
            if eta_>1:
                plt.text(*[1.025,0.184], 'convex', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=40, fontsize=15, color='DarkRed')
                plt.text(*[1.054,0.208], 'concave', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=60, fontsize=11, color='r')
            else:
                plt.text(*[1.07,-0.264], 'concave', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=15, fontsize=15, color='r')
                plt.text(*[0.955,-0.15], 'convex', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=70, fontsize=15, color='DarkRed')
        else:
            if eta_>1:
                plt.text(*[0.7,-0.05], 'convex', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=11, fontsize=15, color='DarkRed')
                plt.text(*[1.15,0.42], 'concave', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=60, fontsize=10, color='r')
                plt.text(*[1.4,-0.72], 'concave', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=10, fontsize=15, color='b')
                plt.text(*[1.5,-2.3], 'convex', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=-80, fontsize=15, color='DarkBlue')
            else:
                plt.text(*[1.3,-0.26], 'concave', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=10, fontsize=13, color='r')
                plt.text(*[0.9,-0.14], 'convex', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=65, fontsize=11, color='DarkRed')
                plt.text(*[0.98,-0.65], 'convex', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=75, fontsize=15, color='DarkBlue')
                plt.text(*[0.66,-1.65], 'concave', #transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=75, fontsize=15, color='b')
        plt.grid(True, ls=':')

    def Fstar_F_polar(self, gmeq, job_name, pr, do_zoom=False, fig_size=None):
        name = job_name+'_Fstar_F_polar'
        fig = self.create_figure(name, fig_size)

        eta_ = pr.model.eta

        if eta_>1:
            r_min_ = 0.1
            r_max_ = 100
            scale_fn = lambda a: np.log10(a)
            alpha_fn = lambda a: np.pi-a
        else:
            r_min_ = 0.1
            r_max_ = 10
            scale_fn = lambda a: np.log10(a)
            alpha_fn = lambda a: a
        v_scale_fn = lambda v: scale_fn(v)*1

        # Lines visualizing critical, bounding angles: ray velocity
        if eta_>1:
            plt.polar([np.pi/2+(np.arctan(gmeq.tanalpha_crit))]*2,
                      [scale_fn(r_min_),scale_fn(r_max_)], '-',
                      color='r' if eta_>1 else 'DarkRed',
                      alpha=0.4, lw=2, label=r'$\alpha_{\mathrm{lim}}$')
        plt.polar(alpha_fn(np.arcsin(self.v_supc_array[:,0]/norm(self.v_supc_array, axis=1))),
                  v_scale_fn(norm(self.v_supc_array, axis=1)),
                  'r' if eta_>1 else 'DarkRed',
                  label=r'$F=1$,  $\beta\geq\beta_\mathrm{c}$')
        plt.polar([np.pi/2+(np.arctan(gmeq.tanalpha_crit))]*2,
                  [scale_fn(r_min_),scale_fn(r_max_)], '-.',
                  color='DarkRed' if eta_>1 else 'r',
                  lw=1, label=r'$\alpha_{\mathrm{c}}$')
        plt.polar(alpha_fn(np.arcsin(self.v_infc_array[:,0]/norm(self.v_infc_array, axis=1))),
                  v_scale_fn(norm(self.v_infc_array, axis=1)),
                  'DarkRed' if eta_>1 else 'r',
                  lw=None if eta_==Rational(3,2) else None,
                  label=r'$F=1$,  $\beta<\beta_\mathrm{c}$')

        unit_circle_array = np.array([[theta_,1] for theta_ in np.linspace(0,(np.pi/2)*1.2,100)])
        plt.polar(unit_circle_array[:,0], scale_fn(unit_circle_array[:,1]), '-',
                  color='g', lw=1, label='unit circle')

        if eta_>1:
            plt.polar([np.arctan(self.tanbeta_max)]*2, [scale_fn(r_min_),scale_fn(r_max_)],
                      '-', color='b', alpha=0.3, lw=1.5, label=r'$\beta_{\mathrm{max}}$')
        plt.polar(np.arcsin(self.p_supc_array[:,0]/norm(self.p_supc_array, axis=1)),
                      scale_fn(norm(self.p_supc_array, axis=1)),
                  'b' if eta_>1 else 'DarkBlue',
                  label=r'$F^*\!\!=1$,  $\beta\geq\beta_\mathrm{c}$')
        plt.polar([np.arctan(gmeq.tanbeta_crit)]*2, [scale_fn(r_min_),scale_fn(r_max_)], '--',
                  color='DarkBlue' if eta_>1 else 'b',
                  lw=1, label=r'$\beta_{\mathrm{c}}$')
        plt.polar(np.arcsin(self.p_infc_array[:,0]/norm(self.p_infc_array, axis=1)),
                  scale_fn(norm(self.p_infc_array, axis=1)),
                  'DarkBlue' if eta_>1 else 'b',
                  label=r'$F^*\!\!=1$,  $\beta<\beta_\mathrm{c}$')

        plt.polar(
            (np.arcsin(self.p_supc_array[-1,0]/norm(self.p_supc_array[-1]))
             +np.arcsin(self.p_infc_array[0,0]/norm(self.p_infc_array[0])))/2,
            (scale_fn(norm(self.p_infc_array[0]))+scale_fn(norm(self.p_supc_array[-1])))/2, 'o',
            color='DarkBlue' if eta_>1 else 'b')

        axes = plt.gca()
        axes.set_theta_zero_location('S')
        horiz_label = r'$\log_{10}{p}$  or  $\log_{10}{v}$'
        vert_label = r'$\log_{10}{v}$  or  $\log_{10}{p}$'

        if eta_>1:
            theta_max_ = 20
            axes.set_thetamax(90+theta_max_)
            axes.text(np.deg2rad(85+theta_max_),0.5, vert_label,
                        rotation=theta_max_, ha='center', va='bottom', fontsize=15)
            axes.text(np.deg2rad(-8),1.2, horiz_label,
                        rotation=90, ha='right', va='bottom', fontsize=15)
            theta_list = [0, 1/6, 2/6, 3/6, np.deg2rad(110)/np.pi]
            xtick_labels = [
                r'$\beta=0^{\!\circ}$',
                r'$\beta=30^{\!\circ}$',
                r'$\beta=60^{\!\circ}$',
                r'$\alpha=0^{\!\circ}$',
                r'$\alpha=20^{\!\circ}$',
            ]
            eta_xy_label = [1.15,0.9]
            legend_xy = [1,0]
            plt.text(*[(np.pi/2)*1.07,0.4], 'convex', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=8, fontsize=15, color='DarkRed')
            plt.text(*[(np.pi/2)*1.17,0.28], 'concave', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=13, fontsize=11, color='r')
            plt.text(*[(np.pi/3)*0.925,0.5], 'concave', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=-35, fontsize=15, color='b')
            plt.text(*[(np.pi/6)*0.7,0.85], 'convex', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=68, fontsize=15, color='DarkBlue')
        else:
            theta_max_ = 0
            axes.set_thetamax(90+theta_max_)
            axes.text(np.deg2rad(92+theta_max_),axes.get_rmax()/5, vert_label,
                        rotation=theta_max_, ha='right', va='bottom', fontsize=15)
            axes.text(np.deg2rad(-8),axes.get_rmax()/5, horiz_label,
                        rotation=90, ha='right', va='bottom', fontsize=15)
            theta_list = [0, 1/6, 2/6, 3/6]
            xtick_labels = [
                r'$\beta=0^{\!\circ}$',
                r'$\beta=30^{\!\circ}$',
                r'$\beta=60^{\!\circ}\!\!,\, \alpha=-30^{\!\circ}$',
                r'$\beta=90^{\!\circ}\!\!,\, \alpha=0^{\!\circ}$',
            ]
            eta_xy_label = [1.2,0.75]
            legend_xy = [0.9,0]
            plt.text(*[(np.pi/2)*0.94,0.4], 'concave', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=11, fontsize=15, color='r')
            plt.text(*[(np.pi/2)*0.9,-0.07], 'convex', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=72, fontsize=13, color='DarkRed')
            plt.text(*[(np.pi/4)*1.2,0.12], 'convex', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=60, fontsize=15, color='DarkBlue')
            plt.text(*[(np.pi/6)*0.5,0.4], 'concave', #transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 rotation=50, fontsize=15, color='b')
            plt.polar(alpha_fn(np.arcsin(self.v_supc_array[:,0]/norm(self.v_supc_array, axis=1))),
                      v_scale_fn(norm(self.v_supc_array, axis=1)), 'DarkRed')

        plt.polar(
            alpha_fn((np.arcsin(self.v_supc_array[-1,0]/norm(self.v_supc_array[-1]))
                      +np.arcsin(self.v_infc_array[0,0]/norm(self.v_infc_array[0])))/2),
            (v_scale_fn(norm(self.v_infc_array[0]))+v_scale_fn(norm(self.v_supc_array[-1])))/2, 'o',
            color='DarkRed' if eta_>1 else 'r')

        xtick_posns = [np.pi*theta_ for theta_ in theta_list]
        plt.xticks(xtick_posns, xtick_labels, ha='left', fontsize=15);

        plt.text(*eta_xy_label, rf'$\eta={gmeq.eta}$', transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=18, color='k')
        plt.legend(loc=legend_xy)

        axes.tick_params(axis='x', pad=0, left=True, length=5, width=1, direction='out')

        axes.set_aspect(1)
        axes.set_rmax(scale_fn(r_max_))
        axes.set_rmin(scale_fn(r_min_))
        axes.set_thetamin(0)
        plt.grid(False, ls=':');


class SlicingPlots(GraphingBase):

    def __init__(self, gmeq, grid_res=301, dpi=100, font_size=11):
        """
        Constructor method.

        Args:
            dpi (int): resolution for rasterized images
            font_size (int): general font size
        """
        # Default construction
        super().__init__(dpi, font_size)
        self.H_Ci_eqn = gmeq.H_Ci_eqn
        self.Ci_H0p5_eqn = gmeq.degCi_H0p5_eqn
        Lc_varphi0_xih0_Ci_eqn = Eq(Lc, solve(gmeq.xih0_Lc_varphi0_Ci_eqn.subs({}), Lc)[0])
        self.gstarhat_eqn = Eq(gstarhat, (simplify(
                                gmeq.gstar_varphi_pxpz_eqn.rhs
                                    .subs(e2d(gmeq.varphi_rx_eqn))
                                    .subs(e2d(gmeq.px_pxhat_eqn))
                                    .subs(e2d(gmeq.pz_pzhat_eqn))
                                    .subs(e2d(gmeq.rx_rxhat_eqn))
                                    .subs(e2d(gmeq.varepsilon_varepsilonhat_eqn))
                                    .subs(e2d(Lc_varphi0_xih0_Ci_eqn))
                            ))/xih_0**2 )

        # Mesh grids for px-pz and rx-px space slicing plots
        self.grid_array =  np.linspace(0,1, grid_res)
        self.grid_array[self.grid_array==0.0] = 1e-6
        self.pxpzhat_grids = np.meshgrid(self.grid_array, -self.grid_array, sparse=False, indexing='ij')
        self.rxpxhat_grids = np.meshgrid(self.grid_array,  self.grid_array, sparse=False, indexing='ij')

    def prep_contour_fig(self, title, xlabel, ylabel):
        self.fig = self.create_figure(title, fig_size=(6,6))
        self.axes = plt.gca()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def define_lambdas(self, sub_, var_list, do_modv=True):
        self.define_H_lambda(sub_=sub_, var_list=var_list)
        self.define_Ci_lambda(sub_=sub_, var_list=var_list)
        self.define_Hessian_eigenvals(sub_=sub_, var_list=var_list)
        if do_modv:
            self.define_gstarhat_lambda(sub_=sub_, var_list=var_list)
            self.define_v_pxpzhat_lambda(sub_=sub_)
            self.define_modv_pxpzhat_lambda(sub_=sub_)
        self.pxhat_lambda = lambda sub_, rxhat_, pzhat_: solve(
            simplify(self.H_Ci_eqn.subs({rxhat:rxhat_, H:Rational(1,2), mu:eta/2})
                .subs(sub_).n().subs({pzhat:pzhat_})).subs({Abs(pxhat**1.0):pxhat}), pxhat)[0]
        self.pzhat_lambda = lambda sub_, rxhat_, pxhat_: (solve(
            simplify(self.H_Ci_eqn.subs({rxhat:rxhat_, H:Rational(1,2), mu:eta/2})
                .subs(sub_).subs({pxhat:pxhat_})), pzhat)[0])

    def pxhatsqrd_Ci_polylike_eqn(self, sub_, pzhat_, ):
        tmp = (self.H_Ci_eqn.rhs.subs({pzhat:pzhat_, H:Rational(1,2), mu:eta/2})
               .subs(omitdict(sub_,[Ci])))**2
        return Eq(4*numer(tmp)-denom(tmp),0)

    def pxhat_Ci_soln(self, eqn_, sub_, rxhat_, tolerance=1e-3):
        solns_ = Matrix(solve(eqn_.subs(sub_),pxhat**2)).subs({rxhat:rxhat_})
        return float(sqrt(
            [re(soln_) for soln_ in solns_.n() if Abs(im(soln_))<tolerance and re(soln_)>0][0]
        ))

    def pxpzhat0_values(self, contour_values_, sub_):
        pxpzhat_values_ = [None]*len(contour_values_)
        for i_,Ci_ in enumerate(contour_values_):
            tmp_sub_ = omitdict(sub_,[rxhat,Ci])
            tmp_sub_[Ci] = rad(Ci_)
            # pzhat for pxhat=1 which is true when rxhat=0
            pzhat_ = self.pzhat_lambda(tmp_sub_,0,1).n()
            eqn_ = self.pxhatsqrd_Ci_polylike_eqn(tmp_sub_, pzhat_)
            x_ = float(self.pxhat_Ci_soln(eqn_, tmp_sub_, rxhat.subs(sub_)))
            y_ = float(self.pzhat_lambda(tmp_sub_,0,1).n())
            pxpzhat_values_[i_] = (x_,y_)
        return pxpzhat_values_

    def define_H_lambda(self, sub_, var_list):
        self.H_lambda =  lambdify(var_list, self.H_Ci_eqn.rhs.subs({mu:eta/2}).subs(sub_), 'numpy')

    def define_Ci_lambda(self, sub_, var_list):
        self.Ci_lambda = lambdify(var_list, self.Ci_H0p5_eqn.rhs.subs({H:Rational(1,2), mu:eta/2}).subs(sub_), 'numpy')

    def define_gstarhat_lambda(self, sub_, var_list):
        self.gstarhat_lambda =  lambdify( var_list, self.gstarhat_eqn.rhs.subs({mu:eta/2}).subs(sub_), modules='numpy' )

    def define_v_pxpzhat_lambda(self, sub_):
        self.v_pxpzhat_lambda = lambdify( (pxhat,pzhat),
            simplify(((self.gstarhat_eqn.rhs.subs({mu:eta/2}).subs(sub_))*Matrix([pxhat,pzhat]))), modules='numpy' )

    def define_modv_pxpzhat_lambda(self, sub_):
        self.modv_pxpzhat_lambda = lambdify( (pxhat,pzhat),
            simplify(((self.gstarhat_eqn.rhs.subs({mu:eta/2}).subs(sub_))*Matrix([pxhat,pzhat])).norm()), modules='numpy' )

    def plot_modv_pzhat_slice(self, sub_, psub_):
        fig_name = ('v_pz_H0p5'
                    + f'_eta{float(eta.subs(sub_).n()):g}'
                    + f'_Ci{deg(Ci.subs(sub_))}'
                    + f'_rxhat{float(rxhat.subs(psub_).n()):g}'
                   ).replace('.','p')
        fig = self.create_figure(fig_name, fig_size=(6,5))
        axes = plt.gca()

        pxhat_eqn_ = self.pxhatsqrd_Ci_polylike_eqn(sub_, pzhat) #.subs(sub_)
        pxhat_poly_ = poly(pxhat_eqn_.lhs.subs(psub_).n(),pxhat)

        pzhat0_ = float(self.pzhat_lambda(sub_,0,1).n())
        pxhat0_ = float(px_value(rxhat.subs(psub_), pzhat0_, pxhat_poly_,
                                 px_var_=pxhat, pz_var_=pzhat))
        modv0_ = self.modv_pxpzhat_lambda(pxhat0_,pzhat0_)

        # For H=1/2
        pzhat_array = np.flipud(np.linspace(-30,0,31, endpoint=True if rxhat.subs(psub_)<0.95 and eta.subs(sub_)<1 else False))
        pxhat_array = np.array([float(px_value(rxhat.subs(psub_),pzhat_, pxhat_poly_, px_var_=pxhat, pz_var_=pzhat))
                       for pzhat_ in pzhat_array])

        H_array = np.array([float(self.H_lambda(pxhat_,pzhat_)) for pxhat_, pzhat_ in zip(pxhat_array, pzhat_array)])

        modp_array = np.sqrt(pxhat_array**2+pzhat_array**2)
        modv_array = np.array([self.modv_pxpzhat_lambda(pxhat_,pzhat_) for pxhat_,pzhat_ in zip(pxhat_array,pzhat_array)])
        projv_array = modv_array*np.cos(np.arctan(-pxhat_array/pzhat_array))

        plt.xlim(min(pzhat_array),max(pzhat_array))
        plt.ylim(0,max(modv_array)*1.05)
        axes.set_autoscale_on(False)

        plt.plot(pzhat_array, modv_array, 'o-', color='k', ms=3, label=r'$|\mathbf{\hat{v}}|$')
        plt.plot(pzhat_array, projv_array, '-', color='r', ms=3, label=r'$|\mathbf{\hat{v}}|\cos\beta$')
        plt.plot(pzhat_array,  (1/modp_array), '-', color='b', ms=3, label=r'$1/|\mathbf{\hat{p}}|$')
        plt.plot([pzhat0_,pzhat0_], [0,max(modv_array)*1.05], ':', color='k', label=r'$\hat{p}_z = \hat{p}_{z_0}$')

        plt.legend(loc='lower left')
        plt.grid('on')
        plt.ylabel(r'$|\mathbf{\hat{v}}|$  or  $|\mathbf{\hat{v}}|\cos\beta$  or  $1/|\mathbf{\hat{p}}|$')
        plt.xlabel(r'$\hat{p}_z\left(H=\frac{1}{2}\right)$')

        # Annotations
        x_ = 1.15
        for i_,label_ in enumerate([r'$\mathcal{H}=\frac{1}{2}$',
                                    r'$\mathbf{\hat{p}}(\mathbf{\hat{v}})=1$',
                                    rf'$\eta={eta.subs(sub_)}$',
                                    r'$\mathsf{Ci}=$'+rf'${deg(Ci.subs(sub_))}\degree$',
                                    r'$\hat{r}^x=$'+rf'${round(rxhat.subs(psub_),2)}$' ]):
            plt.text( *(x_,0.9-i_*0.15), label_, fontsize=16, color='k',
                     horizontalalignment='center', verticalalignment='center', transform=axes.transAxes )

        return fig_name

    def define_Hessian_eigenvals(self, sub_, var_list):
        H_Ci_ = self.H_Ci_eqn.rhs
        dHdpxhat_ = simplify( diff(H_Ci_,pxhat) )
        dHdpzhat_ = simplify( diff(H_Ci_,pzhat) )
        d2Hdpxhat2_ = simplify( diff(dHdpxhat_,pxhat) )
        d2Hdpxhatdpzhat_ = simplify( diff(dHdpxhat_,pzhat) )
        d2Hdpzhatdpxhat_ = simplify( diff(dHdpzhat_,pxhat) )
        d2Hdpzhat2_ = simplify( diff(dHdpzhat_,pzhat) )
        gstar_hessian = (
            Matrix([[d2Hdpxhat2_, d2Hdpxhatdpzhat_],[d2Hdpzhatdpxhat_, d2Hdpzhat2_]])
                                        .subs({mu:eta*2})
                                        .subs(sub_)
                                        .n()
        )
        gstar_hessian_lambda = lambdify( var_list, gstar_hessian )
        self.gstar_signature_lambda = lambda x_,y_: np.int(np.sum(np.sign(np.linalg.eigh(
            np.array(gstar_hessian_lambda(x_,y_),dtype=np.float) )[0])))//2

    def H_rxpx_contours(self, sub_, psf, do_Ci, **kwargs):
        return self.plot_Hetc_contours((self.rxpxhat_grids[0],self.rxpxhat_grids[1]*psf), sub_, do_Ci=do_Ci,
                                        do_fmt_labels=True if psf>1000 else False, do_aspect=False, do_rxpx=True, do_grid=True,
                                        **kwargs)

    def H_pxpz_contours(self, sub_, psf, do_Ci, **kwargs):
        return self.plot_Hetc_contours((self.pxpzhat_grids[0]*psf, self.pxpzhat_grids[1]*psf), sub_, do_Ci=do_Ci,
                                        do_fmt_labels=True if psf>1000 else False, do_aspect=True, do_rxpx=False, do_grid=False,
                                        **kwargs )

    def plot_Hetc_contours(self, grids_, sub_, do_Ci, do_modv=False,
                            do_fmt_labels=False, do_aspect=True, do_rxpx=False,
                            pxpz_points=None, rxpx_points=None,
                            do_log2H=False, do_siggrid=True, do_black_contours=False, do_grid=True,
                            contour_nlevels=None, contour_range=None, v_contour_range=None,
                            contour_values=None, contour_label_locs=None,
                            cmap_expt=0.5):
        # Create figure
        title = ( ('Ci' if do_Ci else ('v' if do_modv else 'H')) + ('_pslice' if do_rxpx else '_pslice')
                    + f'_eta{float(eta.subs(sub_).n()):g}'
                    + ( f'_Ci{deg(Ci.subs(sub_))}' if do_rxpx else f'_rxhat{float(rxhat.subs(sub_).n()):g}')
                ).replace('.','p')
        fig = self.create_figure(title, fig_size=(6,6))
        axes = plt.gca()
        labels = ('\hat{r}^x', '\hat{p}_x') if do_rxpx else ('\hat{p}_x', '\hat{p}_z')
        xlabel, ylabel = [rf'${label_}$' for label_ in labels]
        vars_label = r'$\left({},{}\right)$'.format(*labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if do_grid: plt.grid(':')

        # Generate a grid of H or Ci for a meshgrid of (px,pz) or (rx,px)
        H_grid_ = self.Ci_lambda(*grids_) if do_Ci else self.H_lambda(*grids_)
        modv_grid_ = self.modv_pxpzhat_lambda(*grids_) if (do_modv and not do_rxpx) else None
        # H_grid_[np.isnan(H_grid_)] = 0
        if do_siggrid:
            gstar_signature_grid_ = np.array(grids_[0].shape) # for i_ in [1,2]]
            gstar_signature_grid_ = np.array([self.gstar_signature_lambda(x_,y_)
                     for x_,y_ in zip(grids_[0].flatten(),grids_[1].flatten())])\
                     .reshape((grids_[0]).shape)
            gstar_signature_grid_[np.isnan(gstar_signature_grid_)] = 0

        # Axis labeling
        if do_fmt_labels:
            [axis_.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
                for axis_ in [axes.xaxis, axes.yaxis]]

        # Metric signature colour background
        if do_siggrid:
            cmap_name = 'PiYG'  #, 'plasma_r'
            cmap_ = plt.get_cmap(cmap_name)
            cf = axes.contourf(*grids_, gstar_signature_grid_, levels=1, cmap=cmap_)
    #         axes.pcolormesh(*grids_, np.power(H_grid_,cmap_expt), cmap=cmap_)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('top', size='6%', pad=0.4)
            label_levels = np.array([0.25,0.75])
            labels = (['mixed: -,+','positive: +,+'])
            cbar = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=label_levels,
                               label='metric signature')
    #         cbar.ax.set_label('$g_{\star}$ metric signature')
            cbar.ax.set_xticklabels(labels)
            cbar.ax.xaxis.set_ticks_position('top')
            if do_aspect: axes.set_aspect(1)
            fig.tight_layout()

        y_limit = (axes.get_ylim())

        # beta_crit line
        axes.set_autoscale_on(False)
        tan_beta_crit_ = np.sqrt(float(eta.subs(sub_)))
        beta_crit_ = np.round(np.rad2deg(np.arctan(tan_beta_crit_)),1)
        if do_rxpx:
            x_array = grids_[1][0]                                             # rx (+ve)
            y_array = grids_[1][0]*0 - float(pzhat.subs(sub_))*tan_beta_crit_  # px (+ve)
        else:
            x_array = -grids_[1][0]*tan_beta_crit_  # px (+ve)
            y_array =  grids_[1][0]                 # pz (-ve)
        axes.plot(x_array, y_array, 'Red', lw=3, ls='-', label=r'$\beta_\mathrm{c} = $'+rf'{beta_crit_}$\degree$')

        # px,pz on-shell point
        if pxpz_points is not None and not do_rxpx:
            for i_,(px_,pz_) in enumerate(pxpz_points):
                axes.scatter(px_, pz_, marker='o', s=70, color='k', label=None)
                beta_ = np.round(np.rad2deg(np.arctan(float(-px_/pz_))),1)
                beta_label = r'$\beta_0$' if rxhat.subs(sub_)==0 else r'$\beta$'
                axes.plot(np.array([0,px_*10]),np.array([0,pz_*10]), '-.', color='b',
                          label=beta_label+r'$ = $'+rf'{beta_:g}$\degree$' if i_==0 and not do_Ci else None)

        # pz=pz_0 constant line
        if pxpz_points is not None:
            for i_,(px_,pz_) in enumerate(pxpz_points):
                axes.plot(np.array([0,px_*100]),np.array([pz_,pz_]), ':', lw=2, color='grey',
                          label=r'$\hat{p}_{z} = \hat{p}_{z_0}$' if i_==0 else None)
                beta_ = np.round(np.rad2deg(np.arctan(float(-px_/pz_))),0)

        # Contouring
        cmap_ = plt.get_cmap('Greys_r')
        colors_ = ['k']

        # print(H_grid_.shape, modv_grid_.shape)
        if do_modv:
            fmt_modv = lambda modv_: f'{modv_:g}'
            # levels_ = np.linspace(0,0.5, 51, endpoint=True)
            levels_ = np.linspace(*v_contour_range, contour_nlevels, endpoint=True)
            modv_contours_ = axes.contour(*grids_, modv_grid_, levels_, cmap=cmap_)
                                     # levels_[levels_!=levels_H0p5[0]], linestyles=['solid'],
                                     # cmap=cmap_ if not do_black_contours else None,
                                     # colors=colors_ if do_black_contours else None)
            axes.clabel(modv_contours_, fmt=fmt_modv, inline=True, colors='0.3', fontsize=9)


        if contour_values is None:
            # Contour levels, label formats
            if do_log2H:
                n_levels_ = int(contour_range[1]-contour_range[0]+1)
                n_levels_ = n_levels_*2-1 if do_rxpx else n_levels_
                levels_ = np.linspace(*contour_range, n_levels_, endpoint=True)
                levels_H0p5 = [0]
                fmt_H = lambda H: r'$H={%s}$' % f'{np.round(10**H/2,2 if 10**H<0.5 else (1 if 10**H<5 else 0)):g}' if do_rxpx else r'$2H=10^{%s}$' % f'{H:g}'
                fmt_H0p5 = lambda H: rf'H=0.5'
                manual_location = (0.1,-7)
            else:
                levels_ = np.concatenate([
                    np.linspace(0.0,0.5, contour_nlevels[0], endpoint=False),
                    np.flip(np.linspace(contour_range[1],0.5, contour_nlevels[1], endpoint=False))
                ])
                levels_H0p5 = [0.5]
                fmt_H = lambda H_: rf'{H_:g}'
                fmt_H0p5 = lambda H_: rf'H={H_:g}'
                manual_location = ((np.array([0.6,-0.25]))*np.abs(y_limit[0]))

            # H contours
            contours_ = axes.contour(*grids_, np.log10(2*H_grid_) if do_log2H else H_grid_,
                                     levels_[levels_!=levels_H0p5[0]], linestyles=['solid'],
                                     cmap=cmap_ if not do_black_contours else None,
                                     colors=colors_ if do_black_contours else None)
            axes.clabel(contours_, inline=True, fmt=fmt_H, fontsize=9)
            contour_ = axes.contour(*grids_, np.log10(2*H_grid_) if do_log2H else H_grid_,
                                    levels_H0p5, linewidths=[3],
                                    cmap=cmap_ if not do_black_contours else None,
                                    colors=colors_ if do_black_contours else None)
            axes.clabel(contour_, inline=True, fmt=fmt_H0p5, fontsize=14,
                        manual=[(manual_location[0],manual_location[1])]) if manual_location is not None else None
        else:
            fmt_Ci = lambda Ci_: r'$\mathsf{Ci}=$'+f'{Ci_:g}'+r'$\degree$'
            contour_values_ = np.log10(2*np.array(contour_values)) if do_log2H else np.array(contour_values)
            contours_ = axes.contour(*grids_, np.log10(2*H_grid_) if do_log2H else H_grid_,
                                     contour_values_,
                                     cmap=cmap_ if not do_black_contours else None,
                                     colors=colors_ if do_black_contours else None
                                     )
            axes.clabel(contours_, inline=True, fmt=fmt_Ci, fontsize=12, manual=contour_label_locs)

        axes.set_autoscale_on(False)

        # H() or Ci() or v() annotation
        x_ = 1.25
        if not do_Ci:
            axes.text(*[x_,1.1],
                 (r'$|\mathbf{\hat{v}}|$' if do_modv else r'$\mathcal{H}$')+vars_label,
                 horizontalalignment='center', verticalalignment='center', transform=axes.transAxes,
                 fontsize=18, color='k')
            Ci_ = Ci.subs(sub_)
            axes.text(*[x_,0.91], r'$\mathsf{Ci}=$'+rf'${deg(Ci_)}\degree$',
                 horizontalalignment='center', verticalalignment='center', transform=axes.transAxes,
                 fontsize=16, color='k')
        else:
            axes.text(*[x_,1.], r'$\mathsf{Ci}$'+vars_label,
                 horizontalalignment='center', verticalalignment='center', transform=axes.transAxes,
                 fontsize=18, color='k')

        # eta annotation
        eta_ = eta.subs(sub_)
        axes.text(*[x_,0.8], rf'$\eta={eta_}$',
                 horizontalalignment='center', verticalalignment='center', transform=axes.transAxes,
                 fontsize=16, color='k')

        # pz or rx annotation
        if do_rxpx:
            label_ = r'$\hat{p}_{z_0}=$'
            val_ = round(pzhat.subs(sub_),1)
        else:
            label_ = r'$\hat{r}^x=$'
            val_ = round(rxhat.subs(sub_),2)
        axes.text(*[x_,0.68], label_+rf'${val_}$', transform=axes.transAxes,
             horizontalalignment='center', verticalalignment='center',
             fontsize=16, color='k')

        axes.legend(loc=[1.07,0.29], fontsize=15, framealpha=0)
        return title


class ManuscriptPlots(Graphing):

    def point_pairing(self, name, fig_size=(10,4), dpi=None):
        """
        Schematic illustrating ...

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
        """
        # Build fig
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        brown_  = '#994400'
        blue_   = '#0000dd'
        red_    = '#dd0000'
        purple_ = '#cc00cc'
        gray_ = self.gray_color(2,5)
        n_gray = 6
        gray1_ = self.gray_color(1,n_gray)
        gray2_ = self.gray_color(2,n_gray)

        def remove_ticks_etc(axes_):
        #     return
            axes_.set_xticklabels([])
            axes_.set_xticks([])
            axes_.set_yticklabels([])
            axes_.set_yticks([])
            axes_.set_xlim(0,1)
            axes_.set_ylim(0,1)

        def linking_lines(fig_, axes_A, axes_B, axes_C, axes_D, color_=brown_):
            joins = [0]*3
            kwargs = dict(color=color_, linestyle=':')
            joins[0] = ConnectionPatch(xyA=(0.2,0), coordsA=axes_D.transData,
                                       xyB=(0.4,1),  coordsB=axes_A.transData, **kwargs)
            joins[1] = ConnectionPatch(xyA=(1,0.00), coordsA=axes_D.transData,
                                       xyB=(0,0.9),  coordsB=axes_B.transData, **kwargs)
            joins[2] = ConnectionPatch(xyA=(1,0.60), coordsA=axes_D.transData,
                                       xyB=(0,0.8),  coordsB=axes_C.transData, **kwargs)
            [fig_.add_artist(join_) for join_ in joins]

        def make_xy():
            x = np.linspace(0,1)
            x_ndim = (x-0.5)/(0.9-0.5)
            y = np.exp((0.5+x)*4)/120
            return x_ndim,y

        def isochrones_subfig(fig_, x_, y_, color_=gray_):
            # Top left isochrones 0
            size_zoom_0 = [0.65, 0.55]
            posn_0 = [0.0, 0.75]
            axes_0 = fig_.add_axes([*posn_0, *size_zoom_0])
            plt.axis('off')
            n_isochrones = 6
            [plt.plot(x_, sf_*y_, '-', color=self.gray_color(i_,n_gray), lw=2.5)
                 for i_, sf_ in enumerate(np.linspace(0.5,1.2,n_isochrones))]
            plt.xlim(0,1)
            plt.ylim(0,1)
            sf1_ = 1.3
            sf2_ = 1.3
            arrow_xy_ = np.array([0.2,0.8])
            arrow_dxy_ = np.array([-0.025,0.15])
            motion_xy_ = [0.1,0.8]
            motion_angle_ = 23
            my_arrow_style = ArrowStyle.Simple(head_length=1*sf1_, head_width=1*sf1_,
                                               tail_width=0.1*sf1_)
            axes_0.annotate('', xy=arrow_xy_, xytext=arrow_xy_+arrow_dxy_*sf2_,
                            transform=axes_0.transAxes,
                            arrowprops=dict(arrowstyle=my_arrow_style, color=color_, lw=3))
            plt.text(*motion_xy_, 'motion', color=color_, fontsize=16, rotation=motion_angle_,
                     transform=axes_0.transAxes,
                     horizontalalignment='center', verticalalignment='center')
            return axes_0, posn_0

        def set_colors(obj_type, axes_list, color_):
            [[child.set_color(color_) for child in axes_.get_children()
             if isinstance(child, obj_type)] for axes_ in axes_list]

        def zoom_boxes(fig_, ta_color_=gray2_, tb_color_=gray1_):
            size_zoom_AB = [0.3,0.7]
            size_zoom_C = [0.3,0.7]
            n_pts = 300

            def zoomed_isochrones(axes_, name_text, i_pt1_, i_pt2_, do_many=False, do_legend=False, do_pts_only=False):
                x_array = np.linspace(-1,3, n_pts)
                y_array1 = x_array*0.5
                y_array2 = y_array1+0.7

                # Ta
                if not do_pts_only:
                    plt.plot(x_array, y_array2, '-', color=ta_color_, lw=2.5, label=r'$T_a$')
                xy_pt2 = np.array([x_array[i_pt2_],y_array2[i_pt2_]])
                marker_style2 = dict(marker='o', fillstyle='none', markersize=8,
                                     markeredgecolor=ta_color_, markerfacecolor=ta_color_,
                                     markeredgewidth=2 )
                plt.plot(*xy_pt2, **marker_style2)
                if not do_pts_only:
                    plt.text(*(xy_pt2+np.array([-0.03,0.08])), r'$\mathbf{a}$',
                             color=ta_color_, fontsize=18, rotation=0, transform=axes_.transAxes,
                             horizontalalignment='center', verticalalignment='center')
                # Tb
                if not do_pts_only:
                    plt.plot(x_array, y_array1, '-', color=tb_color_, lw=2.5, label=r'$T_b = T_a+\Delta{T}$')
                i_pts1 = [i_pt1_]
                xy_pts1_tmp = np.array([np.array([x_array[i_pt1__],y_array1[i_pt1__]])
                                    for i_pt1__ in i_pts1])
                xy_pts1 = xy_pts1_tmp.T.reshape((xy_pts1_tmp.shape[2],2)) if do_many else xy_pts1_tmp
                marker_style1 = marker_style2.copy()
                marker_style1.update({'markeredgecolor':tb_color_,'markerfacecolor':'w'})
                [plt.plot(*xy_pt1_, **marker_style1) for xy_pt1_ in xy_pts1]

                if not do_pts_only:
                    b_label_i = 4 if do_many else 0
                    b_label_xy = (xy_pts1[b_label_i]+np.array([0.03,-0.08]))
                    b_label_text = r'$\{\mathbf{b}\}$' if do_many else r'$\mathbf{b}$'
                    plt.text(*b_label_xy, b_label_text,
                             color=tb_color_, fontsize=18, rotation=0, transform=axes_.transAxes,
                             horizontalalignment='center', verticalalignment='center')
                    if do_legend:
                        plt.legend(loc=[0.05,-0.35], fontsize=16, framealpha=0)
                    name_xy = [0.97,0.03]
                    plt.text(*name_xy, name_text,
                             color=brown_, fontsize=14, rotation=0, transform=axes_.transAxes,
                             horizontalalignment='right', verticalalignment='bottom')

                dx = x_array[1]-x_array[0]
                dy = y_array1[1]-y_array1[0]
                return (xy_pts1 if do_many else xy_pts1[0]), xy_pt2, np.array([dx,dy])

            def v_arrow(axes_, xy_pt1, xy_pt2, dxy=[0.12,0.05], a_f=0.54, v_f=0.5, v_label=r'$\mathbf{v}$',
                        color_=red_, do_dashing=False, do_label=False):
                v_lw = 1.5
                axes_.arrow(*((xy_pt1*a_f+xy_pt2*(1-a_f))),*((xy_pt1-xy_pt2)*0.01),
                             lw=1, facecolor=color_, edgecolor=color_,
                             head_width=0.05, overhang=0.3,
                             transform=axes_.transAxes, length_includes_head=True, )
                axes_.plot([xy_pt1[0],xy_pt2[0]], [xy_pt1[1],xy_pt2[1]],
                            ls='--' if do_dashing else '-',
                            lw=v_lw, color=color_ )
                f = v_f
                v_xy = [xy_pt1[0]*f + xy_pt2[0]*(1-f)+dxy[0], xy_pt1[1]*f + xy_pt2[1]*(1-f)+dxy[1]]
                if do_label:
                    axes_.text(*v_xy, v_label,
                                color=color_, fontsize=18, rotation=0, transform=axes_.transAxes,
                                horizontalalignment='right', verticalalignment='bottom')

            def p_bones(axes_, xy_pt1, xy_pt2, dxy, p_f=0.9, color_=blue_, n_bones=5, do_primary=True):
                alpha_ = 0.7
                p_dashing = [1,0] #[4, 4]
                p_lw = 3
                axes_.plot([xy_pt1[0],xy_pt2[0]], [xy_pt1[1],xy_pt2[1]],
                           dashes=p_dashing, lw=p_lw, color=color_, alpha=1 if do_primary else alpha_ )
                for i_, f in enumerate(np.linspace(1,0,n_bones, endpoint=False)):
                    x = xy_pt1[0]*f + xy_pt2[0]*(1-f)
                    y = xy_pt1[1]*f + xy_pt2[1]*(1-f)
                    sf = 4 if do_primary else 3
                    dx = dxy[0]*sf
                    dy = dxy[1]*sf
                    x_pair = [x-dx, x+dx]
                    y_pair = [y-dy, y+dy]
                    axes_.plot(x_pair, y_pair, lw=5 if i_==0 else 2.5, color=color_,
                              alpha=1 if do_primary else alpha_)
                f = p_f
                p_xy = [xy_pt1[0]*f + xy_pt2[0]*(1-f)-0.1, xy_pt1[1]*f + xy_pt2[1]*(1-f)-0.1]
                if do_primary:
                    axes_.text(*p_xy, r'$\mathbf{\widetilde{p}}$',
                                color=color_, fontsize=18, rotation=0, transform=axes_.transAxes,
                                horizontalalignment='right', verticalalignment='bottom')

            def psi_label(axes_, xy_pt1_B, xy_pt2_B, xy_pt1_C, xy_pt2_C, color_=red_):
                label_xy = [0.5,0.53]
                axes_.text(*label_xy, r'$\psi$',
                            color=color_, fontsize=20, rotation=0, transform=axes_.transAxes,
                            horizontalalignment='center', verticalalignment='center')
                angle_B = np.rad2deg(np.arctan( (xy_pt2_B[1]-xy_pt1_B[1])/(xy_pt2_B[0]-xy_pt1_B[0]) ))
                angle_C = np.rad2deg(np.arctan( (xy_pt2_C[1]-xy_pt1_C[1])/(xy_pt2_C[0]-xy_pt1_C[0]) ))
                radius = 0.95
                axes_.add_patch( Arc(xy_pt2_B, radius,radius, color=color_,
                                    linewidth=2, fill=False, zorder=2, #transform=axes_.transAxes,
                                    angle=angle_B, theta1=0, theta2=angle_C-angle_B) )

            def beta_label(axes_, xy_pt1_B, xy_pt2_B, color_=blue_):
                label_xy = [0.28,0.47]
                axes_.text(*label_xy, r'$\beta$',
                            color=color_, fontsize=20, rotation=0, transform=axes_.transAxes,
                            horizontalalignment='center', verticalalignment='center')
                angle_ref = -90
                angle_B = np.rad2deg(np.arctan( (xy_pt2_B[1]-xy_pt1_B[1])/(xy_pt2_B[0]-xy_pt1_B[0]) ))
                radius = 0.88
                axes_.add_patch( Arc(xy_pt2_B, radius,radius, color=color_, linestyle='--',
                                    linewidth=2, fill=False, zorder=2, #transform=axes_.transAxes,
                                    angle=angle_ref, theta1=0, theta2=angle_B-angle_ref) )
                axes_.plot( [xy_pt2_B[0],xy_pt2_B[0]], [xy_pt2_B[1],xy_pt2_B[1]-0.5], ':', color=color_)

            def alpha_label(axes_, xy_pt1_B, xy_pt2_B, color_=red_):
                label_xy = [0.55,0.75]
                axes_.text(*label_xy, r'$-\alpha$',
                            color=color_, fontsize=20, rotation=0, transform=axes_.transAxes,
                            horizontalalignment='center', verticalalignment='center')
                angle_ref = 0
                angle_B = np.rad2deg(np.arctan( (xy_pt2_B[1]-xy_pt1_B[1])/(xy_pt2_B[0]-xy_pt1_B[0]) ))
                radius = 0.88
                axes_.add_patch( Arc(xy_pt2_B, radius,radius, color=color_, linestyle='--',
                                    linewidth=2, fill=False, zorder=2, #transform=axes_.transAxes,
                                    angle=angle_B, theta1=0, theta2=-angle_B) )
                axes_.plot( [xy_pt2_B[0],xy_pt2_B[0]+0.5], [xy_pt2_B[1],xy_pt2_B[1]], ':', color=color_)

            # From zoom box D
            posn_D = np.array(posn_0) + np.array([0.19,0.25])
            size_zoom_D = [0.042,0.1]
            axes_D = fig_.add_axes([*posn_D, *size_zoom_D])
            remove_ticks_etc(axes_D)
            axes_D.patch.set_alpha(0.5)

            # Zoom any point pairing A
            posn_A = [0,0.15]
            axes_A = fig_.add_axes([*posn_A, *size_zoom_AB])
            remove_ticks_etc(axes_A)
            i_pt2_A = 92
            i_pts1_A = [i_pt2_A+i_ for i_ in np.arange(-43,100,15)]
            xy_pts1_A, xy_pt2_A, dxy_A = zoomed_isochrones(axes_A, 'free', i_pts1_A,i_pt2_A,
                                                           do_many=True)
            [v_arrow(axes_A, xy_pt1_A, xy_pt2_A, do_dashing=True,
                     v_f=0.35, v_label=r'$\{\mathbf{v}\}$', do_label=(True if i_==len(xy_pts1_A)-1 else False))
                for i_,xy_pt1_A in enumerate(xy_pts1_A)]
            zoomed_isochrones(axes_A, '', i_pts1_A,i_pt2_A, do_many=True)

            # Zoom intrinsic pairing B
            posn_B = [0.33,0.28]
            axes_B = fig_.add_axes([*posn_B, *size_zoom_AB])
            remove_ticks_etc(axes_B)
            i_pt2_B = i_pt2_A
            i_pt1_B = i_pt2_A+19
            xy_pt1_B, xy_pt2_B, dxy_B = zoomed_isochrones(axes_B, 'isotropic', i_pt1_B,i_pt2_B)
            p_bones(axes_B, xy_pt1_B, xy_pt2_B, dxy_B)
            v_arrow(axes_B, xy_pt1_B, xy_pt2_B, v_f=0.5, dxy=[0.13,0.02], do_label=True)
            zoomed_isochrones(axes_B, '', i_pt1_B,i_pt2_B, do_pts_only=True)

            # Zoom erosion-fn pairing C
            posn_C = [0.66,0.6]
            axes_C = fig_.add_axes([*posn_C, *size_zoom_C])
            remove_ticks_etc(axes_C)
            i_pt1_C = i_pt1_B+30
            i_pt2_C = i_pt2_B
            xy_pt1_C, xy_pt2_C, dxy_C = zoomed_isochrones(axes_C, 'anisotropic', i_pt1_C,i_pt2_C,
                                                          do_legend=True)
            p_bones(axes_C, xy_pt1_C, xy_pt2_C, dxy_C, do_primary=False)
            p_bones(axes_C, xy_pt1_B, xy_pt2_B, dxy_B)
            v_arrow(axes_C, xy_pt1_C, xy_pt2_C, a_f=0.8, v_f=0.72, dxy=[0.1,0.05], do_label=True)
            zoomed_isochrones(axes_C, '', i_pt1_C,i_pt2_C,
                              do_legend=False, do_pts_only=True)
            psi_label(axes_C, xy_pt1_B, xy_pt2_B, xy_pt1_C, xy_pt2_C, color_=purple_)
            beta_label(axes_C, xy_pt1_B, xy_pt2_B)
            alpha_label(axes_C, xy_pt1_C, xy_pt2_C)

            # Brown zoom boxes and tie lines
            set_colors(Spine, [axes_A, axes_B, axes_C, axes_D], brown_)
            return axes_A, axes_B, axes_C, axes_D, brown_

        x,y = make_xy()
        axes_0, posn_0 = isochrones_subfig(fig, x, y)
        axes_A, axes_B, axes_C, axes_D, brown = zoom_boxes(fig)
        linking_lines(fig, axes_A, axes_B, axes_C, axes_D)

    def covector_isochrones(self, name, fig_size=None, dpi=None):
        """
        Schematic illustrating relationship between normal erosion rate vector, normal slowness covector,
        isochrones, covector components, and vertical/normal erosion rates.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        axes = plt.gca()
        axes.set_aspect(1)

        # Basics
        beta_deg = 60
        beta_ = np.deg2rad(beta_deg)
        origin_distance = 1
        origin_ = (origin_distance,origin_distance*np.tan(beta_))
        x_limit, y_limit = 3.2, 2.5
        p_color = 'b'
        u_color = 'r'

        # Vectors and covectors
        u_perp_ = 2
        p_ = 1/u_perp_
        px_,pz_ = (p_*np.sin(beta_),-p_*np.cos(beta_))
        nDt_ = 1
        L_u_perp_ = u_perp_*nDt_
        L_p_,L_px_,L_pz_ = p_*nDt_,px_*nDt_,pz_*nDt_
        L_ux_,L_uz_ = (L_u_perp_*np.sin(beta_),-L_u_perp_*np.cos(beta_))
        # L_px_,L_pz_ = (L_p_*np.sin(beta_),-L_p_*np.cos(beta_))
        # L_u_right_,L_u_up_ = (1/px_)*nDt_,(1/pz_)*nDt_
        # px_,pz_ = (p_*np.sin(beta_),-p_*np.cos(beta_))

        # Time scale and isochrones
        n_major_isochrones = 2
        ts_major_isochrones = 4
        n_minor_isochrones = (n_major_isochrones-1)*ts_major_isochrones+2
        sf_ = 0.5
        x_array = np.linspace(0,0.5,2)*5
        for i_isochrone in list(range(n_minor_isochrones)):
            color_ = self.gray_color(i_isochrone=i_isochrone, n_isochrones=n_minor_isochrones)
            plt.plot(x_array+sf_*i_isochrone/np.sin(beta_), x_array*np.tan(beta_), color=color_,
                     lw=2.5 if i_isochrone % ts_major_isochrones==0 else 0.75,
                     label=r'$T(\mathbf{r})$'+'$={}$y'.format(i_isochrone)
                            if i_isochrone % ts_major_isochrones==0 else None)

        # r vector
        af, naf = 0.7, 0.3
        r_length = 0.6 #1.5
        r_color = 'gray'
        plt.plot(origin_[0],origin_[1], 'o', color=r_color, ms=15)
        plt.text(origin_[0]-r_length*np.cos(0.5)*naf+0.04,origin_[1]-r_length*np.sin(0.5)*naf-0.04,
                 r'$\mathbf{r}$', horizontalalignment='center', verticalalignment='bottom',
                 fontsize=18, color=r_color)


        # Slowness covector p thick transparent lines
        lw_pxz_ = 10
        alpha_p_ = 0.1
        px_array = np.linspace(origin_[0],origin_[0]+L_px_*4,2)
        pz_array = np.linspace(origin_[1],origin_[1]+L_pz_*4,2)
        plt.plot(px_array,pz_array, lw=lw_pxz_, alpha=alpha_p_, color=p_color, solid_capstyle='butt')
        plt.plot(px_array,(pz_array[0],pz_array[0]), lw=lw_pxz_,
                    alpha=alpha_p_, color=p_color, solid_capstyle='butt')
        plt.plot((px_array[0],px_array[0]),pz_array, lw=lw_pxz_,
                    alpha=alpha_p_, color=p_color, solid_capstyle='butt')

        # Slowness covector p decorations
        hw=0.12
        hl=0.0
        oh=0.0
        lw=2.5
        # np_ = 1+int(0.5+np_scale*((p_-p_min)/p_range)) if p_>=p_min else 1
        for np_,(dx_,dz_) in zip([1,4,3],[[0,-L_uz_/1],[L_ux_/4,-L_uz_/4],[L_ux_/3,0]]):
            x_, z_ = origin_[0], origin_[1]
            for i_head in list(range(1,np_+1)):
                plt.arrow(x_, z_, dx_*i_head, -dz_*i_head,
                        head_width=hw, head_length=hl, lw=lw,
                        shape='full', overhang=oh,
                        length_includes_head=True,
                        ec=p_color)
                if i_head==np_:
                    for sf in [0.995,1/0.995]:
                        plt.arrow(x_, z_, dx_*i_head*sf, -dz_*i_head*sf,
                            head_width=hw, head_length=hl, lw=lw,
                            shape='full', overhang=oh,
                            length_includes_head=True,
                            ec=p_color)

        plt.arrow(px_array[0], pz_array[0], (px_array[0]-px_array[1])/22,(pz_array[0]-pz_array[1])/22,
                    head_width=0.16, head_length=-0.1, lw=2.5,
                    shape='full', overhang=1,
                    length_includes_head=True,
                    head_starts_at_zero=True,
                    ec='b', fc='b')

        # Coordinate axes
        x_off, y_off = 0,-0.4
        my_arrow_style = mpatches.ArrowStyle.Fancy(head_length=.99, head_width=.6, tail_width=0.01)
        axes.annotate('', xy=(x_limit/10-0.05+x_off,y_limit*0.7+y_off),
                        xytext=(x_limit/10-0.275+x_off,y_limit*0.7+y_off),
                        arrowprops=dict(arrowstyle=my_arrow_style, color='k') )
        axes.annotate('', xy=(x_limit/10-0.265+x_off,y_limit*0.7+0.2+y_off),
                        xytext=(x_limit/10-0.265+x_off,y_limit*0.7-0.01+y_off),
                        arrowprops=dict(arrowstyle=my_arrow_style, color='k') )
        plt.text(x_limit/10-0.05+x_off,y_limit*0.7+y_off,'$x$',
                    horizontalalignment='left', verticalalignment='center', fontsize=20, color='k')
        plt.text(x_limit/10-0.265+x_off,y_limit*0.7+0.2+y_off,'$z$',
                    horizontalalignment='center', verticalalignment='bottom', fontsize=18, color='k')

        # Surface isochrone text label
        si_posn = 0.35 #1.1
        plt.text(si_posn+0.13,si_posn*np.tan(beta_), r'surface isochrone $\,\,T(\mathbf{r})$',
                 rotation=beta_deg, horizontalalignment='center', verticalalignment='bottom',
                 fontsize=15,  color='Gray')

        # Angle arc and text
        arc_radius = 1.35
        axes.add_patch( mpatches.Arc((0, 0), arc_radius,arc_radius, color='Gray',
                                     linewidth=1.5, fill=False, zorder=2,
                                     theta1=0, theta2=60) )
        plt.text(0.18,0.18, r'$\beta = $'+'{}'.format(beta_deg)+r'${\degree}$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=20,  color='Gray')

        # from matplotlib import rcParams
        # rcParams['text.usetex'] = True
        # rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        # rcParams['text.latex.preamble'] = r'\usepackage[scaled]{helvet}'
        # rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'
        # rcParams['text.latex.preamble'] = r'\usepackage{eucal}'
        # rcParams['text.latex.preamble'] = r'\usepackage[mathcal]{eucal}'
        # rcParams['text.latex.preamble'] = r'\usepackage[mathscr]{eucal}'


        # Erosion arrow
        l_erosion_arrow = 0.2
        gray_ = self.gray_color(2,5)
        off_ = 0.38
        plt.arrow(origin_[0]+off_+0.02, origin_[1]+off_*np.tan(beta_),
                  l_erosion_arrow*np.tan(beta_),-l_erosion_arrow, head_width=0.08, head_length=0.1, lw=7,
                  length_includes_head=True, ec=gray_, fc=gray_, capstyle='butt', overhang=0.1)
        off_x, off_z = 0.27, 0
        plt.text(origin_[0]+off_+0.02+off_x,origin_[1]+off_*np.tan(beta_)+off_z,'erosion',
                 rotation=beta_deg-90, horizontalalignment='center', verticalalignment='center',
                 fontsize=15,  color=gray_)

        # Unit normal n vector
        off_x, off_z = 1.1,0.55
        plt.text(origin_[0]+off_x,origin_[1]+off_z,
                 r'$\mathbf{n} = $',
                 horizontalalignment='right', verticalalignment='center', fontsize=15,  color='k')
        plt.text(origin_[0]+off_x,origin_[1]+off_z,
                 r'$\left[ \,\stackrel{\sqrt{3}/{2}}{-1/2}\, \right]$', #  \binom{\sqrt{3}/{2}}{-1/2}
                 horizontalalignment='left', verticalalignment='center', fontsize=20,  color='k')


        # p text annotations
        plt.text(origin_[0]+1.7,origin_[1]+0.1, r'${{p}}_x(\mathbf{{n}}) = 3$',
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=18, color=p_color)
        plt.text(origin_[0]+0.1,origin_[1]-1, r'${{p}}_z(\mathbf{{n}}) = 1$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=18, color=p_color)
        plt.text(origin_[0]+1.77,origin_[1]-1.05, r'$\mathbf{\widetilde{p}}(\mathbf{{n}}) = p = 4$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=18, color=p_color)
        plt.text(origin_[0]+1,origin_[1]-0.4, r'$\mathbf{\widetilde{p}} = [2\sqrt{3} \,\,\, -\!2]$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=18, color=p_color)

        # u arrows
        lw_u = 3
        alpha_ = 0.5
        u_perp, u_vert, u_horiz = 0.25, 1, np.sqrt(3)/3
        af, naf = 0.7, 0.3
        plt.arrow(*origin_, u_perp*np.tan(beta_)*af,-u_perp*af,
                  head_width=0.08, head_length=0.12, length_includes_head=True,
                  lw=lw_u, alpha=alpha_, ec=u_color, fc=u_color, overhang=0.1)
        plt.arrow(origin_[0]+u_perp*np.tan(beta_)*af, origin_[1]-u_perp*af,
                  u_perp*np.tan(beta_)*naf,-u_perp*naf,
                  head_width=0.0, head_length=0, length_includes_head=True,
                  lw=lw_u, alpha=alpha_, ec=u_color, fc='w', overhang=0.1)
        af, naf = 0.6, 0.4
        plt.arrow(*origin_, 0,-u_vert*af,
                  head_width=0.08, head_length=0.12, length_includes_head=True,
                  lw=lw_u, alpha=alpha_, ec=u_color, fc='w', overhang=0.1)
        plt.arrow(origin_[0], origin_[1]-u_vert*af, 0,-u_vert*naf+0.015,
                  head_width=0.0, head_length=0, length_includes_head=True,
                  lw=lw_u, alpha=alpha_, ec=u_color, fc='w', overhang=0.1)
        af, naf = 0.7, 0.3
        plt.arrow(*origin_, u_horiz*af,0,
                  head_width=0.08, head_length=0.12, length_includes_head=True,
                  lw=lw_u, alpha=alpha_, ec=u_color, fc='w', overhang=0.1)
        plt.arrow(origin_[0]+u_horiz*af, origin_[1], u_horiz*naf-0.015,0,
                  head_width=0.0, head_length=0, length_includes_head=True,
                  lw=lw_u, alpha=alpha_, ec=u_color, fc='w', overhang=0.1)

        # u text annotations
        plt.text(origin_[0]+0.23,origin_[1]+0.09,
                 r'${\xi}^{\!\rightarrow} = \frac{1}{p \, \sin\beta} = \frac{1}{2\sqrt{3}}$',
                 horizontalalignment='left', verticalalignment='bottom',
                 fontsize=18, color=u_color)
        plt.text(origin_[0]+0.06,origin_[1]-0.61,
                 r'${\xi}^{\!\downarrow} = \frac{1}{p \, \cos\beta} = \frac{1}{2}$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=18, color=u_color)
        plt.text(origin_[0]+0.5,origin_[1]-0.18,
                 r'${\xi}^{\!\perp} = \frac{1}{4}$', #'$\xi^{\!\perp} = 1/4$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=18, color=u_color)


        # Grid, limits, etc
        plt.axis('off')
        plt.xlim(0,x_limit)
        plt.ylim(0,y_limit)
        plt.grid('on')
        plt.legend(loc='lower center', fontsize=13, framealpha=0.95)
        # plt.legend(loc='center left', fontsize=13, framealpha=0.95)


        # Length scale
        inset_axes_ = inset_axes(axes, width='{}%'.format(31.5), height=0.15, loc=2)
        plt.xticks(np.linspace(0,2,3),labels=[0,0.25,0.5])
        plt.yticks([])
        plt.xlabel(r'distance  [mm]')
        inset_axes_.spines['top'].set_visible(False)
        inset_axes_.spines['left'].set_visible(False)
        inset_axes_.spines['right'].set_visible(False)
        inset_axes_.spines['bottom'].set_visible(True)

    def huygens_wavelets( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                          do_ray_conjugacy=False, do_fast=False,
                          do_legend=True, legend_fontsize=10, annotation_fontsize=11):
        r"""
        Plot the loci of :math:`\mathbf{\widetilde{p}}` and :math:`\mathbf{r}` and
        their behavior defined by :math:`F` relative to the :math:`\xi` circle.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by :meth:`GMPLib create_figure <plot_utils.GraphingBase.create_figure>`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            do_ray_conjugacy (bool): optional generate ray conjugacy schematic?
        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        def trace_indicatrix(sub, n_points, xy_offset=[0,0], sf=1, pz_min=2.5e-2, pz_max=1000):
            rdotx_pz_eqn = gmeq.idtx_rdotx_pz_varphi_eqn.subs(sub)
            rdotz_pz_eqn = gmeq.idtx_rdotz_pz_varphi_eqn.subs(sub)
            rdotx_pz_lambda = lambdify( pz, (re(sy.N(rdotx_pz_eqn.rhs))), 'numpy')
            rdotz_pz_lambda = lambdify( pz, (re(sy.N(rdotz_pz_eqn.rhs))), 'numpy')
            fgtx_pz_array = -pz_min*np.power(2,np.linspace(np.log2(pz_max/pz_min),
                                                           np.log2(pz_min/pz_min),n_points, endpoint=True))
            # print(fgtx_pz_array)
            idtx_rdotx_array = np.array([ float(rdotx_pz_lambda(pz_)) for pz_ in fgtx_pz_array] )
            idtx_rdotz_array = np.array([ float(rdotz_pz_lambda(pz_)) for pz_ in fgtx_pz_array] )
            return idtx_rdotx_array*sf+xy_offset[0],idtx_rdotz_array*sf+xy_offset[1]

        isochrone_color, isochrone_width, isochrone_ms, isochrone_ls \
                = 'Black', 2, 8 if do_ray_conjugacy else 7, '-'
        new_isochrone_color, newpt_isochrone_color, new_isochrone_width, \
            new_isochrone_ms, new_isochrone_ls \
                = 'Gray', 'White', 2 if do_ray_conjugacy else 4, \
                  8 if do_ray_conjugacy else 7, '-'
        wavelet_color = 'DarkRed'
        wavelet_width = 2.5 if do_ray_conjugacy else 1.5
        p_color, p_width = 'Blue', 2
        r_color, r_width = '#15e01a', 1.5

        dt_ = 0.0015
        dz_ = -gmes.xiv_v_array[0]*dt_*1.15
        # Fudge factors are NOT to "correct" curves but rather to account
        #   for the exaggerated Delta{t} that makes the approximations here just wrong
        #   enough to make the r, p etc annotations a bit wonky
        dx_fudge, dp_fudge = 0.005,1.15

        # Old isochrones
        plt.plot( gmes.h_x_array[0],gmes.h_z_array[0], 'o', mec='k',
                  mfc=isochrone_color, ms=isochrone_ms, fillstyle='full', markeredgewidth=0.5,
                  label='point $\mathbf{r}$')
        plt.plot( gmes.h_x_array, gmes.h_z_array, lw=isochrone_width, c=isochrone_color, ls=isochrone_ls,
                  label=r'isochrone  $T(\mathbf{r})=t$' )

        # Adjust plot scale, limits
        if do_ray_conjugacy:
            x_limits = [0.35,0.62]; y_limits = [0.018,0.12]
        else:
            # plt.xlim([0.5,0.8]); plt.ylim(0.045,0.2)
            x_limits = [0.45,0.75]; y_limits = [0.03,0.19]
        plt.xlim(*x_limits); plt.ylim(*y_limits)
        axes.set_aspect(1)

        # New isochrones
        plt.plot( gmes.h_x_array+dx_fudge, gmes.h_z_array + dz_,
                  c=new_isochrone_color, lw=new_isochrone_width, ls=new_isochrone_ls )

        # Erosion arrow
        i_ = 161 if do_ray_conjugacy else 180
        rx_, rz_ = (gmes.h_x_array[i_]+gmes.h_x_array[i_-1])/2, (gmes.h_z_array[i_]+gmes.h_z_array[i_-1])/2
        if do_ray_conjugacy:
            rx_, rz_ = ( (gmes.h_x_array[i_]+gmes.h_x_array[i_-1])/2,
                         (gmes.h_z_array[i_]+gmes.h_z_array[i_-1])/2 )
        else:
            rx_, rz_ = gmes.h_x_array[i_+1], gmes.h_z_array[i_+1]
            rx_ += dx_fudge
            rz_ += dz_
        beta_ = float(gmes.beta_p_interp(rx_))
        beta_deg = np.rad2deg(beta_)
        sf = 0.03 if do_ray_conjugacy else 0.06
        lw = 3 if do_ray_conjugacy else 5
        l_erosion_arrow = 0.4
        gray_ = self.gray_color(2,5)
        plt.arrow( rx_,rz_, l_erosion_arrow*np.tan(beta_)*sf,-l_erosion_arrow*sf,
                   head_width=0.15*sf, head_length=0.15*sf, lw=lw,
                   length_includes_head=True, ec=gray_, fc=gray_, capstyle='butt', overhang=0.1*sf )

        # Erosion label
        off_x, off_z = (-0.002, +0.015) if do_ray_conjugacy else (0.02, -0.005)
        rotation = beta_deg if do_ray_conjugacy else beta_deg-90
        plt.text( rx_+off_x,rz_+off_z,'erosion', rotation=rotation,
                  horizontalalignment='center', verticalalignment='top',
                  fontsize=annotation_fontsize,  color=gray_ )

        # Specify where to sample indicatrices
        i_start = 0; i_end = 220; n_i = 3 if do_fast else 15
        i_list = [111] if do_ray_conjugacy else \
                [int(i) for i in i_start+(i_end-i_start)*np.linspace(0,1,n_i)**0.7]

        # Construct indicatrix wavelets
        for idx,i_ in enumerate(i_list):
            print(f'{idx}: {i_}')
            i_from = i_list[0]
            rx_, rz_ = gmes.h_x_array[i_], gmes.h_z_array[i_]
            drx_, drz = gmes.rdotx_interp(rx_)*dt_, gmes.rdotz_interp(rx_)*dt_
            rxn_, rzn_ = rx_+drx_, rz_+drz
            recip_p_ = (1/gmes.p_interp(rx_))*dt_
            pxn_, pzn_ = rx_ + (1/gmes.px_interp(rx_))*dt_, rz_ + (1/gmes.pz_interp(rx_))*dt_
             #(1/gmes.pz_array[0])*dt_
            beta_ = float(gmes.beta_p_interp(rx_))
            dpx_, dpz_ = recip_p_*np.sin(beta_)*dp_fudge, -recip_p_*np.cos(beta_)*dp_fudge
            varphi_ = float(gmeq.varphi_rx_eqn.rhs.subs(sub).subs({rx:rx_}))
            n_points = 80 if do_ray_conjugacy else 5 if do_fast else 50
            pz_max = 1000 if do_ray_conjugacy else 1000
            idtx_rdotx_array,idtx_rdotz_array \
                = trace_indicatrix( {varphi:varphi_},  n_points=n_points,
                                    xy_offset=[rx_, rz_], sf=dt_, pz_min=1e-3, pz_max=pz_max )

            # Plot wavelets
            lw = 1.5
            plt.plot( idtx_rdotx_array,idtx_rdotz_array, lw=wavelet_width, ls='-',
                      c=wavelet_color,
                      label='erosional wavelet $\{\Delta\mathbf{r}\}$' if i_==i_from else None )
            if not do_ray_conjugacy:
                k_ = 0
                plt.plot( [rx_*k_+idtx_rdotx_array[0]*(1-k_),idtx_rdotx_array[0]],
                          [rz_*k_+idtx_rdotz_array[0]*(1-k_),idtx_rdotz_array[0]],
                          lw=lw, c=wavelet_color, alpha=0.7, ls='-' )
                plt.plot( [rx_,idtx_rdotx_array[0]],[rz_,idtx_rdotz_array[0]],
                          lw=lw, c=wavelet_color, alpha=0.4, ls='-' )
            else:
                plt.plot( [rx_,idtx_rdotx_array[0]],[rz_,idtx_rdotz_array[0]],
                          lw=lw, c=wavelet_color, alpha=1, ls='--' )

            # Ray arrows & new points
            sf = 1.7 if do_ray_conjugacy else 0.1
            plt.arrow(rx_, rz_, (rxn_-rx_), (rzn_-rz_),
                      head_width=0.007*sf,
                      head_length=0.009*sf,
                      overhang=0.18, width=0.0003,
                      length_includes_head=True,
                      alpha=1, ec='w', fc=r_color, linestyle='-') #, shape='full')
            plt.plot( [rx_, rxn_], [rz_, rzn_],
                      lw=2 if do_ray_conjugacy else r_width, alpha=1, c=r_color, linestyle='-', # shape='full',
                      label=r'ray increment  $\Delta{\mathbf{r}}$' if i_==i_from else None)
            plt.plot( rxn_, rzn_, 'o', mec=new_isochrone_color, mfc=newpt_isochrone_color,
                      ms=new_isochrone_ms, fillstyle=None, markeredgewidth=1.5)

            # Normal slownesses
            plt.plot([rx_,rx_+dpx_],[rz_,rz_+dpz_],'-', c=p_color, lw=3 if do_ray_conjugacy else r_width,
                     label=r'front increment  $\mathbf{\widetilde{p}}\Delta{t}\,/\,{p}^2$'
                     if i_==i_from else None)

            # axes.annotate('', xy=(rx_,rz_), xytext=(rx_-dpx_*1e-6,rz_-dpz_*1e-6),
            #               arrowprops={'headlength':0.4*sf, 'headwidth':0.2*sf, 'lw':1.5, 'ec':'b', 'fc':'w'},
            #               va='center')
            # my_arrow_style = mpatches.ArrowStyle.Fancy(head_length=1*sf, head_width=.8*sf, tail_width=0.1*sf)
            # axes.annotate('', xy=(rx_,rz_), xytext=(rx_-dpx_*1e-6,rz_-dpz_*1e-6),
            #               arrowprops=dict(arrowstyle=my_arrow_style, color='b', lw=1.5))
            if True or do_ray_conjugacy:
                sf = 1.5
                plt.arrow(rx_-dpx_*0.15, rz_-dpz_*0.15, -dpx_*0.1, -dpz_*0.1,
                            head_width=0.007*sf, head_length=-0.006*sf, lw=1*sf,
                            shape='full', overhang=1,
                            length_includes_head=True,
                            head_starts_at_zero=True,
                            ec='b', fc='b')
            # Old points
            plt.plot( rx_, rz_, 'o', mec='k', mfc=isochrone_color, ms=isochrone_ms,
                      fillstyle='full', markeredgewidth=0.5 )

        # New isochrones
        plt.plot( 0,0, c=new_isochrone_color, lw=new_isochrone_width, ls=new_isochrone_ls,
                  label=r'isochrone  $T(\mathbf{r}\!+\!\Delta{\mathbf{r}})=t+\Delta{t}$')
        plt.plot( rxn_, rzn_, 'o', mec=new_isochrone_color, mfc=newpt_isochrone_color,
                  ms=new_isochrone_ms, fillstyle=None, markeredgewidth=1.5,
                  label=r'point $\mathbf{r}+\Delta{\mathbf{r}}$')

        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=14)
        plt.ylabel(r'Elevation, $z/L_{\mathrm{c}}$  [-]', fontsize=14)
        if do_legend:
            plt.legend(loc='upper left', fontsize=legend_fontsize, framealpha=0.95)

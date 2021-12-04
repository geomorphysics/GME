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
import warnings

# Typing
# from typing import Any

# Numpy
import numpy as np

# SymPy
from sympy import deg

# GME
from gme.core.symbols import Ci
from gme.plot.base import Graphing

# MatPlotLib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

__all__ = ['TimeDependent']


class TimeDependent(Graphing):
    """
    Subclasses :class:`gme.plot.Graphing <plot.Graphing>`.
    """
    rp_list = ['rx','rz','px','pz']
    rpt_list = rp_list+['t']

    def profile_isochrones( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                            do_zero_isochrone=True, do_overlay=False, fig=None,
                            do_rays=True, ray_subsetting=5, ray_lw=0.5, ray_ls='-', ray_label='ray',
                            do_isochrones=True, isochrone_subsetting=1, #do_isochrone_p=False, BROKEN
                            isochrone_lw=0.5, isochrone_ls='-',
                            do_annotate_rays=False, n_arrows=10, arrow_sf=0.7, arrow_offset=4,
                            do_annotate_cusps=False, cusp_lw=1.5, do_smooth_colors=False,
                            x_limits=(-0.001,1.001), y_limits=(-0.025,0.525),
                            aspect=None,
                            do_legend=True, do_alt_legend=False, do_grid=True,
                            do_infer_initiation=True,
                            do_etaxi_label=True, eta_label_xy=(0.65,0.85),
                            do_pub_label=False, pub_label=None, pub_label_xy=(0.5,0.92) ) \
                                                                                    -> mpl.figure.Figure:
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
        # HACK
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi) if not do_overlay else None
        # pub_label_xy = [0.5,0.92] if pub_label_xy is None else pub_label_xy
        # eta_label_xy = [0.65,0.85] if eta_label_xy is None else eta_label_xy

        # Unpack for brevity
        if hasattr(gmes,'rpt_isochrones'):
            rx_isochrones, rz_isochrones, _, _, t_isochrones \
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
            for i_ray,(rx_array,rz_array,_) in enumerate(zip(reversed(gmes.rpt_arrays['rx']),
                                                                   reversed(gmes.rpt_arrays['rz']),
                                                                   reversed(gmes.rpt_arrays['t']))):
                if (i_ray//ray_subsetting-i_ray/ray_subsetting)==0:
                    this_ray_label=(ray_label+r' ($t_{\mathrm{oldest}}$)' if i_ray==0 else
                                    ray_label+r' ($t_{\mathrm{newest}}$)' if i_ray==n_rays-1 else
                                    None)
                    if do_annotate_rays:
                        self.arrow_annotate_ray_custom(rx_array, rz_array, axes, i_ray, ray_subsetting, n_rays,
                                                       n_arrows, arrow_sf, arrow_offset,
                                                       x_limits=x_limits, y_limits=y_limits,
                                                       line_style=ray_ls, line_width=ray_lw,
                                                       ray_label=this_ray_label,
                                                       do_smooth_colors=do_smooth_colors)
                    else:
                        plt.plot(rx_array,rz_array, lw=ray_lw,
                                 color=self.mycolors(i_ray, ray_subsetting, n_rays, do_smooth=do_smooth_colors),
                                                     linestyle=ray_ls, label=this_ray_label)

        # Time slices or isochrones of erosion front
        if hasattr(gmes,'rpt_isochrones') and do_isochrones:
            n_isochrones = len(rx_isochrones)
            delta_t = t_isochrones[1]
            i_isochrone, rx_isochrone, rz_isochrone = None, None, None # suppresses annoying pylint warning
            for i_isochrone,(rx_isochrone,rz_isochrone,_) in \
                                        enumerate(zip(rx_isochrones,rz_isochrones,t_isochrones)):
                i_subsetted = (i_isochrone//isochrone_subsetting-i_isochrone/isochrone_subsetting)
                i_subsubsetted = (i_isochrone//(isochrone_subsetting*10)-i_isochrone/(isochrone_subsetting*10))
                if (i_isochrone>0 and i_subsetted==0 and rx_isochrone is not None):
                    plt.plot(rx_isochrone, rz_isochrone, self.gray_color(i_isochrone, n_isochrones),
                             linestyle=isochrone_ls, lw=1.3*isochrone_lw if i_subsubsetted==0 else 0.5*isochrone_lw)
            # Hack legend items
            if rx_isochrone is not None:
                plt.plot(rx_isochrone, rz_isochrone, self.gray_color(i_isochrone, n_isochrones),
                         linestyle=isochrone_ls, lw=1.3*isochrone_lw,
                         label=r'isochrone $\Delta{\hat{t}}=$'+rf'${int(10*delta_t)}$')
                plt.plot(rx_isochrone, rz_isochrone, self.gray_color(i_isochrone, n_isochrones),
                         linestyle=isochrone_ls, lw=0.5*isochrone_lw,
                         label=r'isochrone $\Delta{\hat{t}}=$'+rf'${round(delta_t,1)}$')
                # HACK - broken for reasons unknown
                # if do_isochrone_p:
                #     for (rx_lowres,rz_lowres,px_lowres,pz_lowres) \
                #             in zip(rx_isochrones_lowres,rz_isochrones_lowres,
                #                    px_isochrones_lowres,pz_isochrones_lowres):
                #         _ = [plt.arrow(rx_,rz_,0.02*px_/np.sqrt(px_**2+pz_**2),0.02*pz_/np.sqrt(px_**2+pz_**2),
                #                     ec=color_, fc=color_, lw=0.5*isochrone_lw,
                #                     head_width=0.015, head_length=0, overhang=0)
                #                for (rx_,rz_,px_,pz_) in zip(rx_lowres, rz_lowres, px_lowres, pz_lowres)]

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
            plt.text(*eta_label_xy,
                     rf'$\eta={gmeq.eta_}$'+r'$\quad\mathsf{Ci}=$'+rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='k')
        if do_pub_label:
            plt.text(*pub_label_xy, pub_label, transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

        return fig


    def profile_cusp_speed( self, gmes, gmeq, name, fig_size=None, dpi=None,
                            # sample_spacing=10,
                            x_limits=(-0.05,1.05), t_limits=(0,None), y_limits=(-5,None),
                            legend_loc='lower right', do_x=True, do_infer_initiation=True ) -> None:
        r"""
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
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Drop last cusp because it's sometimes bogus
        _ =  gmes.trxz_cusps
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
        plt.text(0.15,0.2, rf'$\eta={gmeq.eta_}$', transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')

        x_array = np.linspace(0.001 if do_infer_initiation else x_or_t_array[0],1,num=101)
        # color_cx, color_bounds = 'DarkGreen', 'Green'
        color_cx, color_bounds = 'Red', 'DarkRed'
        plt.plot(x_array, gmes.cx_pz_lambda(x_array), color=color_cx, alpha=0.8, lw=2,
                        label=r'$c^x$ model ($p_z$)' )
        # plt.plot(x_array, gmes.cx_pz_tanbeta_lambda(x_array), ':', color='g', alpha=0.8, lw=2, label=r'$c^x$ model ($\tan\beta$)' )
        plt.plot(x_array, gmes.cx_v_lambda(x_array), ':', color='k', alpha=0.8, lw=2,
                        label=r'$c^x$ model ($\mathbf{v}$)' )
        plt.plot(x_array, gmes.vx_interp_fast(x_array), '--', color=color_bounds, alpha=0.8, lw=1,
                        label=r'fast ray $v^x$ bound' )
        plt.plot(x_array, gmes.vx_interp_slow(x_array), '-.', color=color_bounds, alpha=0.8, lw=1,
                        label=r'slow ray $v^x$ bound' )

        _ = plt.xlim(*x_limits) if do_x else plt.xlim(*t_limits)
        _ = plt.ylim(*y_limits) if y_limits is not None else None
        plt.grid(True, ls=':')

        plt.legend(loc=legend_loc, fontsize=12, framealpha=0.95)

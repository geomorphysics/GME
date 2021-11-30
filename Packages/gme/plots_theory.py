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

# Numpy
import numpy as np

# Scipy utils
from scipy.linalg import eig, eigh, det, norm
from scipy.optimize import root_scalar

# SymPy
from sympy import Eq, factor, N, Abs, lambdify, Rational, Matrix, poly, \
                    simplify, diff, sign, sin, tan, deg, solve, sqrt, rad, numer, denom, im, re, \
                    atan, oo

# GMPLib
from gmplib.utils import e2d, omitdict, round as gmround, convert

# GME
from gme.symbols import varphi_r, xiv, xiv_0, pz_min, varphi, px, pz, px_min, \
                        H, beta_max, Lc, gstarhat, xih_0, mu, eta, pxhat, pzhat, rxhat, rzhat, Ci
from gme.equations import px_value
from gme.plot import Graphing

# MatPlotLib
import matplotlib.pyplot as plt
from matplotlib import ticker
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

__all__ = ['TheoryPlots']


class TheoryPlots(Graphing):
    """
    Subclasses :class:`gme.plot.Graphing <plot.Graphing>`.
    """

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
        # fgtx_px_array = np.array([float(re(N(px_pz_lambda(pz_)))) for pz_ in fgtx_pz_array])

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
        px_pz_lambda = lambdify( [pz], re(N(px_pz_eqn.rhs)) )
        fgtx_pz_array = -np.power(10,np.linspace(np.log10(pz_max),np.log10(pz_min), n_points))
        # fgtx_px_array = np.array([float(re(N(px_pz_eqn.rhs.subs({pz:pz_})))) for pz_ in fgtx_pz_array])
        fgtx_px_array = np.array([float(px_pz_lambda(pz_)) for pz_ in fgtx_pz_array])
        return fgtx_px_array, fgtx_pz_array, px_pz_eqn

    def indicatrix(self, gmeq, varphi_, n_points, pz_min=1e-5, pz_max=300):
        rdotx_pz_eqn = gmeq.idtx_rdotx_pz_varphi_eqn.subs({varphi:varphi_})
        rdotz_pz_eqn = gmeq.idtx_rdotz_pz_varphi_eqn.subs({varphi:varphi_})
        rdotx_pz_lambda = lambdify( [pz], re(N(rdotx_pz_eqn.rhs)) )
        rdotz_pz_lambda = lambdify( [pz], re(N(rdotz_pz_eqn.rhs)) )
        fgtx_pz_array = -np.power(10,np.linspace(np.log10(pz_max),np.log10(pz_min), n_points))
        # idtx_rdotx_array = np.array([float(re(N(rdotx_pz_eqn.rhs.subs({pz:pz_}))))
        #                                 for pz_ in fgtx_pz_array])
        # idtx_rdotz_array = np.array([float(re(N(rdotz_pz_eqn.rhs.subs({pz:pz_}))))
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
        px_ = float(N(re(px_pz_eqn.rhs.subs({pz:pz_}))))
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
        plt.plot(beta_array, alpha_array, 'b')
        x_,y_ = np.rad2deg(np.arctan(float(tanbeta_crit_))), np.rad2deg(np.arctan(float(tanalpha_crit_)))
        plt.plot( x_,y_, 'ob' )
        plt.text( x_,y_-y_/9, r'$\beta_c, \,\alpha_c$', color='b', horizontalalignment='center', fontsize=14)
        eta_label = fr'$\eta = ${gmeq.eta}'
        plt.text( 40,y_/4, fr'$\eta = ${gmeq.eta}', color='k', horizontalalignment='center', fontsize=14)
        plt.text( 87,y_*0.67, '(a)' if gmeq.eta==Rational(3,2) else ('(b)' if gmeq.eta==Rational(1,2) else ''),
                  color='k', horizontalalignment='center', fontsize=16 )
        plt.grid('on')
        plt.xlabel(r'Surface tilt  $\beta$   [${\degree}\!$ from horiz]')
        plt.ylabel(r'Ray angle  $\alpha$   [${\degree}\!$ from horiz]')
        np.rad2deg(np.arctan(float(tanbeta_crit_))), np.rad2deg(np.arctan(float(tanalpha_crit_)))

    def beta_anisotropy(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ):
        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(beta_array, alpha_array-beta_array+90, 'b')
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
        plt.ylabel(r'Anisotropy   $\psi = \alpha-\beta+90{\degree}$   [${\degree}\!$]')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.xlim(0,90)
        plt.ylim(0,90)
        plt.plot(beta_array, beta_array, ':')

    def alpha_anisotropy(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ):
        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(alpha_array-beta_array+90, alpha_array, 'b')
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
        plt.xlabel(r'Anisotropy   $\psi = \alpha-\beta+90{\degree}$   [${\degree}\!$]')
        axes = plt.gca()
        axes.invert_xaxis()
        axes.set_aspect(2)

    def alpha_image(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ):
        # Create figure
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(beta_array, beta_array-(alpha_array-beta_array+90), 'b')
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
            tanbeta_max_eqn = Eq(tan(beta_max),oo)
            self.tanbeta_max = None

        px_min_ = round(float(N(px_min_eqn.rhs.subs({varphi:varphi_}))),4)
        pz_min_ = round(float(N(pz_min_eqn.rhs.subs({varphi:varphi_}))),8)
        px_min_, -pz_min_, np.rad2deg(np.arctan(px_min_/pz_min_)) if pz_min_>0 else None

        # px_H_solns = [simplify(soln) for soln in solve(Eq((4*H_parametric_eqn.lhs)**2, (4*H_parametric_eqn.rhs)**2),px**2)]
        # x,z = symbols('x, z')
        px_H_solns = [simplify(sqrt(soln)) for soln in solve( self.H_parametric_eqn ,px**2)]
        pz_H_solns = [simplify(sqrt(soln)).subs({Abs(px):px})
                      for soln in solve( self.H_parametric_eqn ,pz**2)]
        px_H_soln_ = [soln for soln in px_H_solns if Abs(im(N(soln.subs({pz:1}))))<1e-10][0]
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
        #     Eq(beta_crit, round(N(deg(atan(gmeq.tanbeta_crit_eqn.rhs))),2))
        tanbeta_crit = float(N(gmeq.tanbeta_crit_eqn.rhs))

        self.p_infc_array = p_array[np.abs(p_array[:,0]/p_array[:,1])<tanbeta_crit]
        self.p_supc_array = p_array[np.abs(p_array[:,0]/p_array[:,1])>=tanbeta_crit]

        px_poly_eqn = poly(self.H_parametric_eqn)
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
        plt.xticks(xtick_posns, xtick_labels, ha='left', fontsize=15)

        plt.text(*eta_xy_label, rf'$\eta={gmeq.eta}$', transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=18, color='k')
        plt.legend(loc=legend_xy)

        axes.tick_params(axis='x', pad=0, left=True, length=5, width=1, direction='out')

        axes.set_aspect(1)
        axes.set_rmax(scale_fn(r_max_))
        axes.set_rmin(scale_fn(r_min_))
        axes.set_thetamin(0)
        plt.grid(False, ls=':')

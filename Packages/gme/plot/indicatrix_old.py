"""
---------------------------------------------------------------------

Old way of visualizing indicatrix & figuratrix

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`numpy`
  -  :mod:`sympy`
  -  :mod:`matplotlib`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix: https://docs.sympy.org/latest/modules/matrices\
/immutablematrices.html

---------------------------------------------------------------------

"""
import warnings

# Minimal typing
from typing import Tuple

# Numpy
import numpy as np

# SymPy
from sympy import Eq, N, re, factor, lambdify

# MatPlotLib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, RegularPolygon

# GME
from gme.core.symbols import px, pz, varphi, pz_min
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['IndicatrixOld']


class IndicatrixOld(Graphing):
    """
    Old way of visualizing indicatrix & figuratrix.

    Extends :class:`gme.plot.base.Graphing`.
    """

    def comparison_logpolar(
        self,
        gmeq,
        name,
        fig_size=None,
        dpi=None,
        varphi_=1,
        n_points=100,
        idtx_pz_min=1e-3,
        idtx_pz_max=1000,
        fgtx_pz_min=1e-3,
        fgtx_pz_max=1000,
        y_limits=None
    ) -> None:
        """
        Plot both indicatrix and figuratrix on one log-polar graph.
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        plt.title('Erosion indicatrix & figuratrix', va='bottom', pad=15)
        plt.grid(False, ls=':')

# px_pz_lambda \
# = lambdify( [pz], gmeq.fgtx_px_pz_varphi_eqn.rhs.subs({varphi:varphi_}) )
# fgtx_pz_array \
# = -np.power(10,np.linspace(np.log10(1000),np.log10(1e-6),n_points))
# fgtx_px_array
# = np.array([float(re(N(px_pz_lambda(pz_)))) for pz_ in fgtx_pz_array])

        # Compute figuratrix
        fgtx_px_array, fgtx_pz_array, _ \
            = self.figuratrix(gmeq, varphi_, n_points,
                              fgtx_pz_min, fgtx_pz_max)
        fgtx_p_array = np.log10(np.sqrt(fgtx_px_array**2+fgtx_pz_array**2))
        fgtx_tanbeta_array = -fgtx_px_array/fgtx_pz_array
        fgtx_beta_array = np.arctan(fgtx_tanbeta_array)
        fgtx_theta_array \
            = np.concatenate([fgtx_beta_array,
                              (2*np.pi-fgtx_beta_array)[::-1]])
        fgtx_p_array = np.concatenate([fgtx_p_array, fgtx_p_array[::-1]])

        # Compute indicatrix
        idtx_rdotx_array, idtx_rdotz_array, _, _ \
            = self.indicatrix(gmeq, varphi_, n_points,
                              idtx_pz_min, idtx_pz_max)
        idtx_rdot_array = np.log10(
            np.sqrt(idtx_rdotx_array**2+idtx_rdotz_array**2))
        idtx_alpha_array = np.arctan(-idtx_rdotx_array/idtx_rdotz_array)
        idtx_theta_negrdot_array = idtx_alpha_array[idtx_rdotz_array < 0]
        idtx_rdot_negrdot_array = idtx_rdot_array[idtx_rdotz_array < 0]
        idtx_theta_posrdot_array = np.pi+idtx_alpha_array[idtx_rdotz_array > 0]
        idtx_rdot_posrdot_array = idtx_rdot_array[idtx_rdotz_array > 0]
        idtx_theta_array \
            = np.concatenate([idtx_theta_negrdot_array,
                              idtx_theta_posrdot_array])
        idtx_rdot_array \
            = np.concatenate([idtx_rdot_negrdot_array,
                              idtx_rdot_posrdot_array])

        # Compute reference unit circle
        unit_circle_beta_array = np.linspace(0, 2*np.pi)

        # Do the basic plotting
        # idtx_label = r'indicatrix $F(\mathbf{v},\alpha)$'
        # fgtx_label = r'figuratrix $F^*(\mathbf{\widetilde{p}},\beta)$'
        idtx_label = r'ray velocity'
        fgtx_label = r'normal slowness'
        plt.polar(fgtx_theta_array, fgtx_p_array,
                  'DarkBlue', '-', lw=1.5,
                  label=fgtx_label)
        plt.polar(idtx_theta_array, idtx_rdot_array,
                  'DarkRed', ls='-', lw=1.5,
                  label=idtx_label)
        plt.polar(2*np.pi-idtx_theta_array, idtx_rdot_array,
                  'DarkRed', ls='-', lw=1.5)
        # plt.polar( idtx_theta_posrdot_array, idtx_rdot_posrdot_array,
        # 'magenta', ls='-', lw=3)
        # plt.polar( idtx_theta_negrdot_array, idtx_rdot_negrdot_array,
        # 'k', ls='-', lw=3)
        plt.polar(unit_circle_beta_array, unit_circle_beta_array*0,
                  'g', ':', lw=1,
                  label='unit circle')

        # Critical angles
        beta_crit = np.arctan(float(gmeq.tanbeta_crit))
        alpha_crit = np.pi/2+np.arctan(float(gmeq.tanalpha_crit))
        plt.polar([beta_crit, beta_crit], [-2, 3],
                  ':', color='DarkBlue',
                  label=r'$\beta=\beta_c$')
        plt.polar([alpha_crit, alpha_crit], [-2, 3],
                  ':', color='DarkRed',
                  label=r'$\alpha=\alpha_{\mathrm{ext}}$')

        # Labelling etc
        posn_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
        xtick_posns = [(-np.pi/i_ if i_ != 0 else np.pi) for i_ in posn_list]
        xtick_posns = plt.xticks()[0]
        xtick_labels \
            = [(r'$\frac{\pi}{' + f'{int(x_)}' + '}$' if x_ > 0
                else r'$-\frac{\pi}{' + f'{int(-x_)}' + '}$' if x_ != 0
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
            y_limits = [-2, 3]
            ytick_posns = [-1, 0, 1, 2, 3, 4]
        else:
            ytick_posns = list(range(int(y_limits[0]), int(y_limits[1])+2))
        plt.ylim(*y_limits)
        ytick_labels = [rf'$10^{int(y_)}$' for y_ in ytick_posns]
        plt.yticks(ytick_posns, ytick_labels)

        axes = plt.gca()
        plt.text(np.pi/1.1, (1+y_limits[1]-y_limits[0])*2/3+y_limits[0],
                 rf'$\eta={gmeq.eta_}$', fontsize=16)

        axes.set_theta_zero_location("S")

        handles, labels = axes.get_legend_handles_labels()
        subset = [3, 2, 0, 5, 6]
        handles = [handles[idx] for idx in subset]
        labels = [labels[idx] for idx in subset]
        # Hacked fix to bug in Matplotlib that adds a bogus entry here
        axes.legend(handles, labels, loc='upper left', framealpha=1)

    def text_labels(self, gmeq, varphi_, px_, pz_, rdotx_, rdotz_,
                    zoom_factor, do_text_labels) -> None:
        """
        TBD
        """
        xy_ = (2.5, -2) if zoom_factor >= 1 else (1.25, 0.9)
        plt.text(*xy_, rf'$\varphi={varphi_}\quad\eta={gmeq.eta_}$',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, color='k')
        if do_text_labels:
            plt.text(rdotx_*0.97, rdotz_+0.2, r'$\mathbf{v}$',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=15, color='r')
            sf = 1.2 if varphi_ >= 0.5 else 0.85
            plt.text(px_+0.25, pz_*sf, r'$\mathbf{\widetilde{p}}$',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=15, color='b')

    def arrows(self, px_, pz_, rdotx_, rdotz_) -> None:
        """
        TBD
        """
        p_ = np.sqrt(px_**2+pz_**2)
        # Arrow for rdot direction
        plt.arrow(0, 0, rdotx_, rdotz_,
                  head_width=0.08, head_length=0.16, lw=1,
                  length_includes_head=True, ec='r', fc='r')
        # Fishbone for p direction
        np_ = int(0.5+((p_)*5)) if p_ >= 0.2 else 1
        hw_ = 0.2  # if do_half else 0.2
        plt.arrow(px_, pz_, -px_, -pz_, color='b',
                  head_width=hw_, head_length=-0.1, lw=1,
                  shape='full', overhang=1,
                  length_includes_head=True,
                  head_starts_at_zero=False,
                  ec='b')
        for i_head in list(range(1, np_)):
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

    def lines_and_points(self, pd, axes, zoomx, do_pz, do_shapes) -> None:
        """
        TBD
        """
        # Unpack
        n_vertices_, ls_ = pd['n_vertices'], pd['ls']
        px_, pz_, rdotx_, rdotz_ = pd['px'], pd['pz'], pd['rdotx'], pd['rdotz']
        psqrd_ = (px_**2+pz_**2)
        plt.plot([px_/psqrd_, rdotx_],
                 [pz_/psqrd_, rdotz_], c='r', ls=ls_, lw=1)
        plt.plot([0, px_], [0, pz_], c='b', ls=ls_, lw=1)
        if do_pz:
            plt.plot([zoomx[0], zoomx[1]], [pz_, pz_], c='b', ls=':', lw=1)
        if psqrd_ < 1:
            plt.plot([px_, px_/psqrd_], [pz_, pz_/psqrd_], c='b', ls=ls_, lw=1)
        if do_shapes:
            axes.add_patch(RegularPolygon((px_, pz_),
                                          n_vertices_, radius=0.08,
                                          lw=1, ec='k', fc='b'))
            axes.add_patch(RegularPolygon((rdotx_, rdotz_),
                                          n_vertices_, radius=0.08,
                                          lw=1, ec='k', fc='r'))
        else:
            axes.add_patch(
                Circle((px_, pz_), radius=0.04, lw=1, ec='k', fc='b'))
            axes.add_patch(
                Circle((rdotx_, rdotz_), radius=0.04, lw=1, ec='k', fc='r'))

    def annotations(self, axes, beta_, tanalpha_) -> None:
        """
        TBD
        """
        # alpha arc
        plt.text(0.45, -0.05, r'$\alpha$', color='DarkRed',
                 # transform=axes.transAxes,
                 fontsize=12,
                 horizontalalignment='center', verticalalignment='center')
        axes.add_patch(
            mpatches.Arc((0, 0), 1.2, 1.2, color='DarkRed',
                         linewidth=0.5, fill=False, zorder=2,
                         theta1=270,
                         theta2=90+np.rad2deg(np.arctan(tanalpha_)))
            )
        # beta arc
        plt.text(0.08, -0.30, r'$\beta$', color='DarkBlue',
                 # transform=axes.transAxes,
                 fontsize=10,
                 horizontalalignment='center', verticalalignment='center')
        axes.add_patch(
            mpatches.Arc((0, 0), 0.9, 0.9, color='DarkBlue',
                         linewidth=0.5, fill=False, zorder=2,
                         theta1=270,
                         theta2=270+np.rad2deg(beta_))
            )
        # "faster" direction
        plt.text(3.05, 1.6, 'faster', color='r',  # transform=axes.transAxes,
                 fontsize=12, rotation=55,
                 horizontalalignment='center', verticalalignment='center')
        # "faster" direction
        plt.text(1.03, -2, 'slower', color='b',  # transform=axes.transAxes,
                 fontsize=12, rotation=-85,
                 horizontalalignment='center', verticalalignment='center')

    def legend(
        self,
        gmeq,
        axes,
        do_legend,
        do_half,
        do_ray_slowness=False
    ) -> None:
        """
        TBD
        """
        if not do_legend:
            return
        handles, labels = axes.get_legend_handles_labels()
        if gmeq.eta_ >= 1:
            loc_ = 'center right' if do_half else \
                 'upper left' if do_ray_slowness else \
                 'lower left'
        else:
            loc_ = 'upper right' if do_half else \
                 'upper left' if do_ray_slowness else \
                 'lower left'
        axes.legend(handles[::-1], labels[::-1], loc=loc_)

    def figuratrix(self, gmeq, varphi_, n_points, pz_min_=1e-5, pz_max_=50) \
            -> Tuple[np.array, np.array, Eq]:
        """
        TBD
        """
        px_pz_eqn \
            = Eq(px,
                 factor(gmeq.fgtx_px_pz_varphi_eqn.rhs.subs({varphi: varphi_}))
                 )
        px_pz_lambda = lambdify([pz], re(N(px_pz_eqn.rhs)))
        fgtx_pz_array \
            = -np.power(10, np.linspace(np.log10(pz_max_),
                                        np.log10(pz_min_), n_points))
        # fgtx_px_array = np.array([float(re(N(px_pz_eqn.rhs.subs({pz:pz_}))))
        #          for pz_ in fgtx_pz_array])
        fgtx_px_array = np.array([float(px_pz_lambda(pz_))
                                 for pz_ in fgtx_pz_array])
        return fgtx_px_array, fgtx_pz_array, px_pz_eqn

    def indicatrix(self, gmeq, varphi_, n_points, pz_min_=1e-5, pz_max_=300) \
            -> Tuple[np.array, np.array, Eq, Eq]:
        """
        TBD
        """
        rdotx_pz_eqn = gmeq.idtx_rdotx_pz_varphi_eqn.subs({varphi: varphi_})
        rdotz_pz_eqn = gmeq.idtx_rdotz_pz_varphi_eqn.subs({varphi: varphi_})
        rdotx_pz_lambda = lambdify([pz], re(N(rdotx_pz_eqn.rhs)))
        rdotz_pz_lambda = lambdify([pz], re(N(rdotz_pz_eqn.rhs)))
        fgtx_pz_array \
            = -np.power(10, np.linspace(np.log10(pz_max_),
                                        np.log10(pz_min_), n_points))
        # idtx_rdotx_array \
# = np.array([float(re(N(rdotx_pz_eqn.rhs.subs({pz:pz_}))))
        #                                 for pz_ in fgtx_pz_array])
        # idtx_rdotz_array \
# = np.array([float(re(N(rdotz_pz_eqn.rhs.subs({pz:pz_}))))
        #                                 for pz_ in fgtx_pz_array])
        idtx_rdotx_array \
            = np.array([float(rdotx_pz_lambda(pz_)) for pz_ in fgtx_pz_array])
        idtx_rdotz_array \
            = np.array([float(rdotz_pz_lambda(pz_)) for pz_ in fgtx_pz_array])
        return idtx_rdotx_array, idtx_rdotz_array, rdotx_pz_eqn, rdotz_pz_eqn

    def plot_figuratrix(self, fgtx_px_array, fgtx_pz_array,
                        maybe_recip_fn, do_ray_slowness=False) -> None:
        """
        TBD
        """
        # label = r'normal speed' if do_ray_slowness else r'figuratrix $F^*$'
        label = r'normal velocity' if do_ray_slowness else r'normal slowness'
        plt.plot(*maybe_recip_fn(-fgtx_px_array, fgtx_pz_array),
                 lw=2, c='DarkBlue', ls='-', label=label)
        plt.plot(*maybe_recip_fn(fgtx_px_array, fgtx_pz_array),
                 lw=2, c='DarkBlue', ls='-')

    def plot_indicatrix(self, idtx_rdotx_array, idtx_rdotz_array,
                        maybe_recip_fn, do_ray_slowness=False) -> None:
        """
        TBD
        """
        # label = r'ray slowness' if do_ray_slowness else r'indicatrix $F$'
        label = r'ray slowness' if do_ray_slowness else r'ray velocity'
        plt.plot(*maybe_recip_fn(idtx_rdotx_array, idtx_rdotz_array),
                 lw=2, c='DarkRed', ls='-', label=label)
        plt.plot(*maybe_recip_fn(-idtx_rdotx_array, idtx_rdotz_array),
                 lw=2, c='DarkRed', ls='-')

    def plot_unit_circle(self, varphi_, do_varphi_circle) -> None:
        """
        TBD
        """
        unit_circle_beta_array = np.linspace(0, np.pi)
        plt.plot(np.cos(unit_circle_beta_array*2),
                 np.sin(unit_circle_beta_array*2),
                 lw=1, c='g', ls='-', label='unit circle')
        if do_varphi_circle:
            plt.plot(varphi_*np.cos(unit_circle_beta_array*2),
                     varphi_*np.sin(unit_circle_beta_array*2),
                     lw=1, c='g', ls=':', label=rf'$\varphi={varphi_}$')

    def relative_geometry(
        self,
        gmeq,
        name,
        fig_size=None,
        dpi=None,
        varphi_=1,
        zoom_factor=1,
        do_half=False,
        do_legend=True,
        do_text_labels=True,
        do_arrows=True,
        do_lines_points=True,
        do_shapes=False,
        do_varphi_circle=False,
        do_pz=False,
        do_ray_slowness=False,
        x_max=3.4,
        n_points=100,
        pz_min_=1e-1
    ) -> None:
        r"""
        Plot the loci of :math:`\mathbf{\widetilde{p}}` and :math:`\mathbf{r}`
        and their behavior defined by :math:`F` relative to the
        :math:`\xi` circle.

        Args:
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            varphi_choice:
                the value of :math:`\varphi` to use
            zoom_factor:
                fractional zoom factor relative to preassigned graph limits
            do_half:
                plot only for :math:`x\geq0`?
            do_annotations:
                plot annotated arcs, faster/slower arrows?
            do_legend:
                display the legend?
            do_text_labels:
                display text annotations?
            do_arrows:
                display vector/covector arrows?
            do_lines_points:
                display dashed/dotted tangent lines etc?
            do_shapes:
                plot key points using polygon symbols
                rather than simple circles
            do_varphi_circle :
                plot the :math:`\xi` (typically unit) circle?
            do_pz:
                plot horizontal dotted line indicating magnitude of :math:`p_z`
            do_ray_slowness:
                invert ray speed (but keep ray direction) in indicatrix?
        """

        # Create figure
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Adjust plot scale, limits
        axes = plt.gca()
        axes.set_aspect(1)
        plt.grid(True, ls=':')
        x_min = -0.1 if do_half else -x_max
        y_minmax = 2.4 if gmeq.eta_ >= 1 else 2.4
        zoomx = np.array([x_min, x_max]) * zoom_factor
        zoomz = np.array([-y_minmax, y_minmax]) * zoom_factor
        plt.xlim(zoomx)
        plt.ylim(zoomz)
        plt.xlabel(r'Horizontal component')
        plt.ylabel(r'Vertical component')

        # Prep
        def recip_fn(x_, z_): return [x_/(x_**2+z_**2), z_/(x_**2+z_**2)]
        def null_fn(x_, z_): return [x_, z_]
        maybe_recip_fn = recip_fn if do_ray_slowness else null_fn
        points_tangents_dicts = {
            0.1: {'n_vertices': 3, 'ls': ':',
                  'pz_min': pz_min, 'pz_max': 1000},
            0.15: {'n_vertices': 3, 'ls': ':',
                   'pz_min': pz_min_, 'pz_max': 1000},
            0.5: {'n_vertices': 4, 'ls': '--',
                  'pz_min': pz_min_, 'pz_max': 100},
            1: {'n_vertices': 4, 'ls': '--',
                'pz_min': pz_min_, 'pz_max': 100},
            1.3: {'n_vertices': 5, 'ls': '-.',
                  'pz_min': pz_min_, 'pz_max': 10},
            2: {'n_vertices': 5, 'ls': '-.',
                'pz_min': pz_min_, 'pz_max': 10},
            3: {'n_vertices': 5, 'ls': '-.',
                'pz_min': pz_min_, 'pz_max': 3}
        }

        # Compute some stuff
        pdict = points_tangents_dicts[varphi_]
        fgtx_px_array, fgtx_pz_array, px_pz_eqn \
            = self.figuratrix(gmeq, varphi_, n_points)
        idtx_rdotx_array, idtx_rdotz_array, rdotx_pz_eqn, rdotz_pz_eqn \
            = self.indicatrix(gmeq, varphi_, n_points,
                              pz_min_=pdict['pz_min'], pz_max_=pdict['pz_max'])
        pz_ = -np.cos(np.pi/4)
        px_ = float(N(re(px_pz_eqn.rhs.subs({pz: pz_}))))
        rdotx_ = float(re(rdotx_pz_eqn.rhs.subs({pz: pz_})))
        rdotz_ = float(re(rdotz_pz_eqn.rhs.subs({pz: pz_})))
        # tanalpha_ = (-rdotx_/rdotz_)
        pdict.update({'varphi': varphi_, 'px': px_, 'pz': pz_,
                      'rdotx': rdotx_, 'rdotz': rdotz_})

        # Do the plotting
        self.plot_indicatrix(idtx_rdotx_array, idtx_rdotz_array,
                             maybe_recip_fn, do_ray_slowness)
        self.plot_figuratrix(fgtx_px_array, fgtx_pz_array,
                             maybe_recip_fn, do_ray_slowness)
        self.plot_unit_circle(varphi_, do_varphi_circle)
        if do_lines_points:
            self.lines_and_points(pdict, axes, zoomx, do_pz, do_shapes)
        self.text_labels(gmeq, varphi_, px_, pz_, rdotx_, rdotz_,
                         zoom_factor, do_text_labels)
        if do_arrows:
            self.arrows(px_, pz_, rdotx_, rdotz_)
        # if do_annotations: self.annotations(axes, beta_, tanalpha_)
        self.legend(gmeq, axes, do_legend, do_half, do_ray_slowness)

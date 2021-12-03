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

# pylint: disable=line-too-long, invalid-name, too-many-locals, no-self-use, multiple-statements
import warnings

# Minimal typing
from typing import Tuple

# Numpy
import numpy as np

# SymPy
from sympy import Eq, Abs, lambdify, Rational, Matrix, poly, \
                    simplify, diff, deg, solve, sqrt, rad, numer, denom, im, re

# GMPLib
from gmplib.utils import e2d, omitdict

# GME
from gme.core.symbols import H, Lc, gstarhat, xih_0, mu, eta, \
                             pxhat, pzhat, rxhat, Ci
from gme.core.equations import px_value
from gme.plot.base import Graphing

# MatPlotLib
import matplotlib.pyplot as plt
from matplotlib import ticker
# from matplotlib.ticker import FormatStrFormatter
# import matplotlib.patches as mpatches
# from matplotlib.patches import Patch, FancyArrow, FancyArrowPatch, Arrow, Rectangle, Circle, RegularPolygon,\
#                                 ArrowStyle, ConnectionPatch, Arc
# from matplotlib.spines import Spine
# from matplotlib.legend_handler import HandlerPatch
# from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings("ignore")

__all__ = ['SlicingMath','SlicingPlots']

class SlicingMath():
    """
    TBD
    """

    def __init__(self, gr, sub_, var_list, do_modv=True) -> None:
        r"""
        Generate lambdas for H, Ci, metric signature, gstarhat, v, modv, pxhat and pzhat
        """
        self.H_Ci_eqn = gr.H_Ci_eqn
        self.Ci_H0p5_eqn = gr.Ci_H0p5_eqn
        self.gstarhat_eqn = gr.gstarhat_eqn
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

    def define_H_lambda(self, sub_, var_list) -> None:
        r"""
        TBD
        """
        self.H_lambda =  lambdify(var_list, self.H_Ci_eqn.rhs.subs({mu:eta/2}).subs(sub_), 'numpy')

    def define_Ci_lambda(self, sub_, var_list) -> None:
        r"""
        TBD
        """
        self.Ci_lambda = lambdify(var_list, self.Ci_H0p5_eqn.rhs.subs({H:Rational(1,2), mu:eta/2}).subs(sub_), 'numpy')

    def define_Hessian_eigenvals(self, sub_, var_list) -> None:
        r"""
        TBD
        """
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

    def define_gstarhat_lambda(self, sub_, var_list) -> None:
        r"""
        TBD
        """
        self.gstarhat_lambda =  lambdify( var_list, self.gstarhat_eqn.rhs.subs({mu:eta/2}).subs(sub_), modules='numpy' )

    def define_v_pxpzhat_lambda(self, sub_) -> None:
        r"""
        TBD
        """
        self.v_pxpzhat_lambda = lambdify( (pxhat,pzhat),
            simplify(((self.gstarhat_eqn.rhs.subs({mu:eta/2}).subs(sub_))*Matrix([pxhat,pzhat]))), modules='numpy' )

    def define_modv_pxpzhat_lambda(self, sub_) -> None:
        r"""
        TBD
        """
        self.modv_pxpzhat_lambda = lambdify( (pxhat,pzhat),
            simplify(((self.gstarhat_eqn.rhs.subs({mu:eta/2}).subs(sub_))*Matrix([pxhat,pzhat])).norm()), modules='numpy' )

    def pxhatsqrd_Ci_polylike_eqn(self, sub_, pzhat_ ) -> Eq:
        r"""
        TBD
        """
        tmp = (self.H_Ci_eqn.rhs.subs({pzhat:pzhat_, H:Rational(1,2), mu:eta/2})
               .subs(omitdict(sub_,[Ci])))**2
        return Eq(4*numer(tmp)-denom(tmp),0)

    def pxhat_Ci_soln(self, eqn_, sub_, rxhat_, tolerance=1e-3) -> float:
        r"""
        TBD
        """
        solns_ = Matrix(solve(eqn_.subs(sub_),pxhat**2)).subs({rxhat:rxhat_})
        return float(sqrt(
            [re(soln_) for soln_ in solns_.n() if Abs(im(soln_))<tolerance and re(soln_)>0][0]
        ))

    def pxpzhat0_values(self, contour_values_, sub_) -> list:
        r"""
        TBD
        """
        pxpzhat_values_ = [[float(0),float(0)]]*len(contour_values_)
        for i_,Ci_ in enumerate(contour_values_):
            tmp_sub_ = omitdict(sub_,[rxhat,Ci])
            tmp_sub_[Ci] = rad(Ci_)
            # pzhat for pxhat=1 which is true when rxhat=0
            pzhat_ = self.pzhat_lambda(tmp_sub_,0,1).n()
            eqn_ = self.pxhatsqrd_Ci_polylike_eqn(tmp_sub_, pzhat_)
            x_ = float(self.pxhat_Ci_soln(eqn_, tmp_sub_, rxhat.subs(sub_)))
            y_ = float(self.pzhat_lambda(tmp_sub_,0,1).n())
            pxpzhat_values_[i_] = [x_,y_]
        return pxpzhat_values_


class SlicingPlots(Graphing):
    """
    Subclasses :class:`gme.plot.Graphing <plot.Graphing>`.
    """

    def __init__(self, gmeq, grid_res=301, dpi=100, font_size=11) -> None:
        r"""
        Constructor method.

        Args:
            gmeq (XXX): XXX
            grid_res (XXX): XXX
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

    def plot_modv_pzhat_slice(self, sm, sub_, psub_) -> str:
        r"""
        TBD
        """
        fig_name = ('v_pz_H0p5'
                    + f'_eta{float(eta.subs(sub_).n()):g}'
                    + f'_Ci{deg(Ci.subs(sub_))}'
                    + f'_rxhat{float(rxhat.subs(psub_).n()):g}'
                   ).replace('.','p')
        _ = self.create_figure(fig_name, fig_size=(6,5))
        axes = plt.gca()

        pxhat_eqn_ = sm.pxhatsqrd_Ci_polylike_eqn(sub_, pzhat) #.subs(sub_)
        pxhat_poly_ = poly(pxhat_eqn_.lhs.subs(psub_).n(),pxhat)

        pzhat0_ = float(sm.pzhat_lambda(sub_,0,1).n())
        # pxhat0_ = float(px_value(rxhat.subs(psub_), pzhat0_, pxhat_poly_,
        #                          px_var_=pxhat, pz_var_=pzhat))
        # modv0_ = sm.modv_pxpzhat_lambda(pxhat0_,pzhat0_)

        # For H=1/2
        pzhat_array = np.flipud(np.linspace(-30,0,31, endpoint=bool(rxhat.subs(psub_)<0.95 and eta.subs(sub_)<1)))
        pxhat_array = np.array([float(px_value(rxhat.subs(psub_),pzhat_, pxhat_poly_, px_var_=pxhat, pz_var_=pzhat))
                       for pzhat_ in pzhat_array])

        # H_array = np.array([float(sm.H_lambda(pxhat_,pzhat_)) for pxhat_, pzhat_ in zip(pxhat_array, pzhat_array)])

        modp_array = np.sqrt(pxhat_array**2+pzhat_array**2)
        modv_array = np.array([sm.modv_pxpzhat_lambda(pxhat_,pzhat_) for pxhat_,pzhat_ in zip(pxhat_array,pzhat_array)])
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

    def H_rxpx_contours(self, sm, sub_, psf, do_Ci, **kwargs) -> str:
        r"""
        TBD
        """
        return self.plot_Hetc_contours(sm, (self.rxpxhat_grids[0],self.rxpxhat_grids[1]*psf), sub_, do_Ci=do_Ci,
                                        do_fmt_labels=bool(psf>1000), do_aspect=False, do_rxpx=True,
                                        do_grid=True,
                                        **kwargs)

    def H_pxpz_contours(self, sm, sub_, psf, do_Ci, **kwargs) -> str:
        r"""
        TBD
        """
        return self.plot_Hetc_contours(sm, (self.pxpzhat_grids[0]*psf, self.pxpzhat_grids[1]*psf), sub_, do_Ci=do_Ci,
                                        do_fmt_labels=bool(psf>1000), do_aspect=True, do_rxpx=False,
                                        do_grid=False,
                                        **kwargs )

    def plot_Hetc_contours(self, sm, grids_, sub_, do_Ci, do_modv=False,
                            do_fmt_labels=False, do_aspect=True, do_rxpx=False, pxpz_points=None,
                            do_log2H=False, do_siggrid=True, do_black_contours=False, do_grid=True,
                            contour_nlevels=None, contour_range=None, v_contour_range=None,
                            contour_values=None, contour_label_locs=None) -> str:
        r"""
        TBD
        """
        # Create figure
        title = ( ('Ci' if do_Ci else ('v' if do_modv else 'H')) + ('_pslice' if do_rxpx else '_pslice')
                    + f'_eta{float(eta.subs(sub_).n()):g}'
                    + ( f'_Ci{deg(Ci.subs(sub_))}' if do_rxpx else f'_rxhat{float(rxhat.subs(sub_).n()):g}')
                ).replace('.','p')
        fig = self.create_figure(title, fig_size=(6,6))
        axes = plt.gca()
        labels = (r'\hat{r}^x', r'\hat{p}_x') if do_rxpx else (r'\hat{p}_x', r'\hat{p}_z')
        xlabel, ylabel = [rf'${label_}$' for label_ in labels]
        vars_label = rf'$\left({labels[0]},{labels[1]}\right)$'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if do_grid: plt.grid(':')

        # Generate a grid of H or Ci for a meshgrid of (px,pz) or (rx,px)
        H_grid_ = sm.Ci_lambda(*grids_) if do_Ci else sm.H_lambda(*grids_)
        modv_grid_ = sm.modv_pxpzhat_lambda(*grids_) if (do_modv and not do_rxpx) else None
        if do_siggrid:
            gstar_signature_grid_ = np.array(grids_[0].shape) # for i_ in [1,2]]
            gstar_signature_grid_ = np.array([sm.gstar_signature_lambda(x_,y_)
                     for x_,y_ in zip(grids_[0].flatten(),grids_[1].flatten())])\
                     .reshape((grids_[0]).shape)
            gstar_signature_grid_[np.isnan(gstar_signature_grid_)] = 0

        # Axis labeling
        if do_fmt_labels:
            for axis_ in [axes.xaxis, axes.yaxis]:
                axis_.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

        # Metric signature colour background
        if do_siggrid:
            cmap_name = 'PiYG'  #, 'plasma_r'
            cmap_ = plt.get_cmap(cmap_name)
            cf = axes.contourf(*grids_, gstar_signature_grid_, levels=1, cmap=cmap_)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('top', size='6%', pad=0.4)
            label_levels = np.array([0.25,0.75])
            # mypy: Incompatible types in assignment (expression has type "List[str]", variable has type "Tuple[str, str]")
            tick_labels: Tuple[str,str] = ('mixed: -,+','positive: +,+')
            cbar = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=label_levels,
                               label='metric signature')
            cbar.ax.set_xticklabels(tick_labels)
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
                levels_H0p5 = np.array([0])
                if do_rxpx:
                    fmt_H = lambda H: \
                        rf'$H={np.round(10**H/2,2 if 10**H<0.5 else (1 if 10**H<5 else 0)):g}$' #% f'{}'
                else:
                    fmt_H = lambda H: rf'$2H=10^{H:g}$' # % f'{H:g}'
                fmt_H0p5 = lambda H: r'$H=0.5$'
                manual_location = (0.1,-7)
            else:
                levels_ = np.concatenate([
                    np.linspace(0.0,0.5, contour_nlevels[0], endpoint=False),
                    np.flip(np.linspace(contour_range[1],0.5, contour_nlevels[1], endpoint=False))
                ])
                levels_H0p5 = np.array([0.5])
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
                        manual=[(manual_location[0],manual_location[1])] if manual_location is not None else None)
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

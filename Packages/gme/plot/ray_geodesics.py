"""
---------------------------------------------------------------------

Visualization of geodesics.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`NumPy <numpy>`
  -  :mod:`SciPy <scipy>`
  -  :mod:`SymPy <sympy>`
  -  :mod:`MatPlotLib <matplotlib>`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html

---------------------------------------------------------------------

"""
import warnings

# Typing
from typing import Tuple, Dict, List, Optional

# Numpy
import numpy as np

# Scipy utils
from scipy.linalg import eigh, det

# SymPy
from sympy import deg, re, Matrix

# MatPlotLib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# GME
from gme.core.symbols import Ci, rx, rdotx, rdotz
from gme.core.equations import Equations
from gme.ode.time_invariant import TimeInvariantSolution
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['RayGeodesics']


class RayGeodesics(Graphing):
    """
    Visualization of geodesics.

    Extends :class:`gme.plot.base.Graphing`.

    Args:
        gmes:
            instance of single-ray solution class
            defined in :mod:`gme.ode.single_ray`
        gmeq:
            GME model equations class instance defined in
            :mod:`gme.core.equations`
        n_points:
            optional sample rate along each curve
    """

    def __init__(
            self,
            gmes,
            gmeq,
            n_points,
            do_recompute=False
            ) -> None:
        r"""
        Constructor method
        """
        super().__init__()
        # if not hasattr(self,'x_array') or n_points!=len(self.x_array):
        # HACK!!!
        do_recompute = True
        print('(Re)computing g matrices')

        rx_array = gmes.rx_array
        x_min, x_max = rx_array[0], rx_array[-1]
        # HACK!!!
        self.x_array = np.linspace(x_min, x_max, n_points)  # \
        # if do_recompute or not hasattr(self,'x_array') else self.x_array
        self.t_array = gmes.t_interp_x(self.x_array)  # \
        # if do_recompute or not hasattr(self,'t_array') else self.t_array
        self.rz_array = gmes.rz_interp(self.x_array)  # \
        # if do_recompute or not hasattr(self,'rz_array') else self.rz_array
        self.vx_array = gmes.rdotx_interp(self.x_array)  # \
        # if do_recompute or not hasattr(self,'vx_array') else self.vx_array
        self.vz_array = gmes.rdotz_interp(self.x_array)  # \
        # if do_recompute or not hasattr(self,'vz_array') else self.vz_array
        x_array = self.x_array
        # t_array  = self.t_array
        # rz_array = self.rz_array
        vx_array = self.vx_array
        vz_array = self.vz_array

        if do_recompute:
            self.gstar_matrices_list: List[Matrix] = []
            self.gstar_matrices_array: List[np.array] = []
            self.g_matrices_list: List[Matrix] = []
            self.g_matrices_array: List[np.array] = []
        if not hasattr(gmeq, 'gstar_ij_mat'):
            return
        try:
            self.gstar_matrices_list \
                = [gmeq.gstar_ij_mat.subs({rx: x_, rdotx: vx_, rdotz: vz_})
                    for x_, vx_, vz_ in zip(x_array, vx_array, vz_array)]
        except ValueError as e:
            print(f'Failed to (re)generate gstar_matrices_list: "{e}"')

        try:
            self.gstar_matrices_array \
                = [np.array([float(re(elem_)) for elem_ in g_]).reshape(2, 2)
                    for g_ in self.gstar_matrices_list]
        except ValueError as e:
            print(f'Failed to (re)generate gstar_matrices_array: "{e}"')

        try:
            self.g_matrices_list \
                = [gmeq.g_ij_mat.subs({rx: x_, rdotx: vx_, rdotz: vz_})
                    for x_, vx_, vz_ in zip(x_array, vx_array, vz_array)]
        except ValueError as e:
            print(f'Failed to (re)generate g_matrices_list: "{e}"')

        try:
            self.g_matrices_array \
                = [np.array([float(re(elem_)) for elem_ in g_]).reshape(2, 2)
                    for g_ in self.g_matrices_list]
        except ValueError as e:
            print(f'Failed to (re)generate g_matrices_array: "{e}"')

    def profile_g_properties(
            self,
            gmes: TimeInvariantSolution,
            gmeq: Equations,
            sub: Dict,
            name: str,
            fig_size: Optional[Tuple[float, float]] = None,
            dpi: Optional[int] = None,
            y_limits=None,
            # n_points=121,
            # do_pub_label=False, pub_label='',
            do_gstar=False,
            do_det=False,
            do_eigenvectors=False,
            eta_label_xy=None,
            do_etaxi_label=True,
            legend_loc='lower left',
            # do_mod_v=False,
            # do_recompute=False
            do_pv=False
            ) -> None:
        r"""
        Plot velocity :math:`\dot{r}` along a ray.

        Args:
            gmes:
                instance of single ray solution class defined in
                :mod:`gme.ode.single_ray`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            n_points: sample rate along each curve
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        y_limits = [None, None] if y_limits is None else y_limits
        axes = plt.gca()

        # HACK
        # self.prep_g_arrays(gmes, gmeq, n_points, do_recompute)

        if do_gstar:
            g_matrices_array = self.gstar_matrices_array
        else:
            g_matrices_array = self.g_matrices_array
        x_array = self.x_array
        # t_array  = self.t_array
        rz_array = self.rz_array
        vx_array = self.vx_array
        vz_array = self.vz_array

        if do_gstar:
            # Use of lambdified g matrix here fails for eta=1/4, sin(beta)
            #    for some reason
            # g_matrices_list = [gmeq.gstar_ij_mat_lambdified(x_,vx_,vz_)
            #                for x_,vx_,vz_ in zip(x_array,vx_array,vz_array)]
            g_label = '{g^*}'
            m_label = 'co-metric'
            h_label = 'H'
            eta_label_xy = [0.5, 0.2] if eta_label_xy is None else eta_label_xy
        else:
            # Use of lambdified g* matrix here fails for eta=1/4, sin(beta)
            #   for some reason
            # g_matrices_list = [gmeq.g_ij_mat_lambdified(x_,vx_,vz_)
            #               for x_,vx_,vz_ in zip(x_array,vx_array,vz_array)]
            g_label = '{g}'
            m_label = 'metric'
            h_label = 'L'
            eta_label_xy = [
                0.5, 0.85] if eta_label_xy is None else eta_label_xy
        # g_eigenvalues_array
        #  = np.array([np.real(eig(g_)[0]) for g_ in g_matrices_array])
        # The metric tensor matrices are symmetric therefore Hermitian
        #  so we can use 'eigh'
        # print(f'g_matrices_array = {g_matrices_array}')
        if g_matrices_array is not None:
            g_eigh_array = [eigh(g_) for g_ in g_matrices_array]
            g_det_array = np.array([det(g_) for g_ in g_matrices_array])
        else:
            g_eigh_array = None
            g_det_array = None
        if g_eigh_array is not None:
            g_eigenvalues_array \
                = np.real(np.array([g_eigh_[0] for g_eigh_ in g_eigh_array]))
            g_eigenvectors_array \
                = np.real(np.array([g_eigh_[1] for g_eigh_ in g_eigh_array]))
        else:
            g_eigenvalues_array = None
            g_eigenvectors_array = None
        if do_eigenvectors and g_eigenvectors_array is not None:
            plt.plot(x_array, rz_array, '0.6', ls='-', lw=3, label=r'ray')
            plt.ylabel(r'Eigenvectors of $'+g_label+'$', fontsize=14)
            arrow_sf = 0.5
            my_arrow_style \
                = mpatches.ArrowStyle.Fancy(head_length=0.99*arrow_sf,
                                            head_width=0.6*arrow_sf,
                                            tail_width=0.01*arrow_sf)
            step = 8
            off = 0*step//2
            ev_sf = 0.04
            zipped_arrays = zip(x_array[off::step],
                                rz_array[off::step],
                                g_eigenvectors_array[off::step])
            for x_, rz_, evs_ in zipped_arrays:
                xy_ = np.array([x_, rz_])
                for pm in (-1, +1):
                    axes.annotate('', xy=xy_+pm*evs_[0]*ev_sf, xytext=xy_,
                                  arrowprops={'arrowstyle': my_arrow_style,
                                              'color': 'magenta'})
                    axes.annotate('', xy=xy_+pm*evs_[1]*ev_sf, xytext=xy_,
                                  arrowprops={'arrowstyle': my_arrow_style,
                                              'color': 'DarkGreen'})
            plt.plot(0, 0, 'DarkGreen', ls='-', lw=1.5, label='eigenvector 0')
            plt.plot(0, 0, 'magenta', ls='-', lw=1.5, label=r'eigenvector 1')
            axes.set_aspect(1)
        elif do_det and g_det_array is not None:
            plt.plot(x_array, g_det_array, 'DarkBlue', ls='-', lw=1.5,
                     label=r'$\det('+g_label+')$')
            plt.ylabel(r'Det of $'+g_label
                       + '$ (Hessian of $'+h_label+'$)', fontsize=14)
        elif do_pv:
            px_array = gmes.px_interp(x_array)
            pz_array = gmes.pz_interp(x_array)
            pv_array = px_array*vx_array + pz_array*vz_array
            plt.plot(x_array, pv_array, 'r', ls='-', lw=2, label=r'$p_i v^i$')
            if self.gstar_matrices_array is not None:
                gstarpp_array \
                    = [np.dot(
                        np.dot(gstar_, np.array([px_, pz_])),
                        np.array([px_, pz_])
                        )
                        for gstar_, px_, pz_
                        in zip(self.gstar_matrices_array, px_array, pz_array)]
                plt.plot(x_array, gstarpp_array, '0.5', ls='--', lw=3,
                         label=r'$g^j p_j p_j$')
            if self.g_matrices_array is not None:
                gvv_array \
                    = [np.dot(
                        np.dot(g_, np.array([vx_, vz_])),
                        np.array([vx_, vz_])
                        )
                       for g_, vx_, vz_
                       in zip(self.g_matrices_array, vx_array, vz_array)]
                plt.plot(x_array, gvv_array, 'k', ls=':',
                         lw=4, label=r'$g_i v^iv^i$')
            plt.ylabel(r'Inner product of $\mathbf{\widetilde{p}}$'
                       + r' and $\mathbf{{v}}$',
                       fontsize=14)
            legend_loc = 'upper left'
        elif g_eigenvalues_array is not None:
            (sign_ev0, label_ev0) \
                = (-1, 'negative  ') if g_eigenvalues_array[0, 0] < 0 \
                else (1, 'positive  ')
            (sign_ev1, label_ev1) \
                = (-1, 'negative  ') if g_eigenvalues_array[0, 1] < 0 \
                else (1, 'positive  ')
            plt.yscale('log')
            plt.plot(x_array,
                     sign_ev1*(g_eigenvalues_array[:, 1]),
                     'DarkGreen',
                     ls='-',
                     lw=1.5,
                     label=f'{label_ev1}'+rf'$\lambda_{g_label}(1)$')
            plt.plot(x_array,
                     sign_ev0*(g_eigenvalues_array[:, 0]),
                     'magenta',
                     ls='-',
                     lw=1.5,
                     label=f'{label_ev0}'+rf'$\lambda_{g_label}(0)$')
            plt.ylabel(f'Eigenvalues of {m_label} tensor '+rf'${g_label}$',
                       fontsize=12)
        else:
            return

        if do_eigenvectors:
            axes.set_ylim(*y_limits)
        elif do_det:
            ylim = plt.ylim()
            if ylim[1] < 0:
                axes.set_ylim(ylim[0], 0)
            if ylim[0] > 0:
                axes.set_ylim(0, ylim[1])
            # axes.set_ylim( -(ylim[1]-ylim[0])/20,ylim[1] )
        elif do_pv:
            axes.set_ylim(0, 2)
        plt.grid(True, ls=':')
        plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]', fontsize=14)
        # axes.set_ylim(ylim[0]*1.1,-0)
        plt.legend(loc=legend_loc, fontsize=12, framealpha=0.95)
        if do_etaxi_label:
            plt.text(*eta_label_xy,
                     rf'$\eta={gmeq.eta_}$'
                     + r'$\quad\mathsf{Ci}=$'
                     + rf'${round(float(deg(Ci.subs(sub))))}\degree$',
                     transform=axes.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, color='k')


#

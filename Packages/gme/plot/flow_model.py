"""
---------------------------------------------------------------------

Visualization of flow model.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`numpy`
  -  :mod:`matplotlib`
  -  :mod:`mpl_toolkits`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix: https://docs.sympy.org/latest/modules/matrices\
/immutablematrices.html

---------------------------------------------------------------------

"""
# pylint: disable = too-few-public-methods, no-self-use

# Library
import warnings

# Typing
from typing import Dict, Tuple, Optional

# Numpy
import numpy as np

# MatPlotLib
import matplotlib.pyplot as plt

# GME
from gme.core.symbols import rx, x_h, Lc
from gme.core.equations import Equations
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['FlowModel']


class FlowModel(Graphing):
    """
    Visualization of flow model.

    Extends :class:`gme.plot.base.Graphing`.
    """

    def profile_flow_model(
            self,
            gmeq: Equations,
            sub: Dict,
            name: str,
            fig_size: Optional[Tuple[float, float]] = None,
            dpi: Optional[int] = None,
            n_points: int = 26,
            subtitle: str = '',
            do_subtitling: bool = False,
            do_extra_annotations: bool = False
            ) -> None:
        """
        Plot the flow component of the erosion model.

        Args:
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be
                used for equation substitutions
            name:
                name of figure (key in figure dictionary)
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            n_points:
                optional number of points to plot along curve
            subtitle:
                optional sub-title (likely 'ramp' or 'ramp-flat' or similar)
            do_subtitling:
                optionally annotate with eta value etc
            do_extra_annotations:
                optionally annotate with labels
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        x_array = np.linspace(0, float(Lc.subs(sub)), n_points)
        varphi_array = [gmeq.varphi_rx_eqn.rhs.subs(
            sub).subs({rx: x_}) for x_ in x_array]
        varphi_xh1p0_array = [
            gmeq.varphi_rx_eqn.rhs.subs({x_h: 1}).subs(sub).subs({rx: x_})
            for x_ in x_array
        ]
        plt.plot(x_array, varphi_array, '-', color=self.colors[0],
                 label='hillslope-channel model')
        plt.plot(x_array, varphi_xh1p0_array, '--', color=self.colors[0],
                 label='channel-only model')

        # axes.set_aspect(1)
        plt.grid(True, ls=':')
        plt.xlabel(
            r'Dimensionless horizontal distance, $x/L_{\mathrm{c}}$  [-]')
        plt.ylabel(r'$\varphi(x)$  [-]')
        if do_subtitling:
            plt.text(0.1, 0.15, rf'$\eta={gmeq.eta_}$',
                     transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=12, color='k')
            plt.text(0.05, 0.22, subtitle,
                     transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=12, color='k')
        if do_extra_annotations:
            plt.text(0.4, 0.45, 'channel',
                     transform=axes.transAxes,
                     rotation=-43,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, color='0.2')
            plt.text(0.83, 0.16, 'hillslope',
                     transform=axes.transAxes,
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=11, color='0.2')

        y_limits: Tuple[float, float] = axes.get_ylim()
        x_h_: float = float(x_h.subs(sub))
        varphi_h_: float = float(
            gmeq.varphi_rx_eqn.rhs.subs({rx: x_h}).subs(sub))
        plt.plot([x_h_, x_h_], [varphi_h_-30, varphi_h_+70], 'b:')
        plt.text(x_h_, varphi_h_+77, r'$x_h/L_{\mathrm{c}}$',
                 # transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=12, color='b')

        plt.legend(loc='upper right', fontsize=11, framealpha=0.95)
        plt.xlim(None, 1.05)
        plt.ylim(*y_limits)


#

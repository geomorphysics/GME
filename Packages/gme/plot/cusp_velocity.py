"""
---------------------------------------------------------------------

Visualization of velocity components of surface cusp.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`NumPy <numpy>`
  -  :mod:`MatPlotLib <matplotlib>`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html


---------------------------------------------------------------------

"""
# Library
import warnings

# Typing
from typing import Dict, Tuple, Optional

# Numpy
import numpy as np

# MatPlotLib
import matplotlib.pyplot as plt

# GME
from gme.core.equations import Equations
from gme.ode.velocity_boundary import VelocityBoundarySolution
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['CuspVelocity']


class CuspVelocity(Graphing):
    """
    Visualization of velocity components of surface cusp.

    Extends :class:`gme.plot.base.Graphing`.
    """

    def profile_cusp_horizontal_speed(
        self,
        gmes: VelocityBoundarySolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        x_limits: Tuple[float, float] = (-0.05, 1.05),
        y_limits: Tuple[float, Optional[float]] = (-5, None),
        t_limits: Tuple[float, Optional[float]] = (0, None),
        legend_loc: str = 'lower right',
        do_x: bool = True,
        do_infer_initiation: bool = True
    ) -> None:
        r"""
        Plot horizontal speed of cusp propagation

        Args:
            gmes:
                instance of velocity boundary solution class defined in
                :mod:`gme.ode.velocity_boundary`
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            sub:
                dictionary of model parameter values to be used for
                equation substitutions
            name:
                name of plot in figures dictionary
            fig_size:
                optional figure width and height in inches
            dpi:
                optional rasterization resolution
            x_limits:
                optional [x_min, x_max] horizontal plot range
            y_limits:
                optional [z_min, z_max] vertical plot range
            t_limits:
                optional [t_min, t_max] time range
            legend_loc:
                where to plot the legend
            do_x:
                optional plot x-axis as dimensionless horizontal distance
                :math:`x/L_{\mathrm{c}}`;
                otherwise plot as time :math:`t`
            do_infer_initiation:
                optional draw dotted line inferring cusp initiation at the left
                boundary

        Todo:
            implement `do_infer_initiation`
        """
        del sub
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

        # Drop last cusp because it's sometimes bogus
        # _ =  gmes.trxz_cusps
        x_or_t_array \
            = gmes.cusps['rxz'][:-1][:, 0] if do_x else gmes.cusps['t'][:-1]
        vc_array = np.array(
            [(x1_-x0_)/(t1_-t0_) for (x0_, z0), (x1_, z1), t0_, t1_
             in zip(gmes.cusps['rxz'][:-1], gmes.cusps['rxz'][1:],
                    gmes.cusps['t'][:-1], gmes.cusps['t'][1:])]
        )

        plt.plot(x_or_t_array, vc_array, '.', ms=7, label='measured')

        if do_x:
            plt.xlabel(r'Distance, $x/L_{\mathrm{c}}$  [-]')
        else:
            plt.xlabel(r'Time, $t$')
        plt.ylabel(r'Cusp horiz propagation speed,  $c^x$')

        axes = plt.gca()
        plt.text(0.15, 0.2, rf'$\eta={gmeq.eta_}$', transform=axes.transAxes,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, color='k')

        x_array \
            = np.linspace(0.001 if do_infer_initiation else x_or_t_array[0],
                          1,
                          num=101)
        # color_cx, color_bounds = 'DarkGreen', 'Green'
        color_cx, color_bounds = 'Red', 'DarkRed'
        plt.plot(x_array,
                 gmes.cx_pz_lambda(x_array),
                 color=color_cx,
                 alpha=0.8,
                 lw=2,
                 label=r'$c^x$ model ($p_z$)')
        plt.plot(x_array,
                 gmes.cx_v_lambda(x_array),
                 ':',
                 color='k',
                 alpha=0.8,
                 lw=2,
                 label=r'$c^x$ model ($\mathbf{v}$)')
        plt.plot(x_array,
                 gmes.vx_interp_fast(x_array),
                 '--',
                 color=color_bounds,
                 alpha=0.8,
                 lw=1,
                 label=r'fast ray $v^x$ bound')
        plt.plot(x_array,
                 gmes.vx_interp_slow(x_array),
                 '-.',
                 color=color_bounds,
                 alpha=0.8,
                 lw=1,
                 label=r'slow ray $v^x$ bound')

        _ = plt.xlim(*x_limits) if do_x else plt.xlim(*t_limits)
        _ = plt.ylim(*y_limits) if y_limits is not None else None
        plt.grid(True, ls=':')

        plt.legend(loc=legend_loc, fontsize=12, framealpha=0.95)


#

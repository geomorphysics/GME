"""
---------------------------------------------------------------------

GME visualization base class.

---------------------------------------------------------------------

Requires Python packages:
  -  :mod:`NumPy <numpy>`
  -  :mod:`MatPlotLib <matplotlib>`
  -  `GMPLib`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html


---------------------------------------------------------------------

"""
# Library
import warnings
# import logging
from typing import List, Tuple, Dict, Optional

# NumPy
import numpy as np

# MatPlotLib
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import cm

# GMPLib
from gmplib.plot import GraphingBase

# GME
# from gme.core.symbols import xih_0, xiv_0, t

warnings.filterwarnings("ignore")

__all__ = ['Graphing']


class Graphing(GraphingBase):
    """
    GME visualization base class.

    Subclasses :class:`gmplib.plot.GraphingBase>`
    """

    def mycolors(
        self,
        i: int,
        r: int,
        n: int,
        do_smooth: bool = False,
        cmap_choice: str = 'brg'
    ) -> List[str]:
        r"""
        Generate a color palette
        """
        if not do_smooth:
            colors_ = self.colors[(i//r) % self.n_colors]
        else:
            cmap = cm.get_cmap(cmap_choice)
            colors_ = cmap(i/(n-1))
        return colors_

    def gray_color(
        self,
        i_isochrone: int = 0,
        n_isochrones: int = 1
    ) -> str:
        r"""
        Make a grey shade for to communicate isochrone time
        """
        return f'{(n_isochrones-1-i_isochrone)/(n_isochrones-1)*0.75}'

    def correct_quadrant(self, angle: float) -> float:
        r"""
        If angle :math:`|\theta|\approx 0`, set :math:`\theta=0`;
        otherwise, if angle :math:`\theta<0`,
        map :math:`\theta \rightarrow \pi-\theta`.

        Args:
            angle (float): angle in radians

        Returns:
            Modified value of angle.
        """
        if abs(angle) <= 1e-10:
            angle_ = 0.0
        elif angle > 0.0:
            angle_ = angle
        else:
            angle_ = np.pi+angle
        return angle_

    def draw_rays_with_arrows_simple(
        self,
        axes: Axes,
        sub: Dict,
        xi_vh_ratio: float,
        t_array: np.ndarray,
        rx_array: np.ndarray,
        rz_array: np.ndarray,
        v_array: Optional[np.ndarray] = None,
        n_t: Optional[int] = None,
        n_rays: int = 4,
        ls: str = '-',
        sf: float = 1,
        color: Optional[str] = None,
        do_labels: bool = True,
        do_one_ray: bool = False
    ) -> None:
        """
        Plot ray and arrowheads along the ray to visualize the
        direction of motion.

        Args:
            axes:
                'axes' instance for current figure
            sub:
                dictionary of model parameter values to be used for
                equation substitutions
            t_array:
                sample times along the ray
            rx_array:
                x coordinates along the sampled ray
            rz_array:
                z coordinates along the sampled ray
            ls:
                optional line style
        """
        del sub
        i_max = len(t_array) if n_t is None else n_t
        i_step = i_max//n_rays
        i_off = i_step*(1+i_max//i_step)-i_max + 1
        my_arrow_style \
            = mpatches.ArrowStyle.Fancy(head_length=.99*sf,
                                        head_width=.6*sf,
                                        tail_width=0.01*sf)
        if v_array is not None:
            v_max, v_min = max(v_array), min(v_array)
            color_map: ListedColormap = plt.get_cmap('plasma')
        t_ref = t_array[i_max-1]
        for i in range(i_max-1, 0, -1):
            color = color if do_one_ray else self.colors[(
                i//i_step) % self.n_colors]
            if (i+i_off)//i_step == (i+i_off)/i_step:
                t_offset = 0 if do_one_ray else t_array[i]*xi_vh_ratio
                t_label = r'$\hat{t}_0=$'+f'{round(t_ref-t_array[i],1)}'
                # $t_0={}$'.format(i_max-i-1)
                plt.plot(rx_array[:i+1], rz_array[:i+1]-t_offset, ls,
                         label=t_label if do_labels else None,
                         color=color)
            # if (i+i_off)//i_step==(i+i_off)/i_step:
                for q in range(1, i-1, 3):
                    if do_one_ray and v_array is not None:
                        v_rel = ((v_array[q]-v_min)/(v_max-v_min))**0.5
                        rgba = color_map(v_rel*0.8)
                    else:
                        rgba = color
                    axes.annotate('',
                                  xy=(rx_array[q+1],
                                      rz_array[q+1]-t_offset),
                                  xytext=(rx_array[q],
                                          rz_array[q]-t_offset),
                                  arrowprops=dict(arrowstyle=my_arrow_style,
                                                  color=rgba))
        if v_array is not None:
            color_map = plt.get_cmap('plasma')
            sm = plt.cm.ScalarMappable(
                cmap=color_map, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = plt.colorbar(sm, ticks=[], shrink=0.4, aspect=5, pad=0.03)
            labels: Tuple = (r'$v_\mathrm{min}$', r'$v_\mathrm{max}$')
            for idx, tick_label in enumerate(labels):
                cbar.ax.text(0.45, idx*1.3-0.15, tick_label,
                             ha='center', va='center')
            # cbar.ax.get_yaxis().labelpad = -100
            # cbar.ax.set_ylabel(r'ray speed  $v$', rotation=90)

    def arrow_annotate_ray_custom(
        self,
        rx_array: np.ndarray,
        rz_array: np.ndarray,
        axes: Axes,
        i_ray: int,
        i_ray_step: int,
        n_rays: int,
        n_arrows: int,
        arrow_sf: float = 0.7,
        arrow_offset: int = 1,
        x_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        y_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        line_style: str = 'dashed',
        line_width: float = 1,
        ray_label: str = None,
        do_smooth_colors: bool = False
    ) -> None:
        """
        Add arrowheads to a ray trajectory to visualize
        the direction of motion.

        Args:
            rx_array:
                x coordinates along the sampled ray
            rz_array:
                z coordinates along the sampled ray
            axes:
                'axes' instance for current figure
            sub:
                dictionary of model parameter values to be used
                for equation substitutions
            i_ray:
                index of this ray among the set currently being plotted
            i_ray_step:
                ray index step, used as divisor when assigning
                a color to match the parent ray color
            n_arrows:
                number of arrows to plot along the ray
            arrow_sf:
                optional scale factor for arrowhead size
            arrow_offset:
                optional offset from ray initial point
                to start adding arrowheads
            x_limits:
                optional horizontal axis range
            y_limits:
                optional vertical axis range
            line_style:
                optional line style
            line_width:
                optional line width
            ray_label:
                optional ray label for legend
        """
        # Drop repeated points on vb
        rxz_array = np.unique(np.vstack([rx_array, rz_array]), axis=1)
        n_pts: int = rxz_array.shape[1]
        my_arrow_style \
            = mpatches.ArrowStyle.Fancy(head_length=0.99*arrow_sf,
                                        head_width=0.6*arrow_sf,
                                        tail_width=0.01*arrow_sf)
        # color = self.colors[(i_ray//i_ray_step)%self.n_colors]
        color: List[str] = self.mycolors(i_ray, i_ray_step, n_rays,
                                         do_smooth=do_smooth_colors)
        q_n: int = n_pts//n_arrows
        if q_n > 0:
            q_from: int = n_pts//(n_arrows)//arrow_offset+n_pts//(n_arrows)
            q_to: int = n_pts-1
            for q in range(q_from, q_to, q_n):
                y_condition: bool \
                    = (y_limits is None or (
                        (rxz_array[1][q+1] > y_limits[0]
                            and rxz_array[1][q+1] < y_limits[1])
                        and (rxz_array[1][q] > y_limits[0]
                             and rxz_array[1][q] < y_limits[1])
                    ))
                x_condition: bool \
                    = (x_limits is None or (
                        (rxz_array[0][q+1] > x_limits[0]
                            and rxz_array[0][q+1] < x_limits[1])
                        and (rxz_array[0][q] > x_limits[0]
                             and rxz_array[0][q] < x_limits[1])
                    ))
                if y_condition and x_condition:
                    arrowprops_ = dict(arrowstyle=my_arrow_style, color=color)
                    axes.annotate('',
                                  xy=(rxz_array[0][q+1], rxz_array[1][q+1]),
                                  xytext=(rxz_array[0][q], rxz_array[1][q]-0),
                                  arrowprops=arrowprops_)
        plt.plot(rxz_array[0],
                 rxz_array[1],
                 color=color,
                 linestyle=line_style,
                 lw=line_width,
                 label=ray_label)

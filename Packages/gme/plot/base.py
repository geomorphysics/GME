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
  -  :mod:`numpy`
  -  :mod:`matplotlib.pyplot`, :mod:`matplotlib.patches`, :mod:`matplotlib.cm`
  -  :mod:`gmplib.plot_utils`
  -  :mod:`gme.core.symbols`

---------------------------------------------------------------------

"""
import warnings
import logging

# Typing
from typing import List

# Numpy
import numpy as np

# MatPlotLib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

# GMPLib
from gmplib.plot_utils import GraphingBase

# GME
from gme.core.symbols import xih_0, xiv_0, t

warnings.filterwarnings("ignore")

__all__ = ['Graphing']


class Graphing(GraphingBase):
    """
    Subclasses :class:`gmplib.plot_utils.GraphingBase <plot_utils.GraphingBase>`
    """
    def mycolors(self, i, r, n, do_smooth=False, cmap_choice='brg') -> List[str]:
        r"""
        Generate a color palette
        """
        if not do_smooth:
            colors_ = self.colors[(i//r)%self.n_colors]
        else:
            cmap = cm.get_cmap(cmap_choice)
            colors_ = cmap(i/(n-1))
        return colors_

    @staticmethod
    def gray_color(i_isochrone=0, n_isochrones=1) -> str:
        r"""
        Make a grey shade for to communicate isochrone time
        """
        return f'{(n_isochrones-1-i_isochrone)/(n_isochrones-1)*0.75}'

    @staticmethod
    def correct_quadrant(angle) -> float:
        r"""
        If angle :math:`|\theta|\approx 0`, set :math:`\theta=0`;
        otherwise, if angle :math:`\theta<0`, map :math:`\theta \rightarrow \pi-\theta`.

        Args:
            angle (float): angle in radians

        Returns:
            Modified value of angle.
        """
        if abs(angle)<=1e-10:
            angle_ = 0.0
        elif angle>0.0:
            angle_ =  angle
        else:
            angle_ = np.pi+angle
        return angle_

    def draw_rays_with_arrows_simple( self, axes, sub, xi_vh_ratio,
                                      t_array, rx_array, rz_array, v_array=None,
                                      n_t=None, n_rays=4,
                                      ls='-', sf=1, color=None,
                                      do_labels=True, do_one_ray=False ) -> None:
        """
        Plot ray and arrowheads along the ray to visualize the direction of motion.

        Args:
            axes (:class:`Matplotlib axes <matplotlib.axes.Axes>`): 'axes' instance
                                        for current figure
            sub (dict): dictionary of model parameter values to be used for
                        equation substitutions
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
                t_offset = 0 if do_one_ray else t_array[i]*xi_vh_ratio
                t_label = r'$\hat{t}_0=$'+f'{round(t_ref-t_array[i],1)}'
                #$t_0={}$'.format(i_max-i-1)
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

    def arrow_annotate_ray_custom( self, rx_array, rz_array, axes, i_ray, i_ray_step,
                                   n_rays, n_arrows,
                                   arrow_sf=0.7, arrow_offset=1,
                                   x_limits=None, y_limits=None,
                                   line_style='dashed', line_width=1, ray_label=None,
                                   do_smooth_colors=False )  -> None:
        """
        Add arrowheads to a ray trajectory to visualize the direction of motion.

        Args:
            rx_array (numpy.ndarray): x coordinates along the sampled ray
            rz_array (numpy.ndarray): z coordinates along the sampled ray
            axes (:class:`Matplotlib axes <matplotlib.axes.Axes>`): 'axes' instance
                for current figure
            sub (dict): dictionary of model parameter values to be used
                for equation substitutions
            i_ray (int): index of this ray among the set currently being plotted
            i_ray_step (int): ray index step, used as divisor when assigning
                a color to match the parent ray color
            n_arrows (int): number of arrows to plot along the ray
            arrow_sf (float): optional scale factor for arrowhead size
            arrow_offset (int): optional offset from ray initial point
                to start adding arrowheads
            x_limits (list): optional horizontal axis range
            y_limits (list): optional vertical axis range
            line_style (str): optional line style
            line_width (str): optional line width
            ray_label (str): optional ray label for legend
        """
        # Drop repeated points on vb
        rxz_array = np.unique( np.vstack([rx_array,rz_array]), axis=1 )
        n_pts: int = rxz_array.shape[1]
        my_arrow_style = mpatches.ArrowStyle.Fancy(head_length=0.99*arrow_sf,
                                                   head_width=0.6*arrow_sf,
                                                   tail_width=0.01*arrow_sf)
        # color = self.colors[(i_ray//i_ray_step)%self.n_colors]
        color: List[str] = self.mycolors(i_ray, i_ray_step, n_rays,
                                         do_smooth=do_smooth_colors)
        q_n: int = n_pts//n_arrows
        if q_n>0:
            q_from: int = n_pts//(n_arrows)//arrow_offset+n_pts//(n_arrows)
            q_to: int = n_pts-1
            for q in range(q_from, q_to, q_n):
                y_condition: bool = (y_limits is None or (
                        (rxz_array[1][q+1]>y_limits[0] and rxz_array[1][q+1]<y_limits[1])
                        and (rxz_array[1][q]>y_limits[0] and rxz_array[1][q]<y_limits[1])
                    )
                )
                x_condition: bool = (x_limits is None or (
                        (rxz_array[0][q+1]>x_limits[0] and rxz_array[0][q+1]<x_limits[1])
                        and (rxz_array[0][q]>x_limits[0] and rxz_array[0][q]<x_limits[1])
                    )
                )
                if y_condition and x_condition:
                    axes.annotate('', xy=((rxz_array[0][q+1]),(rxz_array[1][q+1])),
                                  xytext=(rxz_array[0][q], rxz_array[1][q]-0),
                                  arrowprops=dict(arrowstyle=my_arrow_style, color=color) )
        plt.plot(rxz_array[0],rxz_array[1], color=color,
                  linestyle=line_style, lw=line_width, label=ray_label)

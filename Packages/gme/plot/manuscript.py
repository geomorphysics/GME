"""
---------------------------------------------------------------------

Generate plots for publications.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`numpy`
  -  :mod:`sympy`
  -  :mod:`matplotlib`
  -  :mod:`mpl_toolkits`
  -  :mod:`gme`

---------------------------------------------------------------------

"""
import warnings

# Typing
from typing import Tuple, List

# Numpy
import numpy as np

# SymPy
from sympy import N, lambdify, re

# MatPlotLib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ArrowStyle, ConnectionPatch, Arc
from matplotlib.spines import Spine
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# GME
from gme.core.symbols import rx, varphi, pz
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['Manuscript']


class Manuscript(Graphing):
    r"""
    Generate plots for publications.

    Subclasses :class:`gme.plot.base.Graphing`.
    """
    def point_pairing(self, name, fig_size=(10,4), dpi=None) -> None:
        """
        Schematic illustrating ...

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by
                :meth:`GMPLib create_figure <plot.GraphingBase.create_figure>`
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

        def remove_ticks_etc(axes_) -> None:
            r"""
            TBD
            """
            axes_.set_xticklabels([])
            axes_.set_xticks([])
            axes_.set_yticklabels([])
            axes_.set_yticks([])
            axes_.set_xlim(0,1)
            axes_.set_ylim(0,1)

        def linking_lines(fig_, axes_A, axes_B, axes_C, axes_D, color_=brown_) -> None:
            r"""
            TBD
            """
            joins = [0]*3
            kwargs = dict(color=color_, linestyle=':')
            joins[0] = ConnectionPatch(xyA=(0.2,0), coordsA=axes_D.transData,
                                       xyB=(0.4,1),  coordsB=axes_A.transData, **kwargs)
            joins[1] = ConnectionPatch(xyA=(1,0.00), coordsA=axes_D.transData,
                                       xyB=(0,0.9),  coordsB=axes_B.transData, **kwargs)
            joins[2] = ConnectionPatch(xyA=(1,0.60), coordsA=axes_D.transData,
                                       xyB=(0,0.8),  coordsB=axes_C.transData, **kwargs)
            for join_ in joins: fig_.add_artist(join_)

        def make_xy() -> Tuple[float,float]:
            r"""
            TBD
            """
            x = np.linspace(0,1)
            x_ndim = (x-0.5)/(0.9-0.5)
            y = np.exp((0.5+x)*4)/120
            return x_ndim,y

        def isochrones_subfig(fig_, x_, y_, color_=gray_) \
                    -> Tuple[mpl.axes._axes.Axes,List[float]]:
            r"""
            TBD
            """
            # Top left isochrones 0
            size_zoom_0 = [0.65, 0.55]
            posn_0 = [0.0, 0.75]
            axes_0 = fig_.add_axes([*posn_0, *size_zoom_0])
            plt.axis('off')
            n_isochrones = 6
            for i_, sf_ in enumerate(np.linspace(0.5,1.2,n_isochrones)):
                plt.plot(x_, sf_*y_, '-', color=self.gray_color(i_,n_gray), lw=2.5)
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
                            arrowprops=dict(arrowstyle=my_arrow_style,
                                            color=color_, lw=3))
            plt.text(*motion_xy_, 'motion', color=color_, fontsize=16,
                     rotation=motion_angle_,
                     transform=axes_0.transAxes,
                     horizontalalignment='center', verticalalignment='center')
            return axes_0, posn_0

        def set_colors(obj_type, axes_list, color_) -> None:
            r"""
            TBD
            """
            for axes_ in axes_list:
                _ = [child.set_color(color_) for child in axes_.get_children()
                     if isinstance(child, obj_type)]

        def zoom_boxes(fig_, ta_color_=gray2_, tb_color_=gray1_) \
                            -> Tuple[mpl.axes._axes.Axes, mpl.axes._axes.Axes,
                                     mpl.axes._axes.Axes, mpl.axes._axes.Axes, str]:
            r"""
            TBD
            """
            size_zoom_AB = [0.3,0.7]
            size_zoom_C = [0.3,0.7]
            n_pts = 300

            def zoomed_isochrones(axes_, name_text, i_pt1_, i_pt2_, do_many=False,
                                  do_legend=False, do_pts_only=False) \
                                            -> Tuple[np.array,np.array,np.array]:
                x_array = np.linspace(-1,3, n_pts)
                y_array1 = x_array*0.5
                y_array2 = y_array1+0.7

                # Ta
                if not do_pts_only:
                    plt.plot(x_array, y_array2, '-', color=ta_color_, lw=2.5,
                             label=r'$T_a$')
                xy_pt2 = np.array([x_array[i_pt2_],y_array2[i_pt2_]])
                marker_style2 = dict(marker='o', fillstyle='none',
                                     markersize=8,
                                     markeredgecolor=ta_color_,
                                     markerfacecolor=ta_color_,
                                     markeredgewidth=2 )
                plt.plot(*xy_pt2, **marker_style2)
                if not do_pts_only:
                    plt.text(*(xy_pt2+np.array([-0.03,0.08])), r'$\mathbf{a}$',
                             color=ta_color_, fontsize=18, rotation=0,
                             transform=axes_.transAxes,
                             horizontalalignment='center', verticalalignment='center')
                # Tb
                if not do_pts_only:
                    plt.plot(x_array, y_array1, '-', color=tb_color_, lw=2.5,
                             label=r'$T_b = T_a+\Delta{T}$')
                i_pts1 = [i_pt1_]
                xy_pts1_tmp = np.array([np.array([x_array[i_pt1__],y_array1[i_pt1__]])
                                    for i_pt1__ in i_pts1])
                xy_pts1 = xy_pts1_tmp.T.reshape((xy_pts1_tmp.shape[2],2)) if do_many \
                          else xy_pts1_tmp
                marker_style1 = marker_style2.copy()
                marker_style1.update({'markeredgecolor':tb_color_,'markerfacecolor':'w'})
                for xy_pt1_ in xy_pts1: plt.plot(*xy_pt1_, **marker_style1)

                if not do_pts_only:
                    b_label_i = 4 if do_many else 0
                    b_label_xy = (xy_pts1[b_label_i]+np.array([0.03,-0.08]))
                    b_label_text = r'$\{\mathbf{b}\}$' if do_many else r'$\mathbf{b}$'
                    plt.text(*b_label_xy, b_label_text,
                             color=tb_color_, fontsize=18, rotation=0,
                             transform=axes_.transAxes,
                             horizontalalignment='center', verticalalignment='center')
                    if do_legend:
                        plt.legend(loc=[0.05,-0.35], fontsize=16, framealpha=0)
                    name_xy = [0.97,0.03]
                    plt.text(*name_xy, name_text,
                             color=brown_, fontsize=14, rotation=0,
                             transform=axes_.transAxes,
                             horizontalalignment='right', verticalalignment='bottom')

                dx = x_array[1]-x_array[0]
                dy = y_array1[1]-y_array1[0]
                return (xy_pts1 if do_many else xy_pts1[0]), xy_pt2, np.array([dx,dy])

            def v_arrow(axes_, xy_pt1, xy_pt2, dxy=None, a_f=0.54, v_f=0.5,
                        v_label=r'$\mathbf{v}$',
                        color_=red_, do_dashing=False, do_label=False) -> None:
                v_lw = 1.5
                axes_.arrow(*((xy_pt1*a_f+xy_pt2*(1-a_f))),*((xy_pt1-xy_pt2)*0.01),
                             lw=1, facecolor=color_, edgecolor=color_,
                             head_width=0.05, overhang=0.3,
                             transform=axes_.transAxes, length_includes_head=True, )
                axes_.plot([xy_pt1[0],xy_pt2[0]], [xy_pt1[1],xy_pt2[1]],
                            ls='--' if do_dashing else '-',
                            lw=v_lw, color=color_ )
                f = v_f
                v_xy = (xy_pt1[0]*f + xy_pt2[0]*(1-f)+dxy[0],
                        xy_pt1[1]*f + xy_pt2[1]*(1-f)+dxy[1])
                if do_label:
                    axes_.text(*v_xy, v_label,
                                color=color_, fontsize=18, rotation=0,
                                transform=axes_.transAxes,
                                horizontalalignment='right', verticalalignment='bottom')

            def p_bones(axes_, xy_pt1, xy_pt2, dxy, p_f=0.9, color_=blue_,
                        n_bones=5, do_primary=True) -> None:
                alpha_ = 0.7
                p_dashing = [1,0] #[4, 4]
                p_lw = 3
                axes_.plot([xy_pt1[0],xy_pt2[0]], [xy_pt1[1],xy_pt2[1]],
                           dashes=p_dashing, lw=p_lw, color=color_,
                           alpha=1 if do_primary else alpha_ )
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
                p_xy = (xy_pt1[0]*f + xy_pt2[0]*(1-f)-0.1,
                        xy_pt1[1]*f + xy_pt2[1]*(1-f)-0.1)
                if do_primary:
                    axes_.text(*p_xy, r'$\mathbf{\widetilde{p}}$',
                                color=color_, fontsize=18, rotation=0,
                                transform=axes_.transAxes,
                                horizontalalignment='right', verticalalignment='bottom')

            def psi_label(axes_, xy_pt1_B, xy_pt2_B, xy_pt1_C, xy_pt2_C,
                          color_=red_) -> None:
                label_xy = [0.5,0.53]
                axes_.text(*label_xy, r'$\psi$',
                            color=color_, fontsize=20, rotation=0,
                            transform=axes_.transAxes,
                            horizontalalignment='center', verticalalignment='center')
                angle_B = np.rad2deg(np.arctan(
                                (xy_pt2_B[1]-xy_pt1_B[1])/(xy_pt2_B[0]-xy_pt1_B[0]) ))
                angle_C = np.rad2deg(np.arctan(
                                (xy_pt2_C[1]-xy_pt1_C[1])/(xy_pt2_C[0]-xy_pt1_C[0]) ))
                radius = 0.95
                axes_.add_patch( Arc(xy_pt2_B, radius,radius, color=color_,
                                    linewidth=2, fill=False, zorder=2,
                                    #transform=axes_.transAxes,
                                    angle=angle_B, theta1=0, theta2=angle_C-angle_B) )

            def beta_label(axes_, xy_pt1_B, xy_pt2_B, color_=blue_) -> None:
                label_xy = [0.28,0.47]
                axes_.text(*label_xy, r'$\beta$',
                            color=color_, fontsize=20, rotation=0,
                            transform=axes_.transAxes,
                            horizontalalignment='center', verticalalignment='center')
                angle_ref = -90
                angle_B = np.rad2deg(np.arctan(
                                (xy_pt2_B[1]-xy_pt1_B[1])/(xy_pt2_B[0]-xy_pt1_B[0]) ))
                radius = 0.88
                axes_.add_patch( Arc(xy_pt2_B, radius,radius, color=color_, linestyle='--',
                                    linewidth=2, fill=False, zorder=2,
                                    #transform=axes_.transAxes,
                                    angle=angle_ref, theta1=0, theta2=angle_B-angle_ref) )
                axes_.plot( [xy_pt2_B[0],xy_pt2_B[0]], [xy_pt2_B[1],xy_pt2_B[1]-0.5],
                            ':', color=color_)

            def alpha_label(axes_, xy_pt1_B, xy_pt2_B, color_=red_) -> None:
                label_xy = [0.55,0.75]
                axes_.text(*label_xy, r'$-\alpha$',
                            color=color_, fontsize=20, rotation=0,
                            transform=axes_.transAxes,
                            horizontalalignment='center', verticalalignment='center')
                # angle_ref = 0
                angle_B = np.rad2deg(np.arctan(
                                (xy_pt2_B[1]-xy_pt1_B[1])/(xy_pt2_B[0]-xy_pt1_B[0]) ))
                radius = 0.88
                axes_.add_patch( Arc(xy_pt2_B, radius,radius, color=color_, linestyle='--',
                                    linewidth=2, fill=False, zorder=2,
                                    #transform=axes_.transAxes,
                                    angle=angle_B, theta1=0, theta2=-angle_B) )
                axes_.plot( [xy_pt2_B[0],xy_pt2_B[0]+0.5], [xy_pt2_B[1],xy_pt2_B[1]],
                            ':', color=color_)

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
            xy_pts1_A, xy_pt2_A, _ = zoomed_isochrones(axes_A, 'free', i_pts1_A,i_pt2_A,
                                                       do_many=True)
            for i_,xy_pt1_A in enumerate(xy_pts1_A):
                v_arrow(axes_A, xy_pt1_A, xy_pt2_A, dxy=[0.12,0.05], do_dashing=True,
                         v_f=0.35, v_label=r'$\{\mathbf{v}\}$',
                         do_label=bool(i_==len(xy_pts1_A)-1))
            zoomed_isochrones(axes_A, '', i_pts1_A,i_pt2_A, do_many=True)

            # Zoom intrinsic pairing B
            posn_B = [0.33,0.28]
            axes_B = fig_.add_axes([*posn_B, *size_zoom_AB])
            remove_ticks_etc(axes_B)
            i_pt2_B = i_pt2_A
            i_pt1_B = i_pt2_A+19
            xy_pt1_B, xy_pt2_B, dxy_B = zoomed_isochrones(axes_B, 'isotropic',
                                                          i_pt1_B,i_pt2_B)
            p_bones(axes_B, xy_pt1_B, xy_pt2_B, dxy_B)
            v_arrow(axes_B, xy_pt1_B, xy_pt2_B, v_f=0.5, dxy=[0.13,0.02], do_label=True)
            zoomed_isochrones(axes_B, '', i_pt1_B,i_pt2_B, do_pts_only=True)

            # Zoom erosion-fn pairing C
            posn_C = [0.66,0.6]
            axes_C = fig_.add_axes([*posn_C, *size_zoom_C])
            remove_ticks_etc(axes_C)
            i_pt1_C = i_pt1_B+30
            i_pt2_C = i_pt2_B
            xy_pt1_C, xy_pt2_C, dxy_C = zoomed_isochrones(axes_C, 'anisotropic',
                                                          i_pt1_C, i_pt2_C,
                                                          do_legend=True)
            p_bones(axes_C, xy_pt1_C, xy_pt2_C, dxy_C, do_primary=False)
            p_bones(axes_C, xy_pt1_B, xy_pt2_B, dxy_B)
            v_arrow(axes_C, xy_pt1_C, xy_pt2_C,
                    a_f=0.8, v_f=0.72, dxy=[0.1,0.05], do_label=True)
            zoomed_isochrones(axes_C, '', i_pt1_C, i_pt2_C,
                              do_legend=False, do_pts_only=True)
            psi_label(axes_C, xy_pt1_B, xy_pt2_B, xy_pt1_C, xy_pt2_C, color_=purple_)
            beta_label(axes_C, xy_pt1_B, xy_pt2_B)
            alpha_label(axes_C, xy_pt1_C, xy_pt2_C)

            # Brown zoom boxes and tie lines
            set_colors(Spine, [axes_A, axes_B, axes_C, axes_D], brown_)
            return axes_A, axes_B, axes_C, axes_D, brown_

        x,y = make_xy()
        _, posn_0 = isochrones_subfig(fig, x, y)
        axes_A, axes_B, axes_C, axes_D, _ = zoom_boxes(fig)
        linking_lines(fig, axes_A, axes_B, axes_C, axes_D)

    def covector_isochrones(self, name, fig_size=None, dpi=None) -> None:
        """
        Schematic illustrating relationship between normal erosion rate vector,
        normal slowness covector, isochrones, covector components,
        and vertical/normal erosion rates.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by
                :meth:`GMPLib create_figure <plot.GraphingBase.create_figure>`
        """
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)

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
        _,L_px_,L_pz_ = p_*nDt_,px_*nDt_,pz_*nDt_
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
            color_ = self.gray_color(i_isochrone=i_isochrone,
                                     n_isochrones=n_minor_isochrones)
            plt.plot(x_array+sf_*i_isochrone/np.sin(beta_), x_array*np.tan(beta_),
                     color=color_,
                     lw=2.5 if i_isochrone % ts_major_isochrones==0 else 0.75,
                     label=r'$T(\mathbf{r})$'+rf'$={i_isochrone}$y'
                            if i_isochrone % ts_major_isochrones==0 else None)

        # r vector
        af, naf = 0.7, 0.3
        r_length = 0.6 #1.5
        r_color = 'gray'
        plt.plot(origin_[0],origin_[1], 'o', color=r_color, ms=15)
        plt.text(origin_[0]-r_length*np.cos(0.5)*naf+0.04,
                 origin_[1]-r_length*np.sin(0.5)*naf-0.04,
                 r'$\mathbf{r}$',
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=18, color=r_color)

        # Slowness covector p thick transparent lines
        lw_pxz_ = 10
        alpha_p_ = 0.1
        px_array = np.linspace(origin_[0],origin_[0]+L_px_*4,2)
        pz_array = np.linspace(origin_[1],origin_[1]+L_pz_*4,2)
        plt.plot(px_array,pz_array, lw=lw_pxz_,
                    alpha=alpha_p_, color=p_color, solid_capstyle='butt')
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

        plt.arrow(px_array[0], pz_array[0],
                    (px_array[0]-px_array[1])/22,(pz_array[0]-pz_array[1])/22,
                    head_width=0.16, head_length=-0.1, lw=2.5,
                    shape='full', overhang=1,
                    length_includes_head=True,
                    head_starts_at_zero=True,
                    ec='b', fc='b')

        # Coordinate axes
        x_off, y_off = 0,-0.4
        my_arrow_style \
            = mpatches.ArrowStyle.Fancy(head_length=.99, head_width=.6, tail_width=0.01)
        axes.annotate('', xy=(x_limit/10-0.05+x_off,y_limit*0.7+y_off),
                        xytext=(x_limit/10-0.275+x_off,y_limit*0.7+y_off),
                        arrowprops=dict(arrowstyle=my_arrow_style, color='k') )
        axes.annotate('', xy=(x_limit/10-0.265+x_off,y_limit*0.7+0.2+y_off),
                        xytext=(x_limit/10-0.265+x_off,y_limit*0.7-0.01+y_off),
                        arrowprops=dict(arrowstyle=my_arrow_style, color='k') )
        plt.text(x_limit/10-0.05+x_off,y_limit*0.7+y_off,'$x$',
                    horizontalalignment='left', verticalalignment='center',
                    fontsize=20, color='k')
        plt.text(x_limit/10-0.265+x_off,y_limit*0.7+0.2+y_off,'$z$',
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=18, color='k')

        # Surface isochrone text label
        si_posn = 0.35 #1.1
        plt.text(si_posn+0.13,si_posn*np.tan(beta_),
                 r'surface isochrone $\,\,T(\mathbf{r})$',
                 rotation=beta_deg,
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=15,  color='Gray')

        # Angle arc and text
        arc_radius = 1.35
        axes.add_patch( mpatches.Arc((0, 0), arc_radius,arc_radius, color='Gray',
                                     linewidth=1.5, fill=False, zorder=2,
                                     theta1=0, theta2=60) )
        plt.text(0.18,0.18, rf'$\beta = {beta_deg}\degree$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=20,  color='Gray')

        # Erosion arrow
        l_erosion_arrow = 0.2
        gray_ = self.gray_color(2,5)
        off_ = 0.38
        plt.arrow(origin_[0]+off_+0.02, origin_[1]+off_*np.tan(beta_),
                  l_erosion_arrow*np.tan(beta_),-l_erosion_arrow,
                  head_width=0.08, head_length=0.1, lw=7,
                  length_includes_head=True, ec=gray_, fc=gray_,
                  capstyle='butt', overhang=0.1)
        off_x, off_z = 0.27, 0.0
        plt.text(origin_[0]+off_+0.02+off_x,origin_[1]+off_*np.tan(beta_)+off_z,'erosion',
                 rotation=beta_deg-90,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=15,  color=gray_)

        # Unit normal n vector
        off_x, off_z = 1.1, 0.55
        plt.text(origin_[0]+off_x, origin_[1]+off_z,
                 r'$\mathbf{n} = $',
                 horizontalalignment='right', verticalalignment='center',
                 fontsize=15,  color='k')
        plt.text(origin_[0]+off_x, origin_[1]+off_z,
                 r'$\left[ \,\genfrac{}{}{0}{}{\sqrt{3}/{2}}{-1/2}\, \right]$',
                 #  \binom{\sqrt{3}/{2}}{-1/2}
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=20,  color='k')

        # p text annotations
        plt.text(origin_[0]+1.7,origin_[1]+0.1,
                 r'${{p}}_x(\mathbf{{n}}) = 3$',
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=18, color=p_color)
        plt.text(origin_[0]+0.1,origin_[1]-1,
                 r'${{p}}_z(\mathbf{{n}}) = 1$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=18, color=p_color)
        plt.text(origin_[0]+1.77,origin_[1]-1.05,
                 r'$\mathbf{\widetilde{p}}(\mathbf{{n}}) = p = 4$',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=18, color=p_color)
        plt.text(origin_[0]+1,origin_[1]-0.4,
                 r'$\mathbf{\widetilde{p}} = [2\sqrt{3} \,\,\, -\!2]$',
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

        # Length scale
        inset_axes_ = inset_axes(axes, width=f'{31.5}%', height=0.15, loc=2)
        plt.xticks(np.linspace(0,2,3),labels=[0,0.25,0.5])
        plt.yticks([])
        plt.xlabel('distance  [mm]')
        inset_axes_.spines['top'].set_visible(False)
        inset_axes_.spines['left'].set_visible(False)
        inset_axes_.spines['right'].set_visible(False)
        inset_axes_.spines['bottom'].set_visible(True)

    def huygens_wavelets( self, gmes, gmeq, sub, name, fig_size=None, dpi=None,
                          do_ray_conjugacy=False, do_fast=False,
                          do_legend=True, legend_fontsize=10,
                          annotation_fontsize=11) -> None:
        r"""
        Plot the loci of :math:`\mathbf{\widetilde{p}}` and :math:`\mathbf{r}` and
        their behavior defined by :math:`F` relative to the :math:`\xi` circle.

        Args:
            fig (:obj:`Matplotlib figure <matplotlib.figure.Figure>`):
                reference to figure instantiated by
                :meth:`GMPLib create_figure <plot.GraphingBase.create_figure>`
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            do_ray_conjugacy (bool): optional generate ray conjugacy schematic?
        """
        fig = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        axes = plt.gca()

        def trace_indicatrix(sub, n_points, xy_offset, sf=1,
                             pz_min_=2.5e-2, pz_max_=1000) -> Tuple[np.array, np.array]:
            r"""
            TBD
            """
            rdotx_pz_eqn = gmeq.idtx_rdotx_pz_varphi_eqn.subs(sub)
            rdotz_pz_eqn = gmeq.idtx_rdotz_pz_varphi_eqn.subs(sub)
            rdotx_pz_lambda = lambdify( pz, (re(N(rdotx_pz_eqn.rhs))), 'numpy')
            rdotz_pz_lambda = lambdify( pz, (re(N(rdotz_pz_eqn.rhs))), 'numpy')
            fgtx_pz_array = -pz_min_*np.power(2,np.linspace(np.log2(pz_max_/pz_min_),
                                                            np.log2(pz_min_/pz_min_),
                                                            n_points, endpoint=True))
            # print(fgtx_pz_array)
            idtx_rdotx_array \
                = np.array([ float(rdotx_pz_lambda(pz_)) for pz_ in fgtx_pz_array] )
            idtx_rdotz_array \
                = np.array([ float(rdotz_pz_lambda(pz_)) for pz_ in fgtx_pz_array] )
            return idtx_rdotx_array*sf+xy_offset[0],idtx_rdotz_array*sf+xy_offset[1]

        isochrone_color, isochrone_width, isochrone_ms, isochrone_ls \
                = 'Black', 2, 8 if do_ray_conjugacy else 7, '-'
        new_isochrone_color, newpt_isochrone_color, new_isochrone_width, \
            new_isochrone_ms, new_isochrone_ls \
                = 'Gray', 'White', 2 if do_ray_conjugacy else 4, \
                  8 if do_ray_conjugacy else 7, '-'
        wavelet_color = 'DarkRed'
        wavelet_width = 2.5 if do_ray_conjugacy else 1.5
        p_color, _ = 'Blue', 2
        r_color, r_width = '#15e01a', 1.5

        dt_ = 0.0015
        dz_ = -gmes.xiv_v_array[0]*dt_*1.15
        # Fudge factors are NOT to "correct" curves but rather to account
        #   for the exaggerated Delta{t} that makes the approximations here just wrong
        #   enough to make the r, p etc annotations a bit wonky
        dx_fudge, dp_fudge = 0.005,1.15

        # Old isochrones
        plt.plot( gmes.h_x_array[0],gmes.h_z_array[0], 'o', mec='k',
                  mfc=isochrone_color, ms=isochrone_ms, fillstyle='full',
                  markeredgewidth=0.5,
                  label=r'point $\mathbf{r}$')
        plt.plot( gmes.h_x_array, gmes.h_z_array, lw=isochrone_width,
                  c=isochrone_color, ls=isochrone_ls,
                  label=r'isochrone  $T(\mathbf{r})=t$' )

        # Adjust plot scale, limits
        # if do_ray_conjugacy:
        #     x_limits = [0.35,0.62]; y_limits = [0.018,0.12]
        # else:
        #     x_limits = [0.45,0.75]; y_limits = [0.03,0.19]
        # HACK!!!
        # plt.xlim(*x_limits); plt.ylim(*y_limits)
        axes.set_aspect(1)

        # New isochrones
        plt.plot( gmes.h_x_array+dx_fudge, gmes.h_z_array + dz_,
                  c=new_isochrone_color, lw=new_isochrone_width, ls=new_isochrone_ls )

        # Erosion arrow
        i_ = 161 if do_ray_conjugacy else 180
        # rx_, rz_ = ( (gmes.h_x_array[i_]+gmes.h_x_array[i_-1])/2,
        #              (gmes.h_z_array[i_]+gmes.h_z_array[i_-1])/2 )
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
        lw = 3.0 if do_ray_conjugacy else 5.0
        l_erosion_arrow = 0.4
        gray_ = self.gray_color(2,5)
        plt.arrow( rx_,rz_, l_erosion_arrow*np.tan(beta_)*sf,-l_erosion_arrow*sf,
                   head_width=0.15*sf, head_length=0.15*sf, lw=lw,
                   length_includes_head=True, ec=gray_, fc=gray_,
                   capstyle='butt', overhang=0.1*sf )

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
            print(rx_,rz_)
            drx_, drz = gmes.rdotx_interp(rx_)*dt_, gmes.rdotz_interp(rx_)*dt_
            rxn_, rzn_ = rx_+drx_, rz_+drz
            recip_p_ = (1/gmes.p_interp(rx_))*dt_
            beta_ = float(gmes.beta_p_interp(rx_))
            dpx_, dpz_ = recip_p_*np.sin(beta_)*dp_fudge, -recip_p_*np.cos(beta_)*dp_fudge
            varphi_ = float(gmeq.varphi_rx_eqn.rhs.subs(sub).subs({rx:rx_}))
            n_points = 80 if do_ray_conjugacy else 5 if do_fast else 50
            pz_max_ = 1000 if do_ray_conjugacy else 1000
            idtx_rdotx_array,idtx_rdotz_array \
                = trace_indicatrix( {varphi:varphi_},  n_points=n_points,
                                    xy_offset=[rx_, rz_], sf=dt_,
                                    pz_min_=1e-3, pz_max_=pz_max_ )

            # Plot wavelets
            lw = 1.5
            plt.plot( idtx_rdotx_array,idtx_rdotz_array, lw=wavelet_width, ls='-',
                      c=wavelet_color,
                      label=r'erosional wavelet $\{\Delta\mathbf{r}\}$' if i_==i_from
                            else None )
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
                      lw=2 if do_ray_conjugacy else r_width, alpha=1,
                      c=r_color, linestyle='-', # shape='full',
                      label=r'ray increment  $\Delta{\mathbf{r}}$' if i_==i_from
                            else None)
            plt.plot( rxn_, rzn_, 'o', mec=new_isochrone_color, mfc=newpt_isochrone_color,
                      ms=new_isochrone_ms, fillstyle=None, markeredgewidth=1.5)

            # Normal slownesses
            plt.plot([rx_,rx_+dpx_],[rz_,rz_+dpz_],'-', c=p_color,
                     lw=3 if do_ray_conjugacy else r_width,
                     label=r'front increment  $\mathbf{\widetilde{p}}\Delta{t}\,/\,{p}^2$'
                     if i_==i_from else None)

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

        print(type(fig.add_axes([*[0.65, 0.55], *[0.65, 0.55]])))
        print('here')

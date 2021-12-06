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
  -  :mod:`numpy`, :mod:`sympy`
  -  :mod:`matplotlib.pyplot`
  -  :mod:`gme.plot.base`

---------------------------------------------------------------------

"""

import warnings

# Numpy
import numpy as np

# SymPy
from sympy import Rational

# MatPlotLib
import matplotlib.pyplot as plt

# GME
from gme.plot.base import Graphing

warnings.filterwarnings("ignore")

__all__ = ['AlphaBeta']


class AlphaBeta(Graphing):
    """
    Subclasses :class:`gme.plot.Graphing <plot.Graphing>`
    """

    def alpha_beta(self, gmeq, name, alpha_array, beta_array, tanalpha_crit_,
                   tanbeta_crit_, fig_size=None, dpi=None ) -> None:
        """
        TBD
        """
        # Create figure
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        # fig = gr.create_figure(job_name+'_alpha_beta', fig_size=(6,3))
        plt.plot(beta_array, alpha_array, 'b')
        x_,y_ = ( np.rad2deg(np.arctan(float(tanbeta_crit_))),
                 np.rad2deg(np.arctan(float(tanalpha_crit_))) )
        plt.plot( x_,y_, 'ob' )
        plt.text( x_,y_-y_/9, r'$\beta_c, \,\alpha_c$',
                 color='b', horizontalalignment='center', fontsize=14)
        plt.text( 40,y_/4, fr'$\eta = ${gmeq.eta_}', color='k',
                  horizontalalignment='center', fontsize=14)
        plt.text( 87,y_*0.67, '(a)' if gmeq.eta_==Rational(3,2)
                                    else ('(b)' if gmeq.eta_==Rational(1,2) else ''),
                  color='k', horizontalalignment='center', fontsize=16 )
        plt.grid('on')
        plt.xlabel(r'Surface tilt  $\beta$   [${\degree}\!$ from horiz]')
        plt.ylabel(r'Ray angle  $\alpha$   [${\degree}\!$ from horiz]')

    def beta_anisotropy(self, gmeq, name, alpha_array, beta_array,
                        tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ) -> None:
        """
        TBD
        """
        # Create figure
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(beta_array, alpha_array-beta_array+90, 'b')
        x_,y_ = ( np.rad2deg(np.arctan(float(tanbeta_crit_))),
                 np.rad2deg(np.arctan(float(tanalpha_crit_))) )
        plt.plot( x_,y_-x_+90, 'ob' )
        if gmeq.eta_<1:
            plt.text( x_*(1.0 if gmeq.eta_<Rational(1,2) else 1.0),
                     (y_-x_+90)*(1.15), r'$\beta_c$', color='b',
                     horizontalalignment='center', fontsize=14)
        else:
            plt.text( x_*1,(y_-x_+90)*(0.85), r'$\beta_c$', color='b',
                      horizontalalignment='center', fontsize=14)
        plt.text( 75,55, fr'$\eta = ${gmeq.eta_}', color='k',
                 horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.text( 30,15, '(a)' if gmeq.eta_==Rational(3,2) else '(b)',
                  color='k', horizontalalignment='center', verticalalignment='center',
                  fontsize=17 )
        plt.grid('on')
        plt.xlabel(r'Surface normal angle  $\beta$   [${\degree}\!$ from vertical]')
        plt.ylabel(r'Anisotropy   $\psi = \alpha-\beta+90{\degree}$   [${\degree}\!$]')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.xlim(0,90)
        plt.ylim(0,90)
        plt.plot(beta_array, beta_array, ':')

    def alpha_anisotropy(self, gmeq, name, alpha_array, beta_array,
                         tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ) -> None:
        """
        TBD
        """
        # Create figure
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(alpha_array-beta_array+90, alpha_array, 'b')
        x_,y_ = ( np.rad2deg(np.arctan(float(tanbeta_crit_))),
                 np.rad2deg(np.arctan(float(tanalpha_crit_))) )
        plt.plot( x_,y_, 'ob' )
        label_ = r'$\psi_c,\alpha_c$'
        if gmeq.eta_<1:
            plt.text( x_,y_*0.90, label_, color='b', horizontalalignment='center',
                      fontsize=14)
        else:
            plt.text( x_,y_*0.75, label_, color='b', horizontalalignment='center',
                      fontsize=14)
        plt.text( 40, 5 if gmeq.eta_>1 else -5, fr'$\eta = ${gmeq.eta_}', color='k',
                 horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.text( *((10,7.5) if gmeq.eta_==Rational(3,2) else (7,-16.5)),
                  '(a)' if gmeq.eta_==Rational(3,2) else '(b)',
                  color='k', horizontalalignment='center', verticalalignment='center',
                  fontsize=17 )
        plt.grid('on')
        plt.ylabel(r'Ray angle  $\alpha$   [${\degree}\!$]')
        plt.xlabel(r'Anisotropy   $\psi = \alpha-\beta+90{\degree}$   [${\degree}\!$]')
        axes = plt.gca()
        axes.invert_xaxis()
        axes.set_aspect(2)

    def alpha_image(self, gmeq, name, alpha_array, beta_array,
                    tanalpha_crit_, tanbeta_crit_, fig_size=None, dpi=None ) -> None:
        """
        TBD
        """
        # Create figure
        _ = self.create_figure(name, fig_size=fig_size, dpi=dpi)
        plt.plot(beta_array, beta_array-(alpha_array-beta_array+90), 'b')
        x_,y_ = ( np.rad2deg(np.arctan(float(tanbeta_crit_))),
                 np.rad2deg(np.arctan(float(tanalpha_crit_))) )
        plt.plot( x_,x_-(y_-x_+90), 'ob' )
        plt.text( x_,-15, r'$\beta_c$', color='b', horizontalalignment='center',
                  fontsize=14)
        plt.text( 40,62.5, fr'$\eta = ${gmeq.eta_}', color='k',
                 horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.text( 70,-62.5, '(a)' if gmeq.eta_==Rational(3,2) else '(b)', color='k',
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=17 )
        plt.grid('on')
        plt.ylabel(r'Image ray angle  $\beta-\psi$   [${\degree}\!$ from vertical]')
        plt.xlabel(r'Surface normal angle  $\beta$   [${\degree}\!$ from vertical]')

"""
---------------------------------------------------------------------

Time-invariant topographic profile construction by ray tracing
aka ODE integration of Hamilton's equations.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`NumPy <numpy>`
  -  :mod:`SciPy <scipy>`
  -  :mod:`SymPy <sympy>`
  -  `GMPLib`_
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html

---------------------------------------------------------------------

"""
# Library
import warnings
import logging
from typing import List, Callable

# NumPy
import numpy as np

# SciPy
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline

# Sympy
from sympy import poly, Poly

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import xiv_0, xih_0, px, Lc
from gme.core.utils import gradient_value
from gme.ode.single_ray import SingleRaySolution

warnings.filterwarnings("ignore")

__all__ = ['TimeInvariantSolution']


class TimeInvariantSolution(SingleRaySolution):
    """
    Integration of Hamilton's equations (ODEs) to generate time-invariant
    (steady-state) profile solutions.

    Extends :class:`gme.ode.single_ray.SingleRaySolution`.
    """

    def postprocessing(
        self,
        spline_order: int = 2,
        extrapolation_mode: int = 0
    ) -> None:
        """
        Process the results of ODE integration, supplementing the
        standard set of array generations.

        Args:
            spline_order:
                optional order of spline interpolation to be used
                when resampling along the profile at regular x intervals
            extrapolation_mode:
                optionally extrapolate (1) or don't extrapolate (1)
                at the end of the interpolation span
        """
        logging.info('gme.core.time_invariant.postprocessing')
        super().postprocessing(spline_order=spline_order,
                               extrapolation_mode=extrapolation_mode)
        xiv0_ = float((xiv_0/xih_0).subs(e2d(self.gmeq.xiv0_xih0_Ci_eqn)))
        self.beta_vt_array = np.arctan(
            (self.rdotz_array+xiv0_)/self.rdotx_array)
        self.beta_vt_interp \
            = InterpolatedUnivariateSpline(self.rx_array,
                                           self.beta_vt_array,
                                           k=spline_order,
                                           ext=extrapolation_mode)
        self.beta_vt_error_interp \
            = lambda x_: 100*(self.beta_vt_interp(x_)-self.beta_p_interp(x_)) \
            / self.beta_p_interp(x_)
        (self.x_array, self.h_array) = [
            np.empty_like(self.t_array) for idx in [0, 1]]
        self.rays: List = []
        for i in range(0, len(self.t_array), 1):
            if i < len(self.t_array):
                self.rays \
                    += [np.array([self.rx_array[:i+1],
                                  self.rz_array[:i+1]+self.t_array[i]*xiv0_])]
            self.x_array[i] = self.rx_array[i]
            self.h_array[i] = self.rz_array[i]+self.t_array[i]*xiv0_
        self.h_interp: Callable \
            = InterpolatedUnivariateSpline(self.x_array,
                                           self.h_array,
                                           k=spline_order,
                                           ext=extrapolation_mode)
        dhdx_interp: Callable \
            = InterpolatedUnivariateSpline(self.x_array,
                                           self.h_array,
                                           k=spline_order,
                                           ext=extrapolation_mode).derivative()
        self.beta_ts_interp = lambda x_: np.arctan(dhdx_interp(x_))
        self.dhdx_array: np.ndarray = dhdx_interp(self.x_array)
        self.beta_ts_array: np.ndarray = self.beta_ts_interp(self.x_array)
        self.beta_ts_error_interp: Callable \
            = lambda x_: 100*(self.beta_ts_interp(x_)-self.beta_p_interp(x_)) \
            / self.beta_p_interp(x_)

    def integrate_h_profile(
        self,
        n_pts: int = 301,
        x_max: float = None,
        do_truncate: bool = True,
        do_use_newton: bool = False
    ) -> None:
        """
        Generate topographic profile by numerically integrating gradient
        using simple quadrature.

        Args:
            n_pts:
                optional sample rate along each curve
            x_max:
                optional x-axis limit
            do_truncate:
                optionally omit that last couple of points?
            do_use_newton:
                optionally use Newton-Raphson method to find gradient values
                (default is to find these values algebraically)
        """
        logging.info('gme.core.time_invariant.integrate_h_profile')
        x_max_: float = float(Lc.subs(self.parameters)) if x_max is None \
            else x_max
        self.h_x_array: np.ndarray = np.linspace(0, x_max_, n_pts)
        px0_poly_eqn: Poly \
            = poly(self.gmeq.poly_px_xiv0_eqn.subs(self.parameters), px)
        if do_truncate:
            h_x_array = self.h_x_array[:-2]
            gradient_array: np.ndarray \
                = np.array([gradient_value(x_,
                                           pz_=self.pz0,
                                           px_poly_eqn=px0_poly_eqn,
                                           do_use_newton=do_use_newton)
                            for x_ in h_x_array])
            h_z_array: np.ndarray \
                = cumtrapz(gradient_array, h_x_array, initial=0)
            h_z_interp: Callable \
                = InterpolatedUnivariateSpline(h_x_array,
                                               h_z_array,
                                               k=2,
                                               ext=0)
            self.h_z_array: np.ndarray = h_z_interp(self.h_x_array)
        else:
            h_x_array = self.h_x_array
            # for x_ in h_x_array:
            #     print(x_,self.gradient_value(x_,parameters=self.parameters))
            gradient_array \
                = np.array([gradient_value(x_,
                                           pz_=self.pz0,
                                           px_poly_eqn=px0_poly_eqn,
                                           do_use_newton=do_use_newton)
                            for x_ in h_x_array])
            self.h_z_array = cumtrapz(gradient_array, h_x_array, initial=0)


#

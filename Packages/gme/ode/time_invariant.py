"""
---------------------------------------------------------------------

ODE integration of Hamilton's equations.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`gmplib.utils`
  -  :mod:`numpy`
  -  :mod:`scipy`
  -  :mod:`sympy`

Imports symbols from :mod:`.symbols` module

---------------------------------------------------------------------

"""
# pylint: disable=line-too-long, invalid-name, too-many-locals, multiple-statements, too-many-arguments, too-many-branches
import warnings

# Numpy
import numpy as np

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import xiv_0, xih_0, px, Lc
from gme.core.equations import gradient_value
from gme.ode.single_ray import SingleRaySolution

# Sympy
from sympy import poly

# SciPy
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline

warnings.filterwarnings("ignore")

rp_list = ['rx','rz','px','pz']
rpt_list = rp_list+['t']

__all__ = ['TimeInvariantSolution']


class TimeInvariantSolution(SingleRaySolution):
    """
    Integration of Hamilton's equations (ODEs) to generate time-invariant (steady-state) profile solutions.
    """
    def more_postprocessing(self, spline_order=2, extrapolation_mode=0) -> None:
        """
        TBD
        """
        xiv0_ = float( (xiv_0/xih_0).subs(e2d(self.gmeq.xiv0_xih0_Ci_eqn)) )
        self.beta_vt_array = np.arctan( (self.rdotz_array+xiv0_)/self.rdotx_array )
        self.beta_vt_interp = InterpolatedUnivariateSpline( self.rx_array, self.beta_vt_array,
                                                            k=spline_order, ext=extrapolation_mode )
        self.beta_vt_error_interp = lambda x_: 100*( self.beta_vt_interp(x_)-self.beta_p_interp(x_) ) \
                                                        / self.beta_p_interp(x_)
        self.x_array, self.h_array = [np.empty_like(self.t_array) for idx in [0,1]]
        self.rays = []
        for i in range(0,len(self.t_array),1):
            if i<len(self.t_array):
                self.rays += [np.array([self.rx_array[:i+1],self.rz_array[:i+1]+self.t_array[i]*xiv0_])]
            self.x_array[i] = self.rx_array[i]
            self.h_array[i] = self.rz_array[i]+self.t_array[i]*xiv0_
        self.h_interp  = InterpolatedUnivariateSpline( self.x_array, self.h_array,
                                                       k=spline_order, ext=extrapolation_mode )
        dhdx_interp = InterpolatedUnivariateSpline( self.x_array, self.h_array,
                                                    k=spline_order, ext=extrapolation_mode ).derivative()
        self.beta_ts_interp = lambda x_: np.arctan(dhdx_interp(x_))
        self.dhdx_array = dhdx_interp(self.x_array)
        self.beta_ts_array = self.beta_ts_interp( self.x_array )
        self.beta_ts_error_interp = lambda x_: 100*( self.beta_ts_interp(x_)-self.beta_p_interp(x_) ) \
                                                    /self.beta_p_interp(x_)

    def integrate_h_profile(self, n_pts=301, x_max=None, do_truncate=True, do_use_newton=False) -> None:
        """
        TBD
        """
        x_max = float(Lc.subs(self.parameters)) if x_max is None else x_max
        self.h_x_array = np.linspace(0,x_max,n_pts)
        px0_poly_eqn = poly(self.gmeq.poly_px_xiv0_eqn.subs(self.parameters),px)#.subs({xih_0:1}) # HACK!!!
        if do_truncate:
            h_x_array = self.h_x_array[:-2]
            gradient_array = np.array([gradient_value(x_, pz_=self.pz0, px_poly_eqn=px0_poly_eqn,
                                        do_use_newton=do_use_newton, parameters=self.parameters)
                                       for x_ in h_x_array])
            h_z_array = cumtrapz(gradient_array, h_x_array, initial=0)
            h_z_interp  = InterpolatedUnivariateSpline( h_x_array, h_z_array, k=2, ext=0 )
            self.h_z_array = h_z_interp(self.h_x_array)
        else:
            h_x_array = self.h_x_array
            # for x_ in h_x_array:
            #     print(x_,self.gradient_value(x_,parameters=self.parameters))
            gradient_array = np.array([gradient_value(x_, pz_=self.pz0, px_poly_eqn=px0_poly_eqn,
                                        do_use_newton=do_use_newton, parameters=self.parameters)
                                       for x_ in h_x_array])
            self.h_z_array = cumtrapz(gradient_array, h_x_array, initial=0)



#

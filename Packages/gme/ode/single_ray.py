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
import warnings
import logging

# Typing
from typing import Tuple #, Any, List #, Callable, List #, Dict, Any, Optional

# Numpy
import numpy as np

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import Lc
from gme.core.equations import pxpz0_from_xiv0
from gme.ode.base import ExtendedSolution, solve_Hamiltons_equations

# SciPy
from scipy.interpolate import InterpolatedUnivariateSpline

warnings.filterwarnings("ignore")

rp_list = ['rx','rz','px','pz']
rpt_list = rp_list+['t']

__all__ = ['SingleRaySolution']


class SingleRaySolution(ExtendedSolution):
    """
    Integration of Hamilton's equations (ODEs) to solve for propagation of a single ray.
    """
    def initial_conditions(self) -> Tuple[float,float,float,float]:
        """
        TBD
        """
        rz0_, rx0_ = 0, 0
        px0_, pz0_ = pxpz0_from_xiv0( self.parameters,
                                      self.gmeq.pz_xiv_eqn,
                                      self.gmeq.poly_px_xiv0_eqn )
        return (rz0_, rx0_ , px0_, pz0_.subs(e2d(self.gmeq.xiv0_xih0_Ci_eqn)))

    def solve(self) -> None:
        """
        TBD
        """
        # Record the ic as a list of one - to be used by solve_ODE_system
        self.ic_list = [self.initial_conditions()]
        self.model_dXdt_lambda = self.make_model()
        parameters_ = {Lc: self.parameters[Lc]}
        logging.debug('ode.single_ray.solve: calling solver')
        ivp_soln, rpt_arrays = solve_Hamiltons_equations( model=self.model_dXdt_lambda,
                                                           method=self.method,
                                                           do_dense=self.do_dense,
                                                           ic=self.ic_list[0],
                                                           parameters=parameters_,
                                                           t_array=self.ref_t_array.copy(),
                                                           x_stop=self.x_stop,
                                                           t_lag=0.0 )
        self.ivp_solns_list = [ivp_soln]
        self.save(rpt_arrays, 0)

    def postprocessing(self, spline_order=2, extrapolation_mode=0) -> None:
        """
        TBD
        """
        # Basics
        (self.rx_array, self.rz_array, self.px_array, self.pz_array, self.t_array) \
            = [self.rpt_arrays[rpt_][0] for rpt_ in rpt_list]
        self.pz0 = self.pz_array[0]
        self.p_array = np.sqrt(self.px_array**2+self.pz_array**2)
        rpdot_array = np.array([self.model_dXdt_lambda(0, rp_)
                                for rp_ in zip(self.rx_array,self.rz_array,self.px_array,self.pz_array)])
        self.rdotx_array, self.rdotz_array, self.pdotx_array, self.pdotz_array \
                                                = [rpdot_array[:,idx] for idx in [0,1,2,3]]
        self.rdot_array  = np.sqrt(self.rdotx_array**2+self.rdotz_array**2)
        self.pdot_array  = np.sqrt(self.pdotx_array**2+self.pdotz_array**2)

        # Interpolation functions to facilitate resampling and co-plotting
        self.t_interp_x = InterpolatedUnivariateSpline( self.rx_array, self.t_array,
                                                        k=spline_order, ext=extrapolation_mode )
        self.rx_interp_t = InterpolatedUnivariateSpline( self.t_array, self.rx_array,
                                                         k=spline_order, ext=extrapolation_mode )
        self.rz_interp_t = InterpolatedUnivariateSpline( self.t_array, self.rz_array,
                                                         k=spline_order, ext=extrapolation_mode )

        self.x_interp_t = InterpolatedUnivariateSpline( self.t_array, self.rx_array,
                                                        k=spline_order, ext=extrapolation_mode )
        self.rz_interp = InterpolatedUnivariateSpline( self.rx_array, self.rz_array,
                                                       k=spline_order, ext=extrapolation_mode )
        self.p_interp = InterpolatedUnivariateSpline( self.rx_array, self.p_array,
                                                      k=spline_order, ext=extrapolation_mode )
        self.rdot_interp = InterpolatedUnivariateSpline( self.rx_array, self.rdot_array,
                                                         k=spline_order, ext=extrapolation_mode )
        self.rdotx_interp = InterpolatedUnivariateSpline( self.rx_array, self.rdotx_array,
                                                          k=spline_order, ext=extrapolation_mode )
        self.rdotz_interp = InterpolatedUnivariateSpline( self.rx_array, self.rdotz_array,
                                                          k=spline_order, ext=extrapolation_mode )
        self.pdot_interp  = InterpolatedUnivariateSpline( self.rx_array, self.pdot_array,
                                                          k=spline_order, ext=extrapolation_mode )
        self.px_interp = InterpolatedUnivariateSpline( self.rx_array, self.px_array,
                                                       k=spline_order, ext=extrapolation_mode )
        self.pz_interp = InterpolatedUnivariateSpline( self.rx_array, self.pz_array,
                                                       k=spline_order, ext=extrapolation_mode )

        # Ray acceleration
        self.rdotx_interp_t = InterpolatedUnivariateSpline( self.t_array, self.rdotx_array,
                                                            k=spline_order, ext=extrapolation_mode )
        self.rdotz_interp_t = InterpolatedUnivariateSpline( self.t_array, self.rdotz_array,
                                                            k=spline_order, ext=extrapolation_mode )
        self.rddotx_interp_t = self.rdotx_interp_t.derivative(n=1)
        self.rddotz_interp_t = self.rdotz_interp_t.derivative(n=1)

        # Angles
        self.tanbeta_array = -self.px_array/self.pz_array
        self.tanalpha_array = -self.rdotx_array/self.rdotz_array
        self.beta_array = np.arctan(self.tanbeta_array)
        self.beta_p_interp = InterpolatedUnivariateSpline( self.rx_array, self.beta_array,
                                                            k=spline_order, ext=extrapolation_mode )

        self.alpha_array = np.mod(np.pi+np.arctan(self.tanalpha_array),np.pi)
        self.alpha_interp = InterpolatedUnivariateSpline( self.rx_array, self.alpha_array,
                                                            k=spline_order, ext=extrapolation_mode )

        self.cosbeta_array = -self.pz_array/self.p_array
        self.sinbeta_array =  self.px_array/self.p_array

        # Erosion rates
        self.u_array = 1/self.p_array
        self.xiv_p_array = self.rdot_array*( np.cos((self.alpha_array)-(self.beta_array)) \
                                                        )/self.cosbeta_array
        self.xiv_p_array[np.isnan(self.xiv_p_array)] = 0
        self.xiv_v_array = (self.u_array/self.cosbeta_array)
        self.uhorizontal_p_array = self.rdot_array*( np.cos((self.alpha_array)-(self.beta_array)) ) \
                                                                /self.sinbeta_array
        self.uhorizontal_p_array[np.isnan(self.uhorizontal_p_array)] = 0
        self.uhorizontal_v_array = self.u_array/np.sin(self.beta_array)
        self.u_interp = InterpolatedUnivariateSpline( self.rx_array, self.u_array,
                                                            k=spline_order, ext=extrapolation_mode )
        self.u_from_rdot_interp = InterpolatedUnivariateSpline( self.rx_array,
                                        self.rdot_array*( np.cos((self.alpha_array)-(self.beta_array)) ),
                                                            k=spline_order, ext=extrapolation_mode )
        # u^\downarrow = \dot{r}*\cos(\alpha-\beta) / \cos(\beta)
        self.xiv_p_interp = InterpolatedUnivariateSpline( self.rx_array, self.xiv_p_array,
                                                            k=spline_order, ext=extrapolation_mode )
        # u^\downarrow = u^\perp / \cos(\beta)
        self.xiv_v_interp = InterpolatedUnivariateSpline( self.rx_array, self.xiv_v_array,
                                                            k=spline_order, ext=extrapolation_mode )
        # u^\downarrow = \dot{r}*\cos(\alpha-\beta) / \sin(\beta)
        self.uhorizontal_p_interp = InterpolatedUnivariateSpline( self.rx_array, self.uhorizontal_p_array,
                                                            k=spline_order, ext=extrapolation_mode )
        # u^\downarrow = u^\perp / \sin(\beta)
        self.uhorizontal_v_interp = InterpolatedUnivariateSpline( self.rx_array, self.uhorizontal_v_array,
                                                            k=spline_order, ext=extrapolation_mode )



#

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

# Typing
# from typing import List, Dict, Any, Tuple, Callable, Optional

# Numpy
import numpy as np

# GMPLib
from gmplib.utils import vprint, e2d

# GME
from gme.equations import gradient_value, pxpz0_from_xiv0
from gme.ode_base import BaseSolution
from gme.symbols import *
from sympy import N, sign, atan, atan2, sin, cos, tan, re, im, sqrt, \
    Matrix, lambdify, Abs, simplify, expand, solve, Eq, Rational, diff, \
    nroots, poly

# SciPy
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import root_scalar, fsolve

warnings.filterwarnings("ignore")

rp_list = ['rx','rz','px','pz']
rpt_list = rp_list+['t']

__all__ = ['OneRaySolution', 'TimeInvariantSolution', 'VelocityBoundarySolution']


class OneRaySolution(BaseSolution):
    """
    Integration of Hamilton's equations (ODEs) to solve for propagation of a single ray.
    """
    def __init__(self, gmeq, parameters, **kwargs):
        """
        Initialize class instance.

        Args:
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            parameters (dict): dictionary of model parameter values to be used for equation substitutions
            kwargs (dict): remaining keyword arguments (see base class for details)
        """
        super().__init__(gmeq, parameters, **kwargs)

    def initial_conditions(self, px_guess=1):
        rz0_, rx0_ = 0, 0
        px0_, pz0_ = pxpz0_from_xiv0(self.parameters, self.gmeq.pz_xiv_eqn, self.gmeq.poly_px_xiv0_eqn, px_guess=px_guess )
        # Hack
        return [rz0_, rx0_ , px0_, pz0_.subs(e2d(self.gmeq.xiv0_xih0_Ci_eqn))]

    def solve(self, px_guess=1):
        self.prep_arrays()
        self.ic = self.initial_conditions(px_guess=px_guess)
        #if self.model_dXdt_lambda is None:
        self.model_dXdt_lambda = self.make_model()
        self.rpt_arrays = self.solve_Hamiltons_equations(t_array=self.ref_t_array.copy())
        rx_length = len(self.rpt_arrays['rx'])
        self.rpt_arrays['t'] = self.rpt_arrays['t'][:rx_length]
        return self.solns

    def postprocessing(self, spline_order=2, extrapolation_mode=0):
        # Basics
        (self.rx_array, self.rz_array, self.px_array, self.pz_array, self.t_array) \
            = [self.rpt_arrays[rpt_] for rpt_ in rpt_list]
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


class TimeInvariantSolution(OneRaySolution):
    """
    Integration of Hamilton's equations (ODEs) to generate time-invariant (steady-state) profile solutions.
    """
    def __init__(self, gmeq, parameters, **kwargs):
        """
        Initialize class instance.

        Args:
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            parameters (dict): dictionary of model parameter values to be used for equation substitutions
            kwargs (dict): remaining keyword arguments (see base class for details)
        """
        super().__init__(gmeq, parameters, **kwargs)

    def more_postprocessing(self, spline_order=2, extrapolation_mode=0):
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
                self.rays += [np.array([self.rx_array[:i+1],
                                self.rz_array[:i+1]+self.t_array[i]*xiv0_])]
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

    def integrate_h_profile(self, n_pts=301, x_max=None, do_truncate=True, do_use_newton=False):
        x_max = float(x_1.subs(self.parameters)) if x_max is None else x_max
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


class VelocityBoundarySolution(BaseSolution):
    """
    Integration of Hamilton's equations (ODEs) from a 'fault slip' velocity boundary.

    Currently the velocity boundary is required to lie along the left domain edge and to be vertical.
    """
    def __init__(self, gmeq, parameters, **kwargs):
        """
        Initialize class instance.

        Args:
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            parameters (dict): dictionary of model parameter values to be used for equation substitutions
            kwargs (dict): remaining keyword arguments (see base class for details)
        """
        super().__init__(gmeq, parameters, **kwargs)

    def initial_conditions(self, t_lag, xiv_0_, px_guess=1):
        self.parameters[xiv_0] = xiv_0_
        # px0_, pz0_ = self.pxpz0_from_xiv0()
        px0_, pz0_tmp = pxpz0_from_xiv0(self.parameters, self.gmeq.pz_xiv_eqn, self.gmeq.poly_px_xiv0_eqn, px_guess=px_guess )
        # HACK!!
        pz0_ = pz0_tmp.subs(e2d(self.gmeq.xiv0_xih0_Ci_eqn))
        cosbeta_ = np.sqrt(1/(1+(np.float(px0_/-pz0_))**2))
        rz0_ = t_lag/(pz0_*cosbeta_)
        rx0_ = 0
        return [rx0_,rz0_,px0_,pz0_]

    def solve(self, report_pc_step=1):
        self.prep_arrays()
        self.t_ensemble_max = 0

        # Construct a list of % durations and vertical velocities if only xiv_0 given
        self.tp_xiv0_list = [[1,self.parameters[xiv_0]]] if self.tp_xiv0_list is None \
                            else self.tp_xiv0_list

        # Calculate vertical distances spanned by each tp, uv0
        rz0_array = np.array([self.initial_conditions(tp_*self.t_slip_end, xiv0_)[1]
                                            for tp_,xiv0_ in self.tp_xiv0_list])
        rz0_cumsum_array = np.cumsum(rz0_array)
        offset_rz0_cumsum_array = np.concatenate([np.array([0]),rz0_cumsum_array])[:-1]
        # The total vertical distance spanned by all initial rays is rz0_cumsum_array[-1]
        rz0_total = rz0_cumsum_array[-1]

        # Apportion numbers of rays based on rz0 proportions
        n_block_rays_array = np.array([int(round(self.n_rays*(rz0_/rz0_total))) for rz0_ in rz0_array])
        offset_n_block_rays_array = np.concatenate([np.array([0]),n_block_rays_array])[:-1]
        self.n_rays = np.sum(n_block_rays_array)
        n_rays = self.n_rays
        #assert(len(self.tp_xiv0_list)==len(n_block_rays_array))

        # Step through each "block" of rays tied to a different boundary velocity
        #   and generate an initial condition for each ray
        t_lag_xiv0_ic_list = [None]*n_rays
        prev_t_lag = 0
        for i_tp_xiv0, (n_block_rays, (tp_,xiv0_), prev_rz0, prev_n_block_rays) \
                in enumerate(zip(n_block_rays_array, self.tp_xiv0_list,
                                 offset_rz0_cumsum_array, offset_n_block_rays_array)):
            # Generate initial conditions for all the rays in this block
            for i_ray in list(range(0,n_block_rays)):
                t_lag = (i_ray/(n_block_rays-1))*self.t_slip_end*tp_
                rx0_,rz0_,px0_,pz0_ = self.initial_conditions(t_lag, xiv0_)
                t_lag_xiv0_ic_list[i_ray+prev_n_block_rays] = (prev_t_lag+t_lag, xiv0_,
                                                                [rx0_,rz0_+prev_rz0,px0_,pz0_])
            prev_t_lag += t_lag

        # Generate rays in reverse order so that the first ray is topographically the lowest
        pc_progress = self.report_progress(i=0, n=n_rays, is_initial_step=True)
        for idx,i_ray in enumerate(list(range(0,n_rays))):
            t_lag, xiv0_, self.ic = t_lag_xiv0_ic_list[n_rays-1-i_ray]
            # print(f't_lag={t_lag}')
            self.parameters[xiv_0] = xiv0_
            self.model_dXdt_lambda = self.make_model()
            # Start rays from the bottom of the velocity boundary and work upwards
            #   so that their x,z,t disposition is consistent with initial profile, initial corner
            if self.choice=='Hamilton':
                rpt_arrays = self.solve_Hamiltons_equations( t_array=self.ref_t_array.copy(), t_lag=t_lag )
            else:
                rpt_arrays = self.solve_Hamiltons_equations( t_array=self.ref_t_array.copy(), t_lag=t_lag )
            self.t_ensemble_max = max(self.t_ensemble_max, rpt_arrays['t'][-1])
            # print(xiv0_, i_ray, rpt_arrays['rx'])
            self.save(rpt_arrays, i_ray)
            pc_progress = self.report_progress(i=idx, n=self.n_rays,
                                               pc_step=report_pc_step, progress_was=pc_progress)
        self.report_progress(i=idx, n=self.n_rays, pc_step=report_pc_step, progress_was=pc_progress)




#

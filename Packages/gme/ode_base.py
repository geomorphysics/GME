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

import numpy as np
from gmplib.utils import vprint, e2d
from gme.symbols import *
from sympy import N, sign, atan, atan2, sin, cos, tan, re, im, sqrt, \
    Matrix, lambdify, Abs, simplify, expand, solve, Eq, Rational, diff, \
    poly
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import fsolve
from copy import copy

rp_list = ['rx','rz','px','pz']
rpt_list = rp_list+['t']

__all__ = ['BaseSolution']


class BaseSolution():
    """
    Base class for  classes performing integration of Hamilton's equations (ODEs).
    """
    def __init__( self, gmeq, parameters,
                  choice='Hamilton',
                  method='Radau', do_dense=True, x_stop=0.999,
                  t_end=0.04, t_slip_end=0.08, t_distribn=2,
                  n_rays=20, n_t=101,
                  tp_xiv0_list=None, customize_t_fn=None,
                  verbose=True ):
        """
        Initialize class instance.

        Args:
            gmeq (:class:`~.equations.Equations`):
                    GME model equations class instance defined in :mod:`~.equations`
            parameters (dict): dictionary of model parameter values to be used for equation substitutions
        """
        # Container for GME equations
        self.gmeq = gmeq

        # Model/equation parameters
        self.parameters = parameters
        self.verbose = verbose

        # ODE solution method
        self.choice = choice
        self.method = method
        if self.choice=='Hamilton':
            str = 'Solve Hamilton\'s ODEs'
        else:
            str = 'Solve geodesic ODEs'
        str += f' using {method} method of integration'
        vprint(self.verbose, str)
        self.do_dense = do_dense
        self.x_stop = x_stop

        # Rays
        self.tp_xiv0_list = tp_xiv0_list
        self.n_rays = n_rays
        self.t_end = t_end
        self.t_distribn = t_distribn
        self.t_slip_end = t_slip_end
        self.n_t = n_t
        # To record the longest ray time
        self.t_ensemble_max = None
        # Ray interpolation
        self.interp1d_kind = 'linear'

        # ODEs & related
        # HACK: clumsy way to sub in value of mu
        # px_poly_tmp = expand(simplify(self.gmeq.pzpx_unity_eqn
        #                       .subs({varphi:self.gmeq.varphi_rx_eqn.rhs})
        #                       # .subs({mu:self.gmeq.mu, eta:self.gmeq.eta})
        #                       .subs(self.parameters) ))
        # self.px_poly_lambda = lambdify( [rx,pz], (poly(px_poly_tmp)).as_expr() )
        self.pz_velocity_boundary_eqn = gmeq.pz_xiv_eqn.subs({pz:pz_0, xiv:xiv_0}).subs(self.parameters)

        # self.px_varphi_xiv_eqn = simplify( Eq( self.gmeq.xiv_varphi_pxpz_eqn.lhs**2/xiv**2,
        #                                          self.gmeq.xiv_varphi_pxpz_eqn.rhs**2/xiv**2) )
        # self.px_varphi_xiv_eqn = self.gmeq.px_varphi_xiv_eqn

        # Initial condition equations
        # HACK!!
        # self.px_initial_surface_eqn = self.gmeq.px_initial_eqn #.subs({rx:x})
        # self.pz_initial_surface_eqn = self.gmeq.pz_initial_eqn #.subs({rx:x})

        # Misc
        self.model_dXdt_lambda = None
        self.customize_t_fn = customize_t_fn

    def make_model(self, do_verbose=False):
        # Requires self.parameters to have all the important Hamilton eqns' constants set
        #   - generates a "matrix" of the 4 Hamilton equations with rx,rz,px,pz as variables
        #     and the rest as numbers
        if self.choice=='Hamilton':
            vprint(self.verbose and do_verbose, 'Constructing model Hamilton\'s equations')
            drpdt_eqn_matrix = simplify( Matrix(([eq_ for eq_ in self.gmeq.hamiltons_eqns[0:4] ]))
                                        .subs(self.parameters)
                                        # .subs({mu:self.gmeq.mu, eta:self.gmeq.eta})
                                        .subs({-pz:Abs(pz)}) )
            drpdt_raw_lambda = lambdify( [rx, px, pz], drpdt_eqn_matrix )
            return lambda t_, rp_: np.ndarray.flatten( drpdt_raw_lambda(rp_[0],rp_[2],rp_[3]) )
        else:
            vprint(self.verbose and do_verbose, 'Constructing model geodesic equations')
            drvdt_eqn_matrix = Matrix(([(eq_.rhs) for eq_ in self.gmeq.geodesic_eqns ])) #factor
            drvdt_raw_lambda = lambdify( [rx, rdotx, rdotz], drvdt_eqn_matrix )
            return lambda t_, rv_: np.ndarray.flatten( drvdt_raw_lambda(rv_[0],rv_[2],rv_[3]) )

    def prep_arrays(self):
        self.ref_t_array = np.linspace(0,1,self.n_t)**self.t_distribn * self.t_end
        self.rpt_arrays = dict()
        [self.rpt_arrays.update({rp_: [None]*self.n_rays}) for rp_ in rpt_list]

    def solve_Hamiltons_equations(self, t_array, t_lag=0):
        soln_ivp, rpt_arrays, i_end = self.solve_ODE_system(t_array=t_array, t_lag=t_lag,
                                                            x_stop=self.x_stop)
        # print('solve_Hamiltons_equations #1:', i_end, len(rpt_arrays['rx']), rpt_arrays['rx'])
        self.solns = [soln_ivp]
        # print(f"i_end={i_end}, {len(rpt_arrays['t'])}, {len(rpt_arrays['rx'])}")
        # if i_end is not None:
        #     # Bug fix here - shouldn't be needed?
        #     if len(rpt_arrays['rx']) < i_end: i_end = len(rpt_arrays['rx'])
        #     if self.verbose:
        #         pass
        #     for rpt_ in rpt_list:
        #         a = copy(rpt_arrays[rpt_])
        #         rpt_arrays[rpt_] = a[:i_end]
        # print('solve_Hamiltons_equations #2:', rpt_arrays['rx'])
        return rpt_arrays

    def solve_ODE_system(self, t_array, t_lag=0, x_stop=0.999):
        # Define stop condition
        def almost_reached_divide(t,y):
            # function yielding >0 if rx<x1*x_stop ~ along profile
            #              and  <0 if rx>x1*x_stop ≈ @divide
            #  - thus triggers an event when rx surpasses x1*x_stop
            #    because = zero-crossing in -ve sense
            return y[0]-x_stop
        almost_reached_divide.terminal = True

        # Perform ODE integration
        soln_ivp = solve_ivp( self.model_dXdt_lambda,
                              [t_array[0],t_array[-1]],
                              self.ic,
                              method=self.method,
                              t_eval=t_array,
                              dense_output=self.do_dense,
                              #min_step=0, #max_step=np.inf,
                              # rtol=1e-3, atol=1e-6,
                              events=almost_reached_divide,
                              vectorized=False )

        # Process solution
        rp_t_soln = soln_ivp.y
        rx_array, rz_array = rp_t_soln[0],rp_t_soln[1]
        # Did we exceed the domain bounds?
        # If so, find the index of the first point out of bounds, otherwise set as None
        i_end = np.argwhere(rx_array>=self.parameters[x_1])[0][0] \
            if len(np.argwhere(rx_array>=self.parameters[x_1]))>0 else None
        # print("i_end:",i_end)
        if i_end is not None:
            if rx_array[0]!=self.parameters[x_1]:
                i_end = min(len(t_array),i_end)
            else:
                i_end = min(len(t_array),2)

        # Record solution
        rpt_lag_arrays = dict()
        if t_lag>0:
            dt = t_array[1]-t_array[0]
            n_lag = int(t_lag/dt)
            rpt_lag_arrays['t'] = np.linspace(0, t_lag, num=n_lag, endpoint=False)
            for rp_idx,rp_ in enumerate(rp_list):
                rpt_lag_arrays[rp_] = np.full(n_lag, rp_t_soln[rp_idx][0])
        else:
            n_lag = 0
            for rpt_idx,rpt_ in enumerate(rpt_list):
                rpt_lag_arrays[rpt_] = np.array([])

        # Report
        if self.verbose=='very':
            if i_end is not None:
                print('From {},{}: out of bounds @ i={} x={} t={}'
                          .format(rx_array[0],rz_array[0], n_lag+i_end if i_end is not None else len(t_array),
                                  rx_array[i_end], t_array[i_end]))
            else:
                print('From {},{}: terminating @ i={} x={} t={}'
                              .format(rx_array[0],rz_array[0], n_lag+i_end if i_end is not None else len(t_array),
                                  rx_array[-1], t_array[-1]))

        rpt_arrays = dict()
        rpt_arrays['t'] = np.concatenate((rpt_lag_arrays['t'],t_array[0:i_end]+t_lag))
        for rp_idx,rp_ in enumerate(rp_list):
            rpt_arrays[rp_] = np.concatenate((rpt_lag_arrays[rp_],rp_t_soln[rp_idx][0:i_end]))
        # print('solve_ODE_system:', rpt_arrays['rx'])

        return soln_ivp, rpt_arrays, (n_lag+i_end if i_end is not None else len(t_array))

    def postprocessing(self):
        self.rp_t_interp_fns = dict()
        [self.rp_t_interp_fns.update({rp_: [None]*self.n_rays}) for rp_ in rp_list]
        fill_value_ = 'extrapolate'
        for i_ray, t_array in enumerate(self.rpt_arrays['t']):
            if t_array.size>1:
                # Generate interpolation functions for each component rx[t], rz[t], px[t], pz[t]
                for rp_ in rp_list:
                    self.rp_t_interp_fns[rp_][i_ray] = interp1d(t_array,
                                                                self.rpt_arrays[rp_][i_ray],
                                                                kind=self.interp1d_kind,
                                                                fill_value=fill_value_,
                                                                assume_sorted=True)

    def resolve_isochrones( self, x_subset=1, t_isochrone_max=0.04, n_isochrones=25,
                            bounds_hack=100, tolerance=1e-3, do_eliminate_caustics=True, dont_crop_cusps=False ):
        """
        Resample the ensemble of rays at selected time slices to generate
        synchronous values of (r,p) aka isochrones.

        Each ray trajectory is numerically integrated independently, and so the integration points along one ray are
        not synchronous with those of any other ray. If we want to compare the (r,p) "positions" of the ray ensemble
        at a chosen time, we need to resample along all the rays at mutually consistent time slices.
        This is achieved by first interpolating along each rx[t], rz[t], pz[t], and pz[t] sequence, and then
        resampling along a reference time sequence.

        Two additional actions are taken:
           (1) termination of each resampled ray at the domain boundary at rx=x_1 (or a bit beyond, for better viz quality);
           (2) termination of any resampled ray that is overtaken by another ray at a cusp.
        """

        # Create isochrones of the evolving surface:
        #   (1) step through each time t_i in the global time sequence
        #   (2) for each ray, generate an (rx,rz,px,pz)[t_i] vector through
        #         interpolation of its numerically integrated sequence
        #   (3) combine these vectors into arrays for the {rx[t_i]}, {rz[t_i]}, {px[t_i]} and {pz[t_i]}
        #   (4) truncate a ray from the point where (if at all) its rx sequence reverses and goes left not right
        #   (5) also truncate if a ray leaves the domain, i.e., rx>x_1

        def prepare_isochrones():
            # Record important parameters
            self.t_isochrone_max = t_isochrone_max
            self.n_isochrones = n_isochrones
            self.x_subset = x_subset
            self.tolerance = tolerance if tolerance is not None else 1e-3

            # Prepare array dictionaries etc
            self.rpt_isochrones = dict()
            [self.rpt_isochrones.update({rpt_: [None]*n_isochrones}) for rpt_ in rpt_list]
            # self.rpt_isochrones_lowres = self.rpt_isochrones.copy()
            self.trxz_cusps = []

        def truncate_isochrone(rpt_isochrone, i_from=None, i_to=None):
            for rp_ in rp_list:
                rpt_isochrone[rp_] = rpt_isochrone[rp_][i_from:i_to]
            return rpt_isochrone

        def find_intercept(rpt_isochrone, slice1, slice2):
            x1_array = rpt_isochrone['rx'][slice1]
            x2_array = rpt_isochrone['rx'][slice2]
            z1_array = rpt_isochrone['rz'][slice1]
            z2_array = rpt_isochrone['rz'][slice2]
            px1_array = rpt_isochrone['px'][slice1]
            px2_array = rpt_isochrone['px'][slice2]
            pz1_array = rpt_isochrone['pz'][slice1]
            pz2_array = rpt_isochrone['pz'][slice2]
            s1_array = np.linspace(0,1,num=len(x1_array))
            s2_array = np.linspace(0,1,num=len(x2_array))

            # Arguments given to interp1d:
            #  - extrapolate: to make sure we don't get a fatal value error when fsolve searches
            #                 beyond the bounds of [0,1]
            #  - copy: use refs to the arrays
            #  - assume_sorted: because s_array ('x') increases monotonically across [0,1]

            kwargs_ = dict(fill_value='extrapolate', copy=False, assume_sorted=True)
            slice = np.s_[::1]
            x1_interp = interp1d(s1_array[slice],x1_array[slice], **kwargs_)
            x2_interp = interp1d(s2_array[slice],x2_array[slice], **kwargs_)
            z1_interp = interp1d(s1_array[slice],z1_array[slice], **kwargs_)
            z2_interp = interp1d(s2_array[slice],z2_array[slice], **kwargs_)
            px1_interp = interp1d(s1_array[slice],px1_array[slice], **kwargs_)
            px2_interp = interp1d(s2_array[slice],px2_array[slice], **kwargs_)
            pz1_interp = interp1d(s1_array[slice],pz1_array[slice], **kwargs_)
            pz2_interp = interp1d(s2_array[slice],pz2_array[slice], **kwargs_)

            xydiff_lambda = lambda s12: ((np.abs(x1_interp(s12[0])-x2_interp(s12[1]))),
                                         (np.abs(z1_interp(s12[0])-z2_interp(s12[1]))))

            s12_intercept,_,ier,mesg = fsolve(xydiff_lambda, [0.99, 0.01], factor=0.1, full_output=True) #, factor=0.1,
            xz1_intercept = x1_interp(s12_intercept[0]),z1_interp(s12_intercept[0])
            xz2_intercept = x2_interp(s12_intercept[1]),z2_interp(s12_intercept[1])
            pxz1_intercept = px1_interp(s12_intercept[0]),pz1_interp(s12_intercept[0])
            pxz2_intercept = px2_interp(s12_intercept[1]),pz2_interp(s12_intercept[1])
            #print(ier, mesg, s12_intercept, s12_intercept)
            # self.x1_interp = x1_interp
            # self.x2_interp = x2_interp
            # self.z1_interp = z1_interp
            # self.z2_interp = z2_interp
            # self.xydiff_lambda = xydiff_lambda

            return xz1_intercept, xz2_intercept, pxz1_intercept, pxz2_intercept

        def eliminate_caustic(is_good_pt_array, rpt_isochrone):
            # Locate and remove caustic and render into a cusp - if there is one

            # Check if there are any false points - if not, just return the whole curve
            if len(rpt_isochrone['rx'][is_good_pt_array])==len(rpt_isochrone['rx']):
                #print('whole curve')
                return rpt_isochrone, (None,None,None)

            # Check if there are ONLY false points, in which case return an empty curve
            elif len(rpt_isochrone['rx'][is_good_pt_array])==0:
                #print('empty curve')
                return truncate_isochrone(rpt_isochrone,-1,-1), (None,None,None)

            # Find false indexes
            false_indexes = np.where(is_good_pt_array==False)[0]
            # If false_indexes[0]==0, there's no left curve, so use only the right half
            if false_indexes[0]==0:
                #print('right half')
                return truncate_isochrone(rpt_isochrone,false_indexes[-1],None), (None,None,None)

            if len(false_indexes)>1:
                last_false_index = [idx for idx,fi in enumerate(zip(false_indexes[:-1],false_indexes[1:]))
                                    if fi[1]-fi[0]>1]
                #print('last:', last_false_index)
                if len(last_false_index)>0:
                    false_indexes = false_indexes[0:(last_false_index[0]+1)]

            # Otherwise, generate interpolations of x and y points for both left and right curves
            slice1 = np.s_[:(false_indexes[0]+1)]
            slice2 = np.s_[(false_indexes[-1]+1):]

            # If false_indexes[-1]==len(false_indexes)-1, use only left half (won't happen?)
            #   (assumes pattern is TTFFFFTT or TTTTFFFF etc)
            if false_indexes[-1]==len(is_good_pt_array)-1 or len(is_good_pt_array[slice2])==1:
                #print('left half')
                return truncate_isochrone(rpt_isochrone,0,false_indexes[0]), (None,None,None)

            # At this point, we presumably have a caustic, so find the cusp intercept
            rxz1_intercept, rxz2_intercept, pxz1_intercept, pxz2_intercept = find_intercept(rpt_isochrone, slice1, slice2)
            #print('Intercept @ ', rxz1_intercept,rxz2_intercept)

            # Delimit the portions to the left and to the right of the cusp
            i1_array = np.where(rpt_isochrone['rx'][slice1]<=rxz1_intercept[0])[0]
            i2_array = np.where(rpt_isochrone['rx'][slice2]>=rxz1_intercept[0])[0]
            #print('i1,i2 arrays', i1_array,i2_array)

            # Rebuild the isochrone dictionary
            rpt_isochrone_rtn = dict()
            for rp_ in rp_list:
                isochrone_left  = rpt_isochrone[rp_][slice1]
                isochrone_right = rpt_isochrone[rp_][slice2]
                isochrone_ = np.concatenate([isochrone_left[i1_array],isochrone_right[i2_array]])
                rpt_isochrone_rtn.update({rp_: isochrone_})
            rpt_isochrone_rtn.update({'t':rpt_isochrone['t']})

            return rpt_isochrone_rtn, \
                ((rpt_isochrone['t'],np.array(rxz1_intercept)),np.array(pxz1_intercept),np.array(pxz2_intercept)) \
                    if rxz1_intercept[0]>0 and (dont_crop_cusps or rxz1_intercept[0]<=self.parameters[x_1]) \
                            else (None,None,None)

        def compose_isochrone(i_isochrone, t_):
            # Prepare a tmp dictionary for the rx,rz,px,pz component arrays delineating this isochrone
            rpt_isochrone = dict()
            for rp_ in rp_list:
                # Sample each ray at time t_ from each of the components of r,p vectors using their interpolating fns
                rp_interpolated_isochrone = [float(interp_fn(t_)) for interp_fn in self.rp_t_interp_fns[rp_]
                                            if interp_fn is not None]
                rpt_isochrone.update( {rp_: np.array(rp_interpolated_isochrone)} )
            rpt_isochrone.update({'t':t_})
            return rpt_isochrone

        def resample_isochrone(rpt_isochrone_in):
            n_s_pts = 301
            rpt_isochrone_out = rpt_isochrone_in.copy()
            s_array = np.cumsum( np.concatenate([np.array([0]),np.sqrt((rpt_isochrone_in['rx'][1:]-rpt_isochrone_in['rx'][:-1])**2
                                                             + (rpt_isochrone_in['rz'][1:]-rpt_isochrone_in['rz'][:-1])**2)]) )
            rs_array = np.linspace(s_array[0],s_array[-1],num=n_s_pts)
            if s_array[-1] < 1e-10: return rpt_isochrone_out    # HACK to avoid interpolating a crap isochrone sampled only at one point
            for rp_ in rp_list:
                rpt_isochrone_interp_fn = interp1d(s_array, rpt_isochrone_out[rp_], kind='linear',
                                                  fill_value='extrapolate', assume_sorted=True)
                rpt_isochrone_out[rp_] = rpt_isochrone_interp_fn(rs_array)
            return rpt_isochrone_out

        # Crop out-of-bounds points and points on caustics for the current isochrone

        def clean_isochrone(rpt_isochrone):
            # Cusps: eliminate all points beyond the first whose rx sequence is negative (left-ward)
            #   - do this by creating a flag array whose elements are True if rx[i+1]>rx[i]
            is_good_pt_array = np.concatenate(((rpt_isochrone['rx'][1:]-rpt_isochrone['rx'][:-1]>=-self.tolerance),[True]))

            # Replace this isochrone with one whose caustic has been removed - if it has one
            # rpt_isochrone_try, (t_rxz_intercept, pxz1_intercept, pxz2_intercept) \
            #         = eliminate_caustic(is_good_pt_array, rpt_isochrone) if do_eliminate_caustics else None, (None,None,None)
            if do_eliminate_caustics:
                rpt_isochrone, (t_rxz_intercept, pxz1_intercept, pxz2_intercept) \
                        = eliminate_caustic(is_good_pt_array, rpt_isochrone)
            else:
                rpt_isochrone, (t_rxz_intercept, pxz1_intercept, pxz2_intercept) = None, (None,None,None)
            # print(rpt_isochrone_try)
            # rpt_isochrone = rpt_isochrone_try if rpt_isochrone_try is not None else rpt_isochrone
            return rpt_isochrone, (t_rxz_intercept, pxz1_intercept, pxz2_intercept)

        def prune_isochrone(rpt_isochrone_in):
            rpt_isochrone_out = rpt_isochrone_in.copy()
            rx_array = rpt_isochrone_out['rx']
            i_bounded_array = (rx_array>=0) & (rx_array<=1.0)
            rpt_isochrone_out['rx'] = rx_array[i_bounded_array]
            did_clip_at_x1 = len(rx_array)>len(rpt_isochrone_out['rx'])
            for rp_ in rp_list:
                rpt_isochrone_out[rp_] = rpt_isochrone_in[rp_][i_bounded_array]
                # Wildly inefficient interpolation to do extrapolation here
                if did_clip_at_x1:
                    rpt_isochrone_interp_fn = interp1d(rx_array[rx_array>=0.95], rpt_isochrone_in[rp_][rx_array>=0.95],
                                                      kind='linear', fill_value='extrapolate', assume_sorted=True)
                    rpt_at_x1 = rpt_isochrone_interp_fn(1.0)
                    rpt_isochrone_out[rp_] = np.concatenate([rpt_isochrone_out[rp_],np.array([rpt_at_x1])])
            return rpt_isochrone_out

        def record_isochrone(rpt_isochrone):
            # Record this isochrone
            for rpt_ in rpt_list:
                self.rpt_isochrones[rpt_][i_isochrone] = rpt_isochrone[rpt_]

        def record_cusp(trxz_cusp, pxz1_intercept, pxz2_intercept):
            if trxz_cusp is not None:
                # Record this intercept
                self.trxz_cusps.append((trxz_cusp, pxz1_intercept, pxz2_intercept))

        def organize_cusps():
            self.cusps = dict()
            self.cusps['t']    = np.array( [t_    for (t_,rxz_),pxz1_,pxz2_ in self.trxz_cusps if rxz_[0]>=0 and rxz_[0]<=1] )
            self.cusps['rxz']  = np.array( [rxz_  for (t_,rxz_),pxz1_,pxz2_ in self.trxz_cusps if rxz_[0]>=0 and rxz_[0]<=1] )
            self.cusps['pxz1'] = np.array( [pxz1_ for (t_,rxz_),pxz1_,pxz2_ in self.trxz_cusps if rxz_[0]>=0 and rxz_[0]<=1] )
            self.cusps['pxz2'] = np.array( [pxz2_ for (t_,rxz_),pxz1_,pxz2_ in self.trxz_cusps if rxz_[0]>=0 and rxz_[0]<=1] )

        def coarsen_isochrone(rpt_isochrone):
            # Reduce the time resolution of the isochrone points to make plotting less cluttered
            for rp_ in rp_list:
                self.rpt_isochrones_lowres[rp_][i_isochrone] \
                    = [ [element_ for idx,element_ in enumerate(array_) if (idx//x_subset-idx/x_subset)==0]
                        for array_ in [rpt_isochrone[rp_]] ][0]
            self.rpt_isochrones_lowres['t'][i_isochrone] = rpt_isochrone['t']

        # Time sequence - with resolution n_isochrones and limit t_isochrone_max

        prepare_isochrones()
        t_array = np.linspace(0,t_isochrone_max,n_isochrones)
        for i_isochrone,t_ in enumerate(t_array):
            rpt_isochrone = compose_isochrone(i_isochrone, t_)
            rpt_isochrone = resample_isochrone(rpt_isochrone)
            rpt_isochrone, (trxz_cusp, pxz1_intercept, pxz2_intercept) = clean_isochrone(rpt_isochrone)
            if rpt_isochrone is not None:
                rpt_isochrone = prune_isochrone(rpt_isochrone)
                record_isochrone(rpt_isochrone)
                record_cusp(trxz_cusp, pxz1_intercept, pxz2_intercept)
        organize_cusps()

    def measure_cusp_propagation(self):
        kwargs_ = dict(kind='linear', fill_value='extrapolate', assume_sorted=True)

        if self.cusps['t'].shape[0]==0:
            self.cx_pz_tanbeta_lambda = None
            self.cx_pz_lambda = None
            self.cx_v_lambda = None
            self.vx_interp_fast = None
            self.vx_interp_slow = None
            return
        t_fast_array, rxz_array, pxz_fast_array, pxz_slow_array = [self.cusps[key_] \
                                                                    for key_ in ['t','rxz','pxz1','pxz2']]
        px_fast_array, pz_fast_array = pxz_fast_array[:,0],pxz_fast_array[:,1]
        px_slow_array, pz_slow_array = pxz_slow_array[:,0],pxz_slow_array[:,1]
        rx_array, rz_array = rxz_array[:,0],rxz_array[:,1]

        p_fast_array = np.sqrt(px_fast_array**2+pz_fast_array**2)
        tanbeta_fast_array = -px_fast_array/pz_fast_array
        sinbeta_fast_array =  px_fast_array/p_fast_array
        cosbeta_fast_array = -pz_fast_array/p_fast_array
        p_fast_interp       = interp1d( rxz_array[:,0], p_fast_array,       **kwargs_ )
        px_fast_interp      = interp1d( rxz_array[:,0], px_fast_array,      **kwargs_ )
        pz_fast_interp      = interp1d( rxz_array[:,0], pz_fast_array,      **kwargs_ )
        tanbeta_fast_interp = interp1d( rxz_array[:,0], tanbeta_fast_array, **kwargs_ )
        sinbeta_fast_interp = interp1d( rxz_array[:,0], sinbeta_fast_array, **kwargs_ )
        cosbeta_fast_interp = interp1d( rxz_array[:,0], cosbeta_fast_array, **kwargs_ )

        p_slow_array = np.sqrt(px_slow_array**2+pz_slow_array**2)
        tanbeta_slow_array = -px_slow_array/pz_slow_array
        sinbeta_slow_array =  px_slow_array/p_slow_array
        cosbeta_slow_array = -pz_slow_array/p_slow_array
        p_slow_interp       = interp1d( rxz_array[:,0], p_slow_array,       **kwargs_ )
        px_slow_interp      = interp1d( rxz_array[:,0], px_slow_array,      **kwargs_ )
        pz_slow_interp      = interp1d( rxz_array[:,0], pz_slow_array,      **kwargs_ )
        tanbeta_slow_interp = interp1d( rxz_array[:,0], tanbeta_slow_array, **kwargs_ )
        sinbeta_slow_interp = interp1d( rxz_array[:,0], sinbeta_slow_array, **kwargs_ )
        cosbeta_slow_interp = interp1d( rxz_array[:,0], cosbeta_slow_array, **kwargs_ )

        # Provide a lambda fn that returns the horizontal cusp velocity component cx
        #   using interpolation functions for all the variables for which we have sampled solutions as arrays
        # sinbeta_diff_lambda = lambda x_: sinbeta_fast_interp(x_)*cosbeta_slow_interp(x_) \
        #                                 - cosbeta_fast_interp(x_)*sinbeta_slow_interp(x_)
        # tanbeta_diff_lambda = lambda x_: tanbeta_fast_interp(x_) - tanbeta_slow_interp(x_)
        self.cx_pz_tanbeta_lambda = lambda x_: (
            - (1/pz_fast_interp(x_) - 1/pz_slow_interp(x_))
            / (tanbeta_fast_interp(x_) - tanbeta_slow_interp(x_))
        )
        self.cx_pz_lambda = lambda x_: (
                                ( pz_slow_interp(x_) - pz_fast_interp(x_) )
            / (px_fast_interp(x_)*pz_slow_interp(x_) - pz_fast_interp(x_)*px_slow_interp(x_))
        )
        rpdot_fast_array = np.array([self.model_dXdt_lambda(0, rp_)
                                     for rp_ in zip(rx_array,rz_array,px_fast_array,pz_fast_array)])
        rpdot_slow_array = np.array([self.model_dXdt_lambda(0, rp_)
                                     for rp_ in zip(rx_array,rz_array,px_slow_array,pz_slow_array)])
        vx_interp_fast = interp1d( rx_array, rpdot_fast_array[:,0], **kwargs_ )
        vx_interp_slow = interp1d( rx_array, rpdot_slow_array[:,0], **kwargs_ )
        vz_interp_fast = interp1d( rx_array, rpdot_fast_array[:,1], **kwargs_ )
        vz_interp_slow = interp1d( rx_array, rpdot_slow_array[:,1], **kwargs_ )
        self.cx_v_lambda = lambda x_: (
            ( (tanbeta_fast_interp(x_)*vx_interp_fast(x_) - vz_interp_fast(x_))
                    - (tanbeta_slow_interp(x_)*vx_interp_slow(x_) - vz_interp_slow(x_)) )
            / (tanbeta_fast_interp(x_) - tanbeta_slow_interp(x_))
        )
        self.vx_interp_fast = vx_interp_fast
        self.vx_interp_slow = vx_interp_slow

    def save(self, rpt_arrays, idx):
        for rpt_ in rpt_list:
            self.rpt_arrays[rpt_][idx] = rpt_arrays[rpt_]
        rx_length = len(self.rpt_arrays['rx'][idx])
        self.rpt_arrays['t'][idx] = self.rpt_arrays['t'][idx][:rx_length]

    def report_progress(self, i, n, progress_was=0, pc_step=1, is_initial_step=False):
        progress_now = 100*np.round((100/pc_step)*i/(n-1 if n>1 else 1))/np.round(100/pc_step)
        if self.verbose:
            if progress_now>progress_was or is_initial_step:
                print('{:0.0f}% '.format(progress_now), end='' if progress_now<100 else '\n')
        return progress_now





#

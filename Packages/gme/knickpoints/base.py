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
from gme.ode.base import BaseSolution
from gme.ode.velocity_boundary import VelocityBoundarySolution
from gme.core.symbols import *
from sympy import N, sign, atan, atan2, sin, cos, tan, re, im, sqrt, \
    Matrix, lambdify, Abs, simplify, expand, solve, Eq, Rational, diff, \
    nroots, poly
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import root_scalar, fsolve

rp_list = ['rx','rz','px','pz']
rpt_list = rp_list+['t']

__all__ = [ 'InitialProfileSolution', 'InitialCornerSolution', 'CompositeSolution']


class InitialProfileSolution(BaseSolution):
    """
    Integration of Hamilton's equations (ODEs) from an initial topographic profile.
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

        # Initial condition equations
        self.rz_initial_surface_eqn = gmeq.rz_initial_eqn.subs({rx:x})

    def initial_conditions(self, x_):
        rx0_ = x_
        rz0_ = float(self.rz_initial_surface_eqn.rhs.subs({x:x_}).subs(self.parameters))
        px0_ = float(self.px_initial_surface_eqn.rhs.subs({x:x_}).subs(self.parameters))
        pz0_ = float(self.pz_initial_surface_eqn.rhs.subs({x:x_}).subs(self.parameters))
        return [rx0_,rz0_,px0_,pz0_]

    def solve(self, report_pc_step=1):
        self.prep_arrays()
        x_1_ = float(x_1.subs(self.parameters))
        #print(self.t_end)
        pc_progress = self.report_progress(i=0, n=self.n_rays, is_initial_step=True)
        # Points along the initial profile boundary
        for idx, x_ in enumerate(np.linspace(0,x_1_,self.n_rays, endpoint=True)):
            #if self.verbose: print('{}: {}'.format(idx,np.round(x_,4)),end='\t')
            self.ic = self.initial_conditions(x_)
            print(f'Initial conditions {idx}: {self.ic}')
            self.model_dXdt_lambda = self.make_model()
            rpt_arrays = self.solve_Hamiltons_equations(t_array=self.ref_t_array.copy())
            self.save(rpt_arrays, idx)
            pc_progress = self.report_progress(i=idx, n=self.n_rays,
                                               pc_step=report_pc_step, progress_was=pc_progress)
        self.report_progress(i=idx, n=self.n_rays, pc_step=report_pc_step, progress_was=pc_progress)


class InitialCornerSolution(BaseSolution):
    """
    Integration of Hamilton's equations (ODEs) from a 'corner' point.

    Provides a set of ray integrations to span solutions along an initial profile
    and solutions from a slip velocity boundary.
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
        self.px_initial_corner_eqn = self.gmeq.px_varphi_rx_beta_eqn.subs({rx:0})
        self.pz_initial_corner_eqn = self.gmeq.pz_varphi_rx_beta_eqn.subs({rx:0})
        self.beta_surface_corner \
             = float(atan2(self.gmeq.px_initial_eqn.rhs.subs({x:0}).subs(self.parameters),
                          -self.gmeq.pz_initial_eqn.rhs.subs({x:0}).subs(self.parameters)))
        pz_velocity_corner = self.pz_velocity_boundary_eqn
        # HACK!!!  doing this to set up px0_poly_eqn
        _ = self.pxpz0_from_xiv0()
        px_velocity_corner = self.px_value(x_=0, pz_=pz_velocity_corner, parameters=self.parameters)
        self.beta_velocity_corner = float(atan2(px_velocity_corner,-pz_velocity_corner))
        if self.choice=='geodesic':
            self.rdot = Matrix([ self.gmeq.hamiltons_eqns[0].rhs.subs(parameters),
                                 self.gmeq.hamiltons_eqns[1].rhs.subs(parameters) ])
        else:
            self.rdot = None

    def initial_conditions(self, beta0_):
        rx0_ = 0
        rz0_ = 0
        px0_ = float( self.px_initial_corner_eqn.rhs.subs({beta:beta0_}).subs(self.parameters) )
        pz0_ = float( self.pz_initial_corner_eqn.rhs.subs({beta:beta0_}).subs(self.parameters) )
        if self.choice=='Hamilton':
            return [rx0_,rz0_,px0_,pz0_]
        else:
            return [rx0_,rz0_,*self.rdot.subs({rx:rx0_,px:px0_,pz:pz0_})]

    def solve(self, report_pc_step=1, verbose=True):
        vprint(self.verbose, 'Solving Hamilton\'s equations' if self.choice=='Hamilton'
                        else 'Solving geodesic equations')
        self.prep_arrays()
        # Surface tilt angles spanning velocity b.c. to initial surface b.c.
        pc_progress = self.report_progress(i=0, n=self.n_rays, is_initial_step=True)
        for idx, beta0_ in enumerate(np.linspace(self.beta_velocity_corner,
                                                 self.beta_surface_corner,
                                                 self.n_rays, endpoint=True)):
            if self.verbose=='very': print('{}: {}'.format(idx,np.round(beta0_,4)),end='\t')
            self.ic = self.initial_conditions(beta0_)
            self.model_dXdt_lambda = self.make_model( do_verbose=False )
            if self.customize_t_fn is not None:
                t_array = self.ref_t_array.copy()
                t_limit = self.customize_t_fn(t_array[-1], self.ic[3]/self.ic[2])
                t_array = t_array[t_array<=t_limit]
                # print(f't_end = {t_array[-1]}')
            else:
                t_array = self.ref_t_array.copy()
            rpt_arrays = self.solve_Hamiltons_equations(t_array=t_array)
            # print(list(zip(rpt_arrays['rx'])))
            self.save(rpt_arrays, idx)
            pc_progress = self.report_progress(i=idx, n=self.n_rays, pc_step=report_pc_step, progress_was=pc_progress)
        self.report_progress(i=idx, n=self.n_rays, pc_step=report_pc_step, progress_was=pc_progress)


class CompositeSolution(BaseSolution):
    """
    Combine rays integrations from (1) initial profile, (2) initial corner, and (3) velocity boundary
    to generate a 'complete' ray integration solution from a complex topographic boundary.
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

    def create_solutions( self, t_end=0.04, t_slip_end=0.08,
                          do_solns=dict(ip=True,ic=True,vb=True),
                          n_rays=dict(ip=101,ic=31,vb=101),
                          n_t=dict(ip=1001,ic=1001,vb=1001) ):
        self.do_solns = do_solns
        self.t_end = t_end
        self.t_slip_end = t_slip_end

        self.ips = InitialProfileSolution( self.gmeq, parameters=self.parameters,
                                           method=self.method, do_dense=self.do_dense,
                                           t_end=self.t_end,
                                           n_rays=n_rays['ip'], n_t=n_t['ip'] ) \
                                           if do_solns['ip'] else None

        self.ics = InitialCornerSolution( self.gmeq, parameters=self.parameters,
                                          method=self.method, do_dense=self.do_dense,
                                          t_end=self.t_end,
                                          n_rays=n_rays['ic'], n_t=n_t['ic'] ) \
                                          if do_solns['ic'] else None

        #t_slip_end = t_scale*1.5 if t_slip_end is None else t_slip_end
        self.vbs = VelocityBoundarySolution( self.gmeq, parameters=self.parameters,
                                             method=self.method, do_dense=self.do_dense,
                                             t_end=self.t_end,
                                             t_slip_end=self.t_slip_end,
                                             n_rays=n_rays['vb'], n_t=n_t['vb'] ) \
                                             if do_solns['vb'] else None

    def solve(self):
        self.ips.solve() if self.do_solns['ip'] else False
        self.ics.solve() if self.do_solns['ic'] else False
        self.vbs.solve() if self.do_solns['vb'] else False
        self.t_ensemble_max =      self.vbs.t_ensemble_max if self.do_solns['vb'] \
                              else self.ics.t_ensemble_max if self.do_solns['ic'] \
                              else self.ips.t_ensemble_max if self.do_solns['ip'] \
                              else None
        self.model_dXdt_lambda =   self.vbs.model_dXdt_lambda if self.do_solns['vb'] \
                              else self.ics.model_dXdt_lambda if self.do_solns['ic'] \
                              else self.ips.model_dXdt_lambda if self.do_solns['ip'] \
                              else None


    def merge_rays(self):
        # Combine all three solutions such that rays are all in rz order
        self.rpt_arrays = dict()
        for rpt_ in rpt_list:
            vbs_arrays = [] if self.vbs is None else self.vbs.rpt_arrays[rpt_][:-1]
            ics_arrays = [] if self.ics is None else self.ics.rpt_arrays[rpt_]
            ips_arrays = [] if self.ips is None else self.ips.rpt_arrays[rpt_]
            self.rpt_arrays.update({ rpt_: vbs_arrays + ics_arrays + ips_arrays })
        # Eliminate empty rays
        rx_arrays = self.rpt_arrays['rx'].copy()
        for rpt_ in rpt_list:
            self.rpt_arrays[rpt_] = [rpt_array for rx_array, rpt_array in zip(rx_arrays, self.rpt_arrays[rpt_] )
                                                                              if len(rx_array)>=0 ]
        self.n_rays = len(self.rpt_arrays['t'])

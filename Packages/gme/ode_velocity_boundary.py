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
from typing import Tuple, List #, Any, List #, Callable, List #, Dict, Any, Optional

# Numpy
import numpy as np

# GME
from gme.equations import pxpz0_from_xiv0
from gme.ode_base import ExtendedSolution
from gme.symbols import xiv_0, Lc

warnings.filterwarnings("ignore")

rp_list = ['rx','rz','px','pz']
rpt_list = rp_list+['t']

__all__ = ['VelocityBoundarySolution']


class VelocityBoundarySolution(ExtendedSolution):
    """
    Integration of Hamilton's equations (ODEs) from a 'fault slip' velocity boundary.

    Currently the velocity boundary is required to lie along the left domain edge and to be vertical.
    """
    def initial_conditions(self, t_lag, xiv_0_, px_guess=1) -> Tuple[float,float,float,float]:
        """
        TBD
        """
        self.parameters[xiv_0] = xiv_0_
        # px0_, pz0_ = self.pxpz0_from_xiv0()
        px0_, pz0_ = pxpz0_from_xiv0(self.parameters, self.gmeq.pz_xiv_eqn,
                                     self.gmeq.poly_px_xiv0_eqn, px_guess=px_guess )
        cosbeta_ = np.sqrt(1/(1+(np.float(px0_/-pz0_))**2))
        rz0_ = t_lag/(pz0_*cosbeta_)
        rx0_ = 0.0
        return (rx0_,rz0_,px0_,pz0_)

    def solve(self, report_pc_step=1) -> None:
        """
        TBD
        """
        # self.prep_arrays()
        self.t_ensemble_max = 0.0

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
        t_lag_list: List[float] = [0.0]*n_rays
        xiv0_list: List[float] = [0.0]*n_rays
        ic_list: List[Tuple[float,float,float,float]] = [(0.0, 0.0, 0.0, 0.0)]*n_rays
        prev_t_lag = 0.0
        for (n_block_rays, (tp_,xiv0_), prev_rz0, prev_n_block_rays) \
                in zip(n_block_rays_array,
                       self.tp_xiv0_list,
                       offset_rz0_cumsum_array,
                       offset_n_block_rays_array):
            # Generate initial conditions for all the rays in this block
            for i_ray in list(range(0,n_block_rays)):
                t_lag = (i_ray/(n_block_rays-1))*self.t_slip_end*tp_
                rx0_,rz0_,px0_,pz0_ = self.initial_conditions(t_lag, xiv0_)
                t_lag_list[i_ray+prev_n_block_rays] = prev_t_lag+t_lag
                xiv0_list[i_ray+prev_n_block_rays] = xiv0_
                ic_list[i_ray+prev_n_block_rays] = (rx0_, rz0_+prev_rz0, px0_, pz0_)
            prev_t_lag += t_lag

        # Generate rays in reverse order so that the first ray is topographically the lowest
        pc_progress = self.report_progress(i=0, n=n_rays, is_initial_step=True)
        self.ic_list = ic_list # to be replaced by reordered sequence
        self.ivp_solns_list = [None]*n_rays
        self.model_dXdt_lambda = self.make_model()  # only do this here if all xiv_0 are equal
        for i_ray in list(range(0,n_rays)):
            t_lag = t_lag_list[n_rays-1-i_ray]
            xiv0_ = xiv0_list[n_rays-1-i_ray]
            self.ic_list[i_ray] = ic_list[n_rays-1-i_ray]
            self.parameters[xiv_0] = xiv0_
            # Start rays from the bottom of the velocity boundary and work upwards
            #   so that their x,z,t disposition is consistent with initial profile, initial corner
            # if self.choice=='Hamilton':
            parameters_ = {Lc: self.parameters[Lc]}
            ivp_soln, rpt_arrays = self.solve_Hamiltons_equations( model=self.model_dXdt_lambda,
                                                                   method=self.method,
                                                                   do_dense=self.do_dense,
                                                                   ic=self.ic_list[i_ray],
                                                                   parameters=parameters_,
                                                                   t_array=self.ref_t_array.copy(),
                                                                   x_stop=self.x_stop,
                                                                   t_lag=t_lag )
            # else:
            #     ivp_soln, rpt_arrays = self.solve_Hamiltons_equations( ic=self.ic_list[i_ray],
            #                                                            t_array=self.ref_t_array.copy(),
            #                                                            t_lag=t_lag )
            self.ivp_solns_list[i_ray] = ivp_soln
            self.t_ensemble_max = max(self.t_ensemble_max, rpt_arrays['t'][-1])
            self.save(rpt_arrays, i_ray)
            pc_progress = self.report_progress(i=i_ray, n=self.n_rays,
                                               pc_step=report_pc_step, progress_was=pc_progress)
        # self.report_progress(i=n_rays, n=n_rays, pc_step=report_pc_step, progress_was=pc_progress)




#

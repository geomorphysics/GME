"""
---------------------------------------------------------------------

Generate a sequence of topographic profiles, using ray tracing
aka ODE integration of Hamilton's equations, for a velocity-boundary condition.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`NumPy <numpy>`
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
from typing import Tuple, List, Optional

# Numpy
import numpy as np

# GME
from gme.core.symbols import xiv_0, xih_0, Lc
from gme.ode.extended import ExtendedSolution
from gme.ode.solve import solve_Hamiltons_equations, report_progress

warnings.filterwarnings("ignore")

__all__ = ['VelocityBoundarySolution']


class VelocityBoundarySolution(ExtendedSolution):
    """
    Integration of Hamilton's equations (ODEs) from a 'fault slip'
    velocity boundary.

    Currently the velocity boundary is required to lie along the
    left domain edge and to be vertical.
    """

    def initial_conditions(self, t_lag, xiv_0_) \
            -> Tuple[float, float, float, float]:
        """
        TBD
        """
        # self.parameters[xiv_0] = xiv_0_
        # px0_, pz0_ = pxpz0_from_xiv0( self.parameters,
        #                               #xiv_0_, self.parameters[xih_0],
        #                               self.gmeq.pz_xiv_eqn,
        #                               self.gmeq.poly_px_xiv0_eqn )
        pz0_: float = (-1/xiv_0_)
        px0_: float = ((xiv_0/xih_0).subs(self.parameters))/xiv_0_
        # print(pz0_,px0_)
        cosbeta_ = np.sqrt(1/(1+(float(px0_/-pz0_))**2))
        rz0_: float = t_lag/(pz0_*cosbeta_)
        rx0_: float = 0.0
        return (rx0_, rz0_, px0_, pz0_)

    def solve(self, report_pc_step=1) -> None:
        """
        TBD
        """

        # self.prep_arrays()
        self.t_ensemble_max = 0.0

        # Construct a list of % durations and vertical velocities
        #   if only xiv_0 given
        self.tp_xiv0_list: Optional[List[Tuple[float, float]]] \
            = [(1, self.parameters[xiv_0])] if self.tp_xiv0_list is None else \
            self.tp_xiv0_list

        # Calculate vertical distances spanned by each tp, uv0
        rz0_array \
            = np.array([self.initial_conditions(tp_*self.t_slip_end, xiv0_)[1]
                        for (tp_, xiv0_) in self.tp_xiv0_list])
        rz0_cumsum_array = np.cumsum(rz0_array)
        offset_rz0_cumsum_array = np.concatenate(
            [np.array([0]), rz0_cumsum_array])[:-1]
        # The total vertical distance spanned by all initial rays
        #   is rz0_cumsum_array[-1]
        rz0_total = rz0_cumsum_array[-1]

        # Apportion numbers of rays based on rz0 proportions
        n_block_rays_array = np.array([int(round(self.n_rays*(rz0_/rz0_total)))
                                       for rz0_ in rz0_array])
        offset_n_block_rays_array = np.concatenate(
            [np.array([0]), n_block_rays_array])[:-1]
        self.n_rays: int = np.sum(n_block_rays_array)
        n_rays: int = self.n_rays
        # assert(len(self.tp_xiv0_list)==len(n_block_rays_array))

        # Step through each "block" of rays tied to a different
        #   boundary velocity and generate an initial condition for each ray
        t_lag_list: List[float] = [0.0]*n_rays
        xiv0_list: List[float] = [0.0]*n_rays
        ic_list: List[Tuple[float, float, float, float]] = [
            (0.0, 0.0, 0.0, 0.0)]*n_rays
        prev_t_lag = 0.0
        for (n_block_rays, (tp_, xiv0_), prev_rz0, prev_n_block_rays) \
                in zip(n_block_rays_array, self.tp_xiv0_list,
                       offset_rz0_cumsum_array, offset_n_block_rays_array):
            # Generate initial conditions for all the rays in this block
            for i_ray in list(range(0, n_block_rays)):
                t_lag = (i_ray/(n_block_rays-1))*self.t_slip_end*tp_
                rx0_, rz0_, px0_, pz0_ = self.initial_conditions(t_lag, xiv0_)
                t_lag_list[i_ray+prev_n_block_rays] = prev_t_lag+t_lag
                xiv0_list[i_ray+prev_n_block_rays] = xiv0_
                ic_list[i_ray + prev_n_block_rays] \
                    = (rx0_, rz0_+prev_rz0, px0_, pz0_)
            prev_t_lag += t_lag
        for ic_ in ic_list:
            logging.debug(f'ode.vb.solve: {ic_}')

        # Generate rays in reverse order so that the first ray is
        #     topographically the lowest
        pc_progress = report_progress(i=0, n=n_rays, is_initial_step=True)
        self.ic_list = [(0.0, 0.0, 0.0, 0.0)]*n_rays
        self.ivp_solns_list = [None]*n_rays
        xiv0_prev = 0.0
        model_dXdt_lambda_prev = None
        for i_ray in list(range(0, n_rays)):
            t_lag = t_lag_list[n_rays-1-i_ray]
            xiv0_ = xiv0_list[n_rays-1-i_ray]
            self.ic_list[i_ray] = ic_list[n_rays-1-i_ray]
            self.parameters[xiv_0] = xiv0_
            model_dXdt_lambda \
                = self.make_model() \
                if model_dXdt_lambda_prev is None or xiv0_prev != xiv0_\
                else model_dXdt_lambda_prev
            # print(f'i_ray={i_ray}  t_lag={t_lag}  xiv0_={xiv0_}  \
            #      {bool(xiv0_==xiv0_prev)}
            #          {bool(model_dXdt_lambda==model_dXdt_lambda_prev)}',
            #            flush=True)
            # Start rays from the bottom of the velocity boundary and work
            #   upwards so that their x,z,t disposition is consistent with
            #         initial profile, initial corner
            # if self.choice=='Hamilton':
            parameters_ = {Lc: self.parameters[Lc]}
            logging.debug(f'ode.vb.solve: calling solver: t_lag={t_lag}')
            ivp_soln, rpt_arrays \
                = solve_Hamiltons_equations(model=model_dXdt_lambda,
                                            method=self.method,
                                            do_dense=self.do_dense,
                                            ic=self.ic_list[i_ray],
                                            parameters=parameters_,
                                            t_array=self.ref_t_array.copy(),
                                            x_stop=self.x_stop,
                                            t_lag=t_lag)
            self.ivp_solns_list[i_ray] = ivp_soln
            self.t_ensemble_max = max(self.t_ensemble_max, rpt_arrays['t'][-1])
            self.save(rpt_arrays, i_ray)
            # logging.debug(f"ode.velocity_boundary.solve: {rpt_arrays['rx']}")
            pc_progress \
                = report_progress(i=i_ray,
                                  n=self.n_rays,
                                  pc_step=report_pc_step,
                                  progress_was=pc_progress)
            xiv0_prev, model_dXdt_lambda_prev = xiv0_, model_dXdt_lambda
            # self.report_progress(i=n_rays, n=n_rays,
            # pc_step=report_pc_step,progress_was=pc_progress)


#

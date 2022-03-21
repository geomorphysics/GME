"""

Single-ray tracing through ODE integration of Hamilton's equations.

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
from typing import Tuple, List, Callable, Any

# Numpy
import numpy as np

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import Lc
from gme.core.utils import pxpz0_from_xiv0
from gme.ode.extended import ExtendedSolution
from gme.ode.base import rpt_tuple
from gme.ode.solve import solve_Hamiltons_equations

# SciPy
from scipy.interpolate import InterpolatedUnivariateSpline

warnings.filterwarnings("ignore")

__all__ = ["SingleRaySolution"]


class SingleRaySolution(ExtendedSolution):
    """
    Trace a single-ray by ODE integration of Hamilton's equations.

    Subclasses :class:`gme.ode.base.ExtendedSolution`.
    """

    # Definitions
    ic_list: List[Tuple[float, float, float, float]]
    # HACK: the Any lambda declarations should all be Callable decls but
    #       Python 3.8/mypy have a bug assigning callables to class variables
    # See: https://github.com/python/mypy/issues/708
    model_dXdt_lambda: Any
    ivp_solns_list: List
    pz0: Any
    p_array: np.ndarray
    rdot_array: np.ndarray
    pdot_array: np.ndarray
    t_interp_x: InterpolatedUnivariateSpline
    rx_interp_t: InterpolatedUnivariateSpline
    rz_interp_t: InterpolatedUnivariateSpline
    x_interp_t: InterpolatedUnivariateSpline
    rz_interp: InterpolatedUnivariateSpline
    p_interp: InterpolatedUnivariateSpline
    rdot_interp: InterpolatedUnivariateSpline
    rdotx_interp: InterpolatedUnivariateSpline
    rdotz_interp: InterpolatedUnivariateSpline
    pdot_interp: InterpolatedUnivariateSpline
    px_interp: InterpolatedUnivariateSpline
    pz_interp: InterpolatedUnivariateSpline
    rdotx_interp_t: InterpolatedUnivariateSpline
    rdotz_interp_t: InterpolatedUnivariateSpline
    rddotx_interp_t: InterpolatedUnivariateSpline
    rddotz_interp_t: InterpolatedUnivariateSpline
    tanbeta_array: np.ndarray
    tanalpha_array: np.ndarray
    beta_array: np.ndarray
    beta_p_interp: InterpolatedUnivariateSpline
    alpha_array: np.ndarray
    alpha_interp: InterpolatedUnivariateSpline
    cosbeta_array: np.ndarray
    sinbeta_array: np.ndarray
    u_array: np.ndarray
    xiv_p_array: np.ndarray
    xiv_v_array: np.ndarray
    uhorizontal_p_array: np.ndarray
    uhorizontal_v_array: np.ndarray
    u_interp: InterpolatedUnivariateSpline
    u_from_rdot_interp: InterpolatedUnivariateSpline
    xiv_p_interp: InterpolatedUnivariateSpline
    xiv_v_interp: InterpolatedUnivariateSpline
    uhorizontal_p_interp: InterpolatedUnivariateSpline
    uhorizontal_v_interp: InterpolatedUnivariateSpline

    def initial_conditions(
        self,
        t_lag: float = 0,
        xiv_0_: float = 0,
    ) -> Tuple[float, float, float, float]:
        """Initialize."""
        rz0_: float = 0
        rx0_: float = 0
        # HACK: poly_px_xiv0_eqn not actually defined...
        (px0_, pz0_) = pxpz0_from_xiv0(
            self.parameters, self.gmeq.pz_xiv_eqn, self.gmeq.poly_px_xiv0_eqn
        )
        print(type(pz0_))
        return (rz0_, rx0_, px0_, pz0_.subs(e2d(self.gmeq.xiv0_xih0_Ci_eqn)))

    def solve(self) -> None:
        """Solve Hamilton's equations for one ray."""
        # Record the ic as a list of one - to be used by solve_ODE_system
        self.ic_list: List[Tuple[float, float, float, float]] = [
            self.initial_conditions()
        ]
        self.model_dXdt_lambda = self.make_model()
        parameters_ = {Lc: self.parameters[Lc]}
        logging.debug("gme.ode.single_ray.solve: calling solver")
        (ivp_soln, rpt_arrays) = solve_Hamiltons_equations(
            model=self.model_dXdt_lambda,
            method=self.method,
            do_dense=self.do_dense,
            ic=self.ic_list[0],
            parameters=parameters_,
            t_array=self.ref_t_array.copy(),
            x_stop=self.x_stop,
            t_lag=0.0,
        )
        self.ivp_solns_list = [ivp_soln]
        self.save(rpt_arrays, 0)

    def postprocessing(self, spline_order=2, extrapolation_mode=0) -> None:
        """Process integration results into a ray trace."""
        # Basics
        [
            self.rx_array,
            self.rz_array,
            self.px_array,
            self.pz_array,
            self.t_array,
        ] = [self.rpt_arrays[rpt_][0] for rpt_ in rpt_tuple]
        self.pz0 = self.pz_array[0]
        self.p_array = np.sqrt(self.px_array ** 2 + self.pz_array ** 2)
        rpdot_array = np.array(
            [
                self.model_dXdt_lambda(0, rp_)
                for rp_ in zip(
                    self.rx_array, self.rz_array, self.px_array, self.pz_array
                )
            ]
        )
        [
            self.rdotx_array,
            self.rdotz_array,
            self.pdotx_array,
            self.pdotz_array,
        ] = [rpdot_array[:, idx] for idx in [0, 1, 2, 3]]
        self.rdot_array = np.sqrt(self.rdotx_array ** 2 + self.rdotz_array ** 2)
        self.pdot_array = np.sqrt(self.pdotx_array ** 2 + self.pdotz_array ** 2)

        # Interpolation functions to facilitate resampling and co-plotting
        self.t_interp_x = InterpolatedUnivariateSpline(
            self.rx_array, self.t_array, k=spline_order, ext=extrapolation_mode
        )
        self.rx_interp_t = InterpolatedUnivariateSpline(
            self.t_array, self.rx_array, k=spline_order, ext=extrapolation_mode
        )
        self.rz_interp_t = InterpolatedUnivariateSpline(
            self.t_array, self.rz_array, k=spline_order, ext=extrapolation_mode
        )

        self.x_interp_t = InterpolatedUnivariateSpline(
            self.t_array, self.rx_array, k=spline_order, ext=extrapolation_mode
        )
        self.rz_interp = InterpolatedUnivariateSpline(
            self.rx_array, self.rz_array, k=spline_order, ext=extrapolation_mode
        )
        self.p_interp = InterpolatedUnivariateSpline(
            self.rx_array, self.p_array, k=spline_order, ext=extrapolation_mode
        )
        self.rdot_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.rdot_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        self.rdotx_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.rdotx_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        self.rdotz_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.rdotz_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        self.pdot_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.pdot_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        self.px_interp = InterpolatedUnivariateSpline(
            self.rx_array, self.px_array, k=spline_order, ext=extrapolation_mode
        )
        self.pz_interp = InterpolatedUnivariateSpline(
            self.rx_array, self.pz_array, k=spline_order, ext=extrapolation_mode
        )
        # Ray acceleration
        self.rdotx_interp_t = InterpolatedUnivariateSpline(
            self.t_array,
            self.rdotx_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        self.rdotz_interp_t = InterpolatedUnivariateSpline(
            self.t_array,
            self.rdotz_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        self.rddotx_interp_t = self.rdotx_interp_t.derivative(n=1)
        self.rddotz_interp_t = self.rdotz_interp_t.derivative(n=1)

        # Angles
        self.tanbeta_array = -self.px_array / self.pz_array
        self.tanalpha_array = -self.rdotx_array / self.rdotz_array
        self.beta_array = np.arctan(self.tanbeta_array)
        self.beta_p_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.beta_array,
            k=spline_order,
            ext=extrapolation_mode,
        )

        self.alpha_array = np.mod(np.pi + np.arctan(self.tanalpha_array), np.pi)
        self.alpha_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.alpha_array,
            k=spline_order,
            ext=extrapolation_mode,
        )

        self.cosbeta_array = -self.pz_array / self.p_array
        self.sinbeta_array = self.px_array / self.p_array

        # Erosion rates
        self.u_array = 1 / self.p_array
        self.xiv_p_array = (
            self.rdot_array
            * (np.cos((self.alpha_array) - (self.beta_array)))
            / self.cosbeta_array
        )
        self.xiv_p_array[np.isnan(self.xiv_p_array)] = 0
        self.xiv_v_array = self.u_array / self.cosbeta_array
        self.uhorizontal_p_array = (
            self.rdot_array
            * (np.cos((self.alpha_array) - (self.beta_array)))
            / self.sinbeta_array
        )
        self.uhorizontal_p_array[np.isnan(self.uhorizontal_p_array)] = 0
        self.uhorizontal_v_array = self.u_array / np.sin(self.beta_array)
        self.u_interp = InterpolatedUnivariateSpline(
            self.rx_array, self.u_array, k=spline_order, ext=extrapolation_mode
        )
        self.u_from_rdot_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.rdot_array * (np.cos((self.alpha_array) - (self.beta_array))),
            k=spline_order,
            ext=extrapolation_mode,
        )
        # u^\downarrow = \dot{r}*\cos(\alpha-\beta) / \cos(\beta)
        self.xiv_p_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.xiv_p_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        # u^\downarrow = u^\perp / \cos(\beta)
        self.xiv_v_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.xiv_v_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        # u^\downarrow = \dot{r}*\cos(\alpha-\beta) / \sin(\beta)
        self.uhorizontal_p_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.uhorizontal_p_array,
            k=spline_order,
            ext=extrapolation_mode,
        )
        # u^\downarrow = u^\perp / \sin(\beta)
        self.uhorizontal_v_interp = InterpolatedUnivariateSpline(
            self.rx_array,
            self.uhorizontal_v_array,
            k=spline_order,
            ext=extrapolation_mode,
        )


#

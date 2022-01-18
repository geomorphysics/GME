"""
---------------------------------------------------------------------

Base module for performing ray tracing of Hamilton's equations.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`NumPy <numpy>`
  -  :mod:`SciPy <scipy>`
  -  :mod:`SymPy <sympy>`
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
# from functools import lru_cache
# from enum import Enum, auto
from typing import List, Dict, Any, Tuple, Callable, Optional

# Abstract classes & methods
from abc import ABC, abstractmethod

# NumPy
import numpy as np

# SciPy
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# SymPy
from sympy import Matrix, lambdify, simplify

# GME
from gme.core.symbols import px, pz, rx, rdotx, rdotz, Lc
from gme.core.equations import Equations

warnings.filterwarnings("ignore")

rp_tuple: Tuple[str, str, str, str] = ('rx', 'rz', 'px', 'pz')
rpt_tuple: Tuple[str, str, str, str, str] = rp_tuple+('t',)

__all__ = ['BaseSolution']


# class Choice(Enum):
#     HAMILTON = auto()
#     GEODESIC = auto()

# class SolveMethod(Enum):
#     DOP853 = auto()


class BaseSolution(ABC):
    """
    Base class for classes performing integration of
    Hamilton's equations (ODEs).
    """

    def __init__(
        self,
        gmeq: Equations,
        parameters: Dict,
        choice: str = 'Hamilton',
        method: str = 'Radau',
        do_dense: bool = True,
        x_stop: float = 0.999,
        t_end: float = 0.04,
        t_slip_end: float = 0.08,
        t_distribn: float = 2,
        n_rays: int = 20,
        n_t: int = 101,
        tp_xiv0_list: Optional[List[Tuple[float, float]]] = None,
        customize_t_fn: Optional[Callable] = None
    ) -> None:
        """
        Initialize class instance.

        Args:
            gmeq:
                GME model equations class instance defined in
                :mod:`gme.core.equations`
            parameters:
                dictionary of model parameter values to be used for
                equation substitutions
        """
        # Container for GME equations
        self.gmeq: Equations = gmeq

        # Model/equation parameters
        self.parameters: Dict = parameters

        # ODE solution method
        self.choice: str = choice
        self.method: str = method
        task = 'Solve Hamilton\'s ODEs' if self.choice == 'Hamilton' \
            else 'Solve geodesic ODEs'
        report = f'gme.ode.base.BaseSolution.init:\n   {task} '\
            + f'using {method} method of integration'
        logging.info(report)
        self.do_dense: bool = do_dense
        self.x_stop: float = x_stop

        # Rays
        self.tp_xiv0_list: Optional[List[Tuple[float, float]]] = tp_xiv0_list
        self.n_rays: int = n_rays
        self.t_end: float = t_end
        self.t_distribn: float = t_distribn
        self.t_slip_end: float = t_slip_end
        self.n_t: int = n_t
        # To record the longest ray time
        self.t_ensemble_max: float
        # Ray interpolation
        self.interp1d_kind: str = 'linear'

        # ODEs & related
        self.pz_velocity_boundary_eqn = gmeq.pz_xiv_eqn.subs(self.parameters)

        # Misc
        self.model_dXdt_lambda: \
            Callable[[float, Tuple[Any, Any, Any, Any]],
                     float] = lambda a, b: 0.0
        self.customize_t_fn: Optional[Callable] = customize_t_fn

        # Preliminary definitions and type annotations
        self.ic_list: List[Tuple[float, float, float, float]]
        self.ref_t_array \
            = np.linspace(0, 1, self.n_t)**self.t_distribn * self.t_end
        self.rpt_arrays: Dict[str, List[np.ndarray]] = {}
        for rp_ in rpt_tuple:
            self.rpt_arrays.update({
                rp_: [np.array([0])]*self.n_rays
            })
        # logging.info(type(self.rpt_arrays['rx']))
        self.ivp_solns_list: List[Any] = []
        self.rp_t_interp_fns: Dict[str, List[Optional[Callable]]] = {}
        self.t_isochrone_max: float = 0.0
        self.n_isochrones: int = 0
        self.x_subset: int = 0
        self.tolerance: float = 0.0
        self.rpt_isochrones: Dict[str, List] = {}
        self.trxz_cusps: List[Tuple[np.ndarray, Any, Any]] = []
        self.cusps: Dict[str, np.ndarray] = {}
        self.cx_pz_tanbeta_lambda: Optional[Callable[[float], float]] = None
        self.cx_pz_lambda: Optional[Callable[[float], float]] = None
        self.cx_v_lambda: Optional[Callable[[float], float]] = None
        self.vx_interp_fast: Optional[Callable[[float], float]] = None
        self.vx_interp_slow: Optional[Callable[[float], float]] = None

    @abstractmethod
    def initial_conditions(self) -> Tuple[float, float, float, float]:
        """
        Dummy method of generating initial conditions that must be defined by
        any subclass
        """

    @abstractmethod
    def solve(self) -> None:
        """
        Dummy method of solution that must be defined by any subclass
        """

    def make_model(self) \
            -> Callable[[float, Tuple[Any, Any, Any, Any]], np.ndarray]:
        """
        Generate a lambda for Hamilton's equations (or the geodesic equations)
           that returns a matrix of dr/dt and dp/dt (resp. dr/dt and dv/dt)
           for a given state [rx,px,pz] (resp. [rx,vx,vx])
        """
        # Requires self.parameters to have all the important Hamilton eqns'
        # constants set
        #   - generates a "matrix" of the 4 Hamilton equations with rx,rz,px,pz
        #     as variables and the rest as numbers
        log_string = 'gme.ode.base.BaseSolution.make_model:\n   '
        if self.choice == 'Hamilton':
            logging.info(
                f'{log_string}Constructing model Hamilton\'s equations')
            drpdt_eqn_matrix = simplify(
                self.gmeq.hamiltons_eqns.subs(self.parameters))
            # .subs({pz:-Abs(pz)})  # HACK - may need this
            drpdt_raw_lambda = lambdify([rx, px, pz], drpdt_eqn_matrix)
            return lambda t_, rp_: \
                np.ndarray.flatten(drpdt_raw_lambda(rp_[0], rp_[2], rp_[3]))

        logging.info(f'{log_string}Constructing model geodesic equations')
        drvdt_eqn_matrix = Matrix(
            ([(eq_.rhs) for eq_ in self.gmeq.geodesic_eqns]))
        drvdt_raw_lambda = lambdify([rx, rdotx, rdotz], drvdt_eqn_matrix)
        return lambda t_, rv_: \
            np.ndarray.flatten(drvdt_raw_lambda(rv_[0], rv_[2], rv_[3]))

    def postprocessing(
        self,
        spline_order: int = 2,
        extrapolation_mode: int = 0
    ) -> None:
        """
        Generate interpolating functions for (r,p)[t] using ray samples
        """
        # dummy, to avoid "overriding parameters" warnings in subclasses
        #  that override this method
        _ = (spline_order, extrapolation_mode)
        for rp_ in rp_tuple:
            self.rp_t_interp_fns.update({rp_: [None]*self.n_rays})
        fill_value_ = 'extrapolate'
        for (i_ray, t_array) in enumerate(self.rpt_arrays['t']):
            if t_array.size > 1:
                # Generate interpolation functions for each component
                #   rx[t], rz[t], px[t], pz[t]
                for rp_ in rp_tuple:
                    # logging.info((i_ray, rp_))
                    self.rp_t_interp_fns[rp_][i_ray] \
                        = interp1d(t_array,
                                   self.rpt_arrays[rp_][i_ray],
                                   kind=self.interp1d_kind,
                                   fill_value=fill_value_,
                                   assume_sorted=True)

    def resolve_isochrones(
        self,
        x_subset: int = 1,
        t_isochrone_max: float = 0.04,
        n_isochrones: int = 25,
        n_resample_pts: int = 301,
        tolerance: float = 1e-3,
        do_eliminate_caustics: bool = True,
        dont_crop_cusps: bool = False
    ) -> None:
        r"""
        Resample the ensemble of rays at selected time slices to generate
        synchronous values of :math:`(\mathbf{r},\mathbf{\widetilde{p}})`
        aka isochrones.

        Each ray trajectory is numerically integrated independently, and
        so the integration points along one ray are
        not synchronous with those of any other ray. If we want to compare
        the :math:`(\mathbf{r},\mathbf{\widetilde{p}})`
        "positions" of the ray ensemble
        at a chosen time, we need to resample along all the rays
        at mutually consistent time slices.
        This is achieved by first interpolating along each
        :math:`\{r^x[t], r^z[t], p_x[t], p_z[t]\}` sequence, and then
        resampling along a reference time sequence.

        Two additional actions are taken:
           (1) termination of each resampled ray at the domain boundary
               at :math:`r^x=L_c` (or a bit beyond, for better viz quality);
           (2) termination of any resampled ray that is overtaken by another
               ray at a cusp.
        """
        # def coarsen_isochrone(rpt_isochrone) -> None:
        #     """
        #     TBD
        #     """
        #     # Reduce the time resolution of the isochrone points
        #       to make plotting less cluttered
        #     self.rpt_isochrones_lowres: Dict[str,List] = {}
        #     for rp_ in rp_tuple:
        #         self.rpt_isochrones_lowres[rp_][i_isochrone] \
        #             = [ [element_ for idx,element_ in enumerate(array_)
        #                  if (idx//x_subset-idx/x_subset)==0]
        #                 for array_ in [rpt_isochrone[rp_]] ][0]
        #     self.rpt_isochrones_lowres['t'][i_isochrone] = rpt_isochrone['t']
        def prepare_isochrones() -> None:
            """
            TBD
            """
            # Record important parameters
            self.t_isochrone_max = t_isochrone_max
            self.n_isochrones = n_isochrones
            self.x_subset = x_subset
            self.tolerance = tolerance if tolerance is not None else 1e-3

            # Prepare array dictionaries etc
            # self.rpt_isochrones: Dict[str,List] = {}
            for rpt_ in rpt_tuple:
                self.rpt_isochrones.update({rpt_: [None]*n_isochrones})
            # self.rpt_isochrones_lowres = self.rpt_isochrones.copy()
            # self.trxz_cusps: List[Tuple[np.ndarray,Any,Any]] = []

        def truncate_isochrone(
            rpt_isochrone: Dict[str, np.ndarray],
            i_from: Optional[int] = None,
            i_to: Optional[int] = None
        ) -> Dict[str, np.ndarray]:
            """
            TBD
            """
            for rp_ in rp_tuple:
                rpt_isochrone[rp_] = rpt_isochrone[rp_][i_from:i_to]
            return rpt_isochrone

        def find_intercept(
            rpt_isochrone,
            slice1,
            slice2
        ) -> Tuple[Any, Any, Any, Any]:
            """
            TBD
            """
            x1_array = rpt_isochrone['rx'][slice1]
            x2_array = rpt_isochrone['rx'][slice2]
            z1_array = rpt_isochrone['rz'][slice1]
            z2_array = rpt_isochrone['rz'][slice2]
            px1_array = rpt_isochrone['px'][slice1]
            px2_array = rpt_isochrone['px'][slice2]
            pz1_array = rpt_isochrone['pz'][slice1]
            pz2_array = rpt_isochrone['pz'][slice2]
            s1_array = np.linspace(0, 1, num=len(x1_array))
            s2_array = np.linspace(0, 1, num=len(x2_array))

            # Arguments given to interp1d:
            #  - extrapolate: to make sure we don't get a fatal value error
            # when fsolve searches beyond the bounds of [0,1]
            #  - copy: use refs to the arrays
            #  - assume_sorted: because s_array ('x') increases monotonically
            #    across [0,1]

            kwargs_ = dict(fill_value='extrapolate',
                           copy=False, assume_sorted=True)
            slice_ = np.s_[::1]
            x1_interp = interp1d(s1_array[slice_], x1_array[slice_], **kwargs_)
            x2_interp = interp1d(s2_array[slice_], x2_array[slice_], **kwargs_)
            z1_interp = interp1d(s1_array[slice_], z1_array[slice_], **kwargs_)
            z2_interp = interp1d(s2_array[slice_], z2_array[slice_], **kwargs_)
            px1_interp = interp1d(
                s1_array[slice_], px1_array[slice_], **kwargs_)
            px2_interp = interp1d(
                s2_array[slice_], px2_array[slice_], **kwargs_)
            pz1_interp = interp1d(
                s1_array[slice_], pz1_array[slice_], **kwargs_)
            pz2_interp = interp1d(
                s2_array[slice_], pz2_array[slice_], **kwargs_)

            def xydiff_lambda(s12): \
                return ((np.abs(x1_interp(s12[0])-x2_interp(s12[1]))),
                        (np.abs(z1_interp(s12[0])-z2_interp(s12[1]))))

            s12_intercept, _, _, _ \
                = fsolve(xydiff_lambda,
                         [0.99, 0.01],
                         factor=0.1,
                         full_output=True)  # , factor=0.1,
            xz1_intercept = x1_interp(
                s12_intercept[0]), z1_interp(s12_intercept[0])
            xz2_intercept = x2_interp(
                s12_intercept[1]), z2_interp(s12_intercept[1])
            pxz1_intercept = px1_interp(
                s12_intercept[0]), pz1_interp(s12_intercept[0])
            pxz2_intercept = px2_interp(
                s12_intercept[1]), pz2_interp(s12_intercept[1])
            # print(ier, mesg, s12_intercept, s12_intercept)
            # self.x1_interp = x1_interp
            # self.x2_interp = x2_interp
            # self.z1_interp = z1_interp
            # self.z2_interp = z2_interp
            # self.xydiff_lambda = xydiff_lambda

            return \
                (xz1_intercept, xz2_intercept, pxz1_intercept, pxz2_intercept)

        def eliminate_caustic(
            is_good_pt_array: np.ndarray,
            rpt_isochrone
        ) -> Tuple[
                Dict[str, np.ndarray],
                Tuple[Optional[np.ndarray],
                      Optional[np.ndarray],
                      Optional[np.ndarray]]
                ]:
            """
            TBD
            """
            # Locate and remove caustic and render into a cusp
            #    - if there is one

            # Check if there are any false points
            #    - if not, just return the whole curve
            if len(rpt_isochrone['rx'][is_good_pt_array]) \
                    == len(rpt_isochrone['rx']):
                # print('whole curve')
                return rpt_isochrone, (None, None, None)

            # Check if there are ONLY false points, in which case return an
            #    empty curve
            if len(rpt_isochrone['rx'][is_good_pt_array]) == 0:
                # print('empty curve')
                return (
                    truncate_isochrone(rpt_isochrone, -1, -1),
                    (None, None, None)
                )

            # Find false indexes
            # HACK ? Comparison 'is_good_pt_array == False'
            #          should be 'is_good_pt_array is False'
            #       if checking for the singleton value False,
            #            or 'not is_good_pt_array'
            #       if testing for falsiness (singleton-comparison)
            false_indexes = np.where(not is_good_pt_array)[0]
            # If false_indexes[0]==0, there's no left curve,
            #    so use only the right half
            if false_indexes[0] == 0:
                # print('right half')
                return (
                    truncate_isochrone(rpt_isochrone, false_indexes[-1], None),
                    (None, None, None)
                )

            if len(false_indexes) > 1:
                last_false_index \
                    = [idx for idx, fi
                       in enumerate(zip(false_indexes[:-1], false_indexes[1:]))
                       if fi[1]-fi[0] > 1]
                # print('last:', last_false_index)
                if len(last_false_index) > 0:
                    false_indexes = false_indexes[0:(last_false_index[0]+1)]

            # Otherwise, generate interpolations of x and y points
            #    for both left and right curves
            slice1 = np.s_[:(false_indexes[0]+1)]
            slice2 = np.s_[(false_indexes[-1]+1):]

            # If false_indexes[-1]==len(false_indexes)-1,
            #        use only left half (won't happen?)
            #   (assumes pattern is TTFFFFTT or TTTTFFFF etc)
            if false_indexes[-1] == len(is_good_pt_array)-1 \
                    or len(is_good_pt_array[slice2]) == 1:
                # print('left half')
                return (truncate_isochrone(rpt_isochrone, 0, false_indexes[0]),
                        (None, None, None))

            # At this point, we presumably have a caustic,
            #    so find the cusp intercept
            rxz1_intercept, _, pxz1_intercept, pxz2_intercept \
                = find_intercept(rpt_isochrone, slice1, slice2)
            # print('Intercept @ ', rxz1_intercept,rxz2_intercept)

            # Delimit the portions to the left and to the right of the cusp
            i1_array = np.where(
                rpt_isochrone['rx'][slice1] <= rxz1_intercept[0])[0]
            i2_array = np.where(
                rpt_isochrone['rx'][slice2] >= rxz1_intercept[0])[0]
            # print('i1,i2 arrays', i1_array,i2_array)

            # Rebuild the isochrone dictionary
            rpt_isochrone_rtn = {}
            for rp_ in rp_tuple:
                isochrone_left = rpt_isochrone[rp_][slice1]
                isochrone_right = rpt_isochrone[rp_][slice2]
                isochrone_ = np.concatenate([isochrone_left[i1_array],
                                             isochrone_right[i2_array]])
                rpt_isochrone_rtn.update({rp_: isochrone_})
            rpt_isochrone_rtn.update({'t': rpt_isochrone['t']})

            intercept: Tuple[Optional[np.ndarray],
                             Optional[np.ndarray],
                             Optional[np.ndarray]]
            if rxz1_intercept[0] > 0 \
                and (dont_crop_cusps
                     or rxz1_intercept[0] <= self.parameters[Lc]):
                intercept = (
                    # (rpt_isochrone['t'],
                    np.array(rxz1_intercept),
                    np.array(pxz1_intercept),
                    np.array(pxz2_intercept)
                )
            else:
                intercept = (None, None, None)
            return (rpt_isochrone_rtn, intercept)

        def compose_isochrone(t_: float) -> Dict[str, np.ndarray]:
            """
            TBD
            """
            # Prepare a tmp dictionary for the rx,rz,px,pz component arrays
            #   delineating this isochrone
            rpt_isochrone: Dict[str, Any] = {}
            for rp_ in rp_tuple:
                # Sample each ray at time t_ from each of the components
                #    of r,p vectors using their interpolating fns
                rp_interpolated_isochrone \
                    = [float(interp_fn(t_))
                        for interp_fn in self.rp_t_interp_fns[rp_]
                       if interp_fn is not None]
                rpt_isochrone.update(
                    {rp_: np.array(rp_interpolated_isochrone)})
            rpt_isochrone.update({'t': t_})
            return rpt_isochrone

        def resample_isochrone(
            rpt_isochrone_in: Dict,
            n_resample_pts: int
        ) -> Dict[str, np.ndarray]:
            """
            TBD
            """
            n_s_pts = n_resample_pts
            rpt_isochrone_out = rpt_isochrone_in.copy()
            s_array \
                = np.cumsum(np.concatenate([
                    np.array([0]),
                    np.sqrt(
                        (rpt_isochrone_in['rx'][1:]
                            - rpt_isochrone_in['rx'][:-1])**2
                        + (rpt_isochrone_in['rz'][1:]
                           - rpt_isochrone_in['rz'][:-1])**2
                        )
                ]))
            rs_array = np.linspace(s_array[0], s_array[-1], num=n_s_pts)
            # HACK to avoid interpolating a crap isochrone sampled
            #   only at one point
            if s_array[-1] < 1e-10:
                return rpt_isochrone_out
            for rp_ in rp_tuple:
                rpt_isochrone_interp_fn \
                    = interp1d(s_array, rpt_isochrone_out[rp_], kind='linear',
                               fill_value='extrapolate', assume_sorted=True)
                rpt_isochrone_out[rp_] = rpt_isochrone_interp_fn(rs_array)
            return rpt_isochrone_out

        # Crop out-of-bounds points and points on caustics
        #   for the current isochrone

        def clean_isochrone(
            rpt_isochrone: Dict
        ) -> Tuple[Optional[Dict[str, np.ndarray]],
                   Tuple[Optional[np.ndarray],
                         Optional[np.ndarray],
                         Optional[np.ndarray]]]:
            """
            TBD
            """
            # Cusps: eliminate all points beyond the first
            #      whose rx sequence is negative (left-ward)
            #  - do this by creating a flag array whose elements are True
            #       if rx[i+1]>rx[i]
            is_good_pt_array \
                = np.concatenate((
                    (rpt_isochrone['rx'][1:]-rpt_isochrone['rx'][:-1]
                     >= -self.tolerance),
                    [True]
                ))

            # Replace this isochrone with one whose caustic
            #    has been removed - if it has one
            # rpt_isochrone_try,
            #   (t_rxz_intercept, pxz1_intercept, pxz2_intercept) \
            #         = eliminate_caustic(is_good_pt_array, rpt_isochrone)
            #            if do_eliminate_caustics else None, (None,None,None)
            rtn: Tuple[Optional[Dict[str, np.ndarray]],
                       Tuple[Optional[np.ndarray],
                             Optional[np.ndarray],
                             Optional[np.ndarray]]]
            # (rpt_isochrone,
            #  (t_rxz_intercept, pxz1_intercept, pxz2_intercept)) \
            if do_eliminate_caustics:
                rtn = eliminate_caustic(is_good_pt_array, rpt_isochrone)
            else:
                rtn = (None, (None, None, None))
            # print(rpt_isochrone_try)
            # rpt_isochrone = rpt_isochrone_try if rpt_isochrone_try
            #                          is not None else rpt_isochrone
            return rtn

        def prune_isochrone(
            rpt_isochrone_in: Dict
        ) -> Dict[str, np.ndarray]:
            """
            TBD
            """
            rpt_isochrone_out = rpt_isochrone_in.copy()
            rx_array = rpt_isochrone_out['rx']
            i_bounded_array = (rx_array >= 0) & (rx_array <= 1.0)
            rpt_isochrone_out['rx'] = rx_array[i_bounded_array]
            did_clip_at_x1 = len(rx_array) > len(rpt_isochrone_out['rx'])
            for rp_ in rp_tuple:
                rpt_isochrone_out[rp_] = rpt_isochrone_in[rp_][i_bounded_array]
                # Wildly inefficient interpolation to do extrapolation here
                if did_clip_at_x1:
                    rpt_isochrone_interp_fn \
                        = interp1d(rx_array[rx_array >= 0.95],
                                   rpt_isochrone_in[rp_][rx_array >= 0.95],
                                   kind='linear', fill_value='extrapolate',
                                   assume_sorted=True)
                    rpt_at_x1 = rpt_isochrone_interp_fn(1.0)
                    rpt_isochrone_out[rp_] \
                        = np.concatenate([rpt_isochrone_out[rp_],
                                          np.array([rpt_at_x1])])
            return rpt_isochrone_out

        def record_isochrone(
            rpt_isochrone: Dict,
            i_isochrone: int
        ) -> None:
            """
            TBD
            """
            # Record this isochrone
            for rpt_ in rpt_tuple:
                self.rpt_isochrones[rpt_][i_isochrone] = rpt_isochrone[rpt_]

        def record_cusp(
            trxz_cusp: Optional[np.ndarray],
            pxz1_intercept,
            pxz2_intercept
        ) -> None:
            """
            TBD
            """
            if trxz_cusp is not None:
                # Record this intercept
                self.trxz_cusps.append(
                    (trxz_cusp, pxz1_intercept, pxz2_intercept))

        def organize_cusps() -> None:
            """
            TBD
            """
            # self.cusps: Dict[str,np.ndarray] = {}
            self.cusps['t'] = np.array([t_ for (t_, rxz_), pxz1_, pxz2_
                                        in self.trxz_cusps
                                        if rxz_[0] >= 0 and rxz_[0] <= 1])
            self.cusps['rxz'] = np.array([rxz_ for (t_, rxz_), pxz1_, pxz2_
                                          in self.trxz_cusps
                                          if rxz_[0] >= 0 and rxz_[0] <= 1])
            self.cusps['pxz1'] = np.array([pxz1_ for (t_, rxz_), pxz1_, pxz2_
                                           in self.trxz_cusps
                                           if rxz_[0] >= 0 and rxz_[0] <= 1])
            self.cusps['pxz2'] = np.array([pxz2_ for (t_, rxz_), pxz1_, pxz2_
                                           in self.trxz_cusps
                                           if rxz_[0] >= 0 and rxz_[0] <= 1])

        # Create isochrones of the evolving surface:
        #   (1) step through each time t_i in the global time sequence
        #   (2) for each ray, generate an (rx,rz,px,pz)[t_i] vector through
        #         interpolation of its numerically integrated sequence
        #   (3) combine these vectors into arrays
        #       for the {rx[t_i]}, {rz[t_i]}, {px[t_i]}
        #       and {pz[t_i]}
        #   (4) truncate a ray from the point where (if at all)
        #         its rx sequence reverses
        #         and goes left not right
        #   (5) also truncate if a ray leaves the domain, i.e., rx>Lc

        # Time sequence - with resolution n_isochrones and
        #    limit t_isochrone_max
        prepare_isochrones()
        t_array = np.linspace(0, t_isochrone_max, n_isochrones)
        for i_isochrone, t_ in enumerate(t_array):
            rpt_isochrone = compose_isochrone(t_)  # i_isochrone,
            rpt_isochrone_resampled \
                = resample_isochrone(rpt_isochrone, n_resample_pts)
            (rpt_isochrone_clean,
             (trxz_cusp, pxz1_intercept, pxz2_intercept)) \
                = clean_isochrone(rpt_isochrone_resampled)
            if rpt_isochrone_clean is not None:
                rpt_isochrone_pruned = prune_isochrone(rpt_isochrone_clean)
                record_isochrone(rpt_isochrone_pruned, i_isochrone)
                record_cusp(trxz_cusp, pxz1_intercept, pxz2_intercept)
        organize_cusps()

    def measure_cusp_propagation(self) -> None:
        """
        TBD
        """
        kwargs_ = dict(kind='linear',
                       fill_value='extrapolate',
                       assume_sorted=True)

        if self.cusps['t'].shape[0] == 0:
            self.cx_pz_tanbeta_lambda = None
            self.cx_pz_lambda = None
            self.cx_v_lambda = None
            self.vx_interp_fast = None
            self.vx_interp_slow = None
            return
        (_, rxz_array, pxz_fast_array, pxz_slow_array) \
            = [self.cusps[key_] for key_ in ['t', 'rxz', 'pxz1', 'pxz2']]
        (px_fast_array, pz_fast_array) \
            = (pxz_fast_array[:, 0], pxz_fast_array[:, 1])
        (px_slow_array, pz_slow_array) \
            = (pxz_slow_array[:, 0], pxz_slow_array[:, 1])
        (rx_array, rz_array) \
            = (rxz_array[:, 0], rxz_array[:, 1])

        # p_fast_array = np.sqrt(px_fast_array**2+pz_fast_array**2)
        tanbeta_fast_array = -px_fast_array/pz_fast_array
        # sinbeta_fast_array =  px_fast_array/p_fast_array
        # cosbeta_fast_array = -pz_fast_array/p_fast_array
        # p_fast_interp       = interp1d( rxz_array[:,0], p_fast_array,
        #                       **kwargs_ )
        px_fast_interp = interp1d(
            rxz_array[:, 0], px_fast_array,      **kwargs_)
        pz_fast_interp = interp1d(
            rxz_array[:, 0], pz_fast_array,      **kwargs_)
        tanbeta_fast_interp = interp1d(
            rxz_array[:, 0], tanbeta_fast_array, **kwargs_)
        # sinbeta_fast_interp
        #     = interp1d( rxz_array[:,0], sinbeta_fast_array, **kwargs_ )
        # cosbeta_fast_interp
        #     = interp1d( rxz_array[:,0], cosbeta_fast_array, **kwargs_ )

        # p_slow_array = np.sqrt(px_slow_array**2+pz_slow_array**2)
        tanbeta_slow_array = -px_slow_array/pz_slow_array
        # sinbeta_slow_array =  px_slow_array/p_slow_array
        # cosbeta_slow_array = -pz_slow_array/p_slow_array
        # p_slow_interp
        #     = interp1d( rxz_array[:,0], p_slow_array, **kwargs_ )
        px_slow_interp = interp1d(
            rxz_array[:, 0], px_slow_array, **kwargs_)
        pz_slow_interp = interp1d(
            rxz_array[:, 0], pz_slow_array, **kwargs_)
        tanbeta_slow_interp = interp1d(
            rxz_array[:, 0], tanbeta_slow_array, **kwargs_)
        # sinbeta_slow_interp \
        #    = interp1d( rxz_array[:,0], sinbeta_slow_array, **kwargs_ )
        # cosbeta_slow_interp \
        #    = interp1d( rxz_array[:,0], cosbeta_slow_array, **kwargs_ )

        # Provide a lambda fn that returns the horizontal cusp velocity
        #   component cx
        #   using interpolation functions for all the variables
        #   for which we have sampled solutions as arrays
        # sinbeta_diff_lambda \
        #   = lambda x_: sinbeta_fast_interp(x_)*cosbeta_slow_interp(x_)\
        #                - cosbeta_fast_interp(x_)*sinbeta_slow_interp(x_)
        # tanbeta_diff_lambda \
        #   = lambda x_: tanbeta_fast_interp(x_)-tanbeta_slow_interp(x_)
        self.cx_pz_tanbeta_lambda \
            = lambda x_: (
                - (1/pz_fast_interp(x_) - 1/pz_slow_interp(x_))
                / (tanbeta_fast_interp(x_) - tanbeta_slow_interp(x_))
                )
        self.cx_pz_lambda \
            = lambda x_: (
                (pz_slow_interp(x_) - pz_fast_interp(x_))
                / (px_fast_interp(x_)*pz_slow_interp(x_)
                    - pz_fast_interp(x_)*px_slow_interp(x_))
                )
        rpdot_fast_array \
            = np.array([self.model_dXdt_lambda(0, rp_)
                        for rp_ in zip(rx_array,
                                       rz_array,
                                       px_fast_array,
                                       pz_fast_array)])
        rpdot_slow_array \
            = np.array([self.model_dXdt_lambda(0, rp_)
                        for rp_ in zip(rx_array,
                                       rz_array,
                                       px_slow_array,
                                       pz_slow_array)])
        vx_interp_fast = interp1d(rx_array, rpdot_fast_array[:, 0], **kwargs_)
        vx_interp_slow = interp1d(rx_array, rpdot_slow_array[:, 0], **kwargs_)
        vz_interp_fast = interp1d(rx_array, rpdot_fast_array[:, 1], **kwargs_)
        vz_interp_slow = interp1d(rx_array, rpdot_slow_array[:, 1], **kwargs_)
        self.cx_v_lambda \
            = lambda x_: (
                ((tanbeta_fast_interp(x_)*vx_interp_fast(x_)
                    - vz_interp_fast(x_))
                 - (tanbeta_slow_interp(x_)*vx_interp_slow(x_)
                    - vz_interp_slow(x_)))
                / (tanbeta_fast_interp(x_) - tanbeta_slow_interp(x_))
                )
        self.vx_interp_fast = vx_interp_fast
        self.vx_interp_slow = vx_interp_slow

    def save(
        self,
        rpt_arrays: Dict,
        idx: int
    ) -> None:
        """
        TBD
        """
        logging.debug('ode.base.BaseSolution.save')
        for rpt_ in rpt_tuple:
            self.rpt_arrays[rpt_][idx] = rpt_arrays[rpt_]
        rx_length = len(self.rpt_arrays['rx'][idx])
        self.rpt_arrays['t'][idx] = self.rpt_arrays['t'][idx][:rx_length]


#

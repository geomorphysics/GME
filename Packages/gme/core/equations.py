"""
Derive an ensemble of GME equations using :mod:`SymPy <sympy>`.

This module provides a derivation of GME theory using :mod:`SymPy <sympy>`
for a 2D slice of a 3D landscape along a channel transect.

Starting from a model equation for the surface-normal erosion rate in terms of
the tilt angle of the topographic surface and the distance downstream
(used to infer the flow component of the model erosion process), we derive
the fundamental function (of a co-Finsler metric space), the corresponding
Hamiltonian,
and the equivalent metric tensor.
The rest of the equations are derived from these core results.

---------------------------------------------------------------------

Requires Python packages/modules:
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
from typing import Dict, Optional

# SymPy
from sympy import Eq, Rational

# GME
from gme.core.rp import RpMixin
from gme.core.xi import XiMixin
from gme.core.varphi import VarphiMixin
from gme.core.fundamental import FundamentalMixin
from gme.core.hamiltons import HamiltonsMixin
from gme.core.ndim import NdimMixin
from gme.core.profile import ProfileMixin
from gme.core.angles import AnglesMixin
from gme.core.metrictensor import MetricTensorMixin
from gme.core.pxpoly import PxpolyMixin

# from gme.core.idtx import IdtxMixin
# from gme.core.geodesic import GeodesicMixin
# from gme.core.ibc import IbcMixin

warnings.filterwarnings("ignore")

__all__ = ["EquationsBase", "EquationsMixedIn", "Equations"]


class EquationsBase:
    """Bare-bones base class for equation definitions."""

    # Definitions
    eta_: Eq
    mu_: Eq
    beta_type: Eq
    varphi_type: Eq
    do_raw: Eq

    def __init__(
        self,
        parameters: Optional[Dict] = None,
        eta_: Rational = Rational(3, 2),
        mu_: Rational = Rational(3, 4),
        beta_type: str = "sin",
        varphi_type: str = "ramp",
        # ibc_type: str = 'convex-up',
        do_raw: bool = True,
    ):
        """Initialize: constructor method."""
        logging.info("gme.core.equations.EquationsBase")

        self.eta_ = eta_
        self.mu_ = mu_
        # self.ibc_type = ibc_type
        self.beta_type = beta_type
        self.varphi_type = varphi_type
        self.do_raw = do_raw


class EquationsMixedIn(
    EquationsBase,
    RpMixin,
    XiMixin,
    VarphiMixin,
    FundamentalMixin,
    HamiltonsMixin,
    NdimMixin,
    ProfileMixin,
    AnglesMixin,
    MetricTensorMixin,
    PxpolyMixin,
):
    """
    Equation definitions mixin.

    Extended base class that's furnished with
    mixins providing all the basic equation definitions, but none of
    which are automatically acted upon.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize: constructor method."""
        logging.info("gme.core.equations.EquationsMixedIn")

        super().__init__(**kwargs)


class Equations(EquationsMixedIn):
    r"""
    Equation definitions.

    Class to solve the set of GME equations(using: mod: `SymPy < sympy >`)
    and to provide them in a form(sometimes lambdified) that can be used for
    numerical evaluation.

    unspecified, up until the Hamiltonian is defined, but eventually values
    for these parameters need to be substituted in order to make further
    progress. In this documentation, we set :math:`\eta = 3/2`, for now.

    TODO: provide solutions for both :math:`\eta = 1/2` and
           :math:`\eta = 3/2` where appropriate.

    Args:
        parameters: dictionary of model parameter values to be
                           used for equation substitutions
                           (used when defining geodesic equations)
        eta\_:
            exponent :math:`\eta ` in slope component of erosion
            model (equivalent of
            gradient exponent :math:`n` in SPIM)
        mu\_:
            exponent :math:`\mu` in flow component of erosion model
            (equivalent of area exponent :math:`m` in SPIM)
        beta_type: choice of slope component of erosion model
                         (`'sin'` or `'tan'`)
        varphi_type: choice of flow component of erosion model
                           (`'ramp'` or `'ramp-flat'`)
        ibc_type: choice of initial boundary shape
                        (`'convex-up'` or `'concave-up'`,
                        i.e., concave vs convex in mathematical parlance)
        do_raw: suppress substitution of :math:`eta` value
                        when defining `xi_varphi_beta_eqn`?
        do_idtx: generate indicatrix and figuratrix equations?
        do_geodesic: generate geodesic equations?
        do_nothing:
            just create the class instance and set its data,
             but don't run Eq of the equation definition methods
        do_new_varphi_model(bool): use new form of varphi model?

    Attributes:
        GME equations(: class: `~sympy.core.relational.Equality` etc):
            See below
    """

    def __init__(self, parameters: Optional[Dict] = None, **kwargs) -> None:
        """Initialize: constructor method."""
        logging.info("gme.core.equations.Equations")

        super().__init__(parameters=parameters, **kwargs)

        self.define_p_eqns()
        self.define_r_eqns()
        self.define_xi_eqns()
        self.define_xi_model_eqn()
        self.define_xi_related_eqns()
        self.define_varphi_model_eqns()
        self.define_varphi_related_eqns()
        self.define_Fstar_eqns()
        self.define_H_eqns()
        self.define_rdot_eqns()
        self.define_pdot_eqns()
        self.define_Hamiltons_eqns()
        self.nondimensionalize()
        self.define_nodimensionalized_Hamiltons_eqns()
        self.define_z_eqns()
        self.define_tanalpha_eqns()
        self.define_tanbeta_eqns()
        self.define_psi_eqns()
        self.define_g_eqns()
        self.define_px_poly_eqn(eta_choice=self.eta_, do_ndim=False)
        self.define_px_poly_eqn(eta_choice=self.eta_, do_ndim=True)


#

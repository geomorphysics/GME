"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

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
  -  :mod:`sympy`
  -  :mod:`gme`

---------------------------------------------------------------------

"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable
import warnings

# Typing
from typing import Dict, Optional

# SymPy
from sympy import Rational

# GME
from gme.core.equations_rp import EquationsRpMixin
from gme.core.equations_xi import EquationsXiMixin
from gme.core.equations_varphi import EquationsVarphiMixin
from gme.core.equations_FH import EquationsFHMixin
from gme.core.equations_hamiltons import EquationsHamiltonsMixin
from gme.core.equations_ndim import EquationsNdimMixin
from gme.core.equations_angles import EquationsAnglesMixin
from gme.core.equations_g import EquationsMetricTensorMixin
from gme.core.equations_idtx import EquationsIdtxMixin
from gme.core.equations_geodesic import EquationsGeodesicMixin
from gme.core.equations_pxpoly import EquationsPxpolyMixin
from gme.core.equations_ibc import EquationsIbcMixin

warnings.filterwarnings("ignore")

__all__ = ['Equations']


class Equations(EquationsRpMixin,
                EquationsXiMixin,
                EquationsVarphiMixin,
                EquationsFHMixin,
                EquationsHamiltonsMixin,
                EquationsNdimMixin,
                EquationsAnglesMixin,
                EquationsMetricTensorMixin,
                EquationsIdtxMixin,
                EquationsGeodesicMixin,
                EquationsPxpolyMixin,
                EquationsIbcMixin):
    r"""
    Class to solve the set of GME equations (using :mod:`SymPy <sympy>`)
    and to provide them in a form (sometimes lambdified) that can be used for
    numerical evaluation.

    Much of the derivation sequence here keeps :math:`\eta` and :math:`\mu`
    unspecified, up until the Hamiltonian is defined, but eventually values
    for these parameters need to be substituted in order to make further
    progress. In this documentation, we set :math:`\eta=3/2`, for now.

    TODO: provide solutions for both :math:`\eta=1/2` and
          :math:`\eta=3/2` where appropriate.

    Args:
        parameters (dict): dictionary of model parameter values to be
                           used for equation substitutions
                           (used when defining geodesic equations)
        eta\_ (:class:`~sympy.core.numbers.Rational`):
            exponent in slope component of erosion model (equivalent of
            gradient exponent :math:`n` in SPIM)
        mu\_ (:class:`~sympy.core.numbers.Rational`):
            exponent in flow component of erosion model
            (equivalent of area exponent :math:`m` in SPIM)
        beta_type (str): choice of slope component of erosion model
                         (`'sin'` or `'tan'`)
        varphi_type (str): choice of flow component of erosion model
                           (`'ramp'` or `'ramp-flat'`)
        ibc_type (str): choice of initial boundary shape
                        (`'convex-up'` or `'concave-up'`,
                        i.e., concave vs convex in mathematical parlance)
        do_raw (bool): suppress substitution of :math:`eta` value
                        when defining `xi_varphi_beta_eqn`?
        do_idtx (bool): generate indicatrix and figuratrix equations?
        do_geodesic (bool): generate geodesic equations?
        do_nothing (bool):
            just create the class instance and set its data,
             but don't run any of the equation definition methods
        do_new_varphi_model (bool): use new form of varphi model?

    Attributes:
        GME equations (:class:`~sympy.core.relational.Equality` etc):
            See below
    """

    def __init__(
        self,
        parameters: Optional[Dict] = None,
        eta_: Rational = Rational(3, 2),
        mu_: Rational = Rational(3, 4),
        beta_type: str = 'sin',
        varphi_type: str = 'ramp',
        ibc_type: str = 'convex-up',
        do_raw: bool = True,
        do_idtx: bool = False,
        do_geodesic: bool = False,
        do_nothing: bool = False,
        do_new_varphi_model: bool = True
    ) -> None:
        r"""
        Constructor method.

        Define/derive all the GME equations (unless `'do_nothing'` is true)
        using :mod:`SymPy <sympy>`.
        """

        self.eta_ = eta_
        self.mu_ = mu_
        self.do_raw = do_raw
        self.ibc_type = ibc_type
        self.beta_type = beta_type
        self.varphi_type = varphi_type
        if do_nothing:
            return

        self.define_p_eqns()
        self.define_r_eqns()
        self.define_xi_eqns()
        self.define_xi_model_eqn()
        self.define_xi_related_eqns()
        self.define_varphi_model_eqn(do_new=do_new_varphi_model)
        self.define_varphi_related_eqns()
        self.define_Fstar_eqns()
        self.define_H_eqns()
        self.define_rdot_eqns()
        self.define_pdot_eqns()
        self.define_Hamiltons_eqns()
        self.nondimensionalize()
        self.define_nodimensionalized_Hamiltons_eqns()
        self.define_tanalpha_eqns()
        self.define_tanbeta_eqns()
        self.define_psi_eqns()
        self.define_g_eqns()
        if do_idtx:
            self.define_idtx_fgtx_eqns()
        if do_geodesic:
            self.prep_geodesic_eqns(parameters if not do_raw else None)
            self.define_geodesic_eqns()  # parameters if not do_raw else None)
        self.define_px_poly_eqn(eta_choice=self.eta_, do_ndim=False)
        self.define_px_poly_eqn(eta_choice=self.eta_, do_ndim=True)
        self.prep_ibc_eqns()
        self.define_ibc_eqns()
        self.set_ibc_eqns()

#

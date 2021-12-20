"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`SymPy <sympy>`
  -  `GMPLib`_
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html

---------------------------------------------------------------------

"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable
import warnings
import logging

# Typing
# from typing import Dict, Type, Optional  # , Tuple, Any, List

# SymPy
from sympy import Eq, simplify, factor, solve, denom, \
                  sin, cos, tan, Abs

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import \
    p, px, pz, beta, xi, xiv, rvec, varphi_r, eta

warnings.filterwarnings("ignore")

__all__ = ['XiMixin']


class XiMixin:
    r"""
    """

    def define_xi_eqns(self) -> None:
        r"""
        Define equations for surface erosion speed :math:`\xi`
        and its vertical behavior

        Attributes:
            xi_p_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\xi^{\perp} := \dfrac{1}{p}`
            xiv_pz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\xi^{\downarrow} := -\dfrac{1}{p_z}`
            p_xi_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`{p} = \dfrac{1}{\xi^{\perp}}`
            pz_xiv_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`{p_z} = -\dfrac{1}{\xi^{\downarrow}}`
        """
        logging.info('core.xi.define_xi_eqns')
        self.xi_p_eqn = Eq(xi, 1/p)
        self.xiv_pz_eqn = (Eq(xiv, -1/pz))
        self.p_xi_eqn = Eq(p, solve(self.xi_p_eqn, p)[0])
        self.pz_xiv_eqn = Eq(pz, solve(self.xiv_pz_eqn, pz)[0])

    def define_xi_model_eqn(self) -> None:
        r"""
        Define the form of the surface erosion model,
        giving the speed of surface motion in its normal direction
        :math:`\xi^{\perp}``.
        For now, the model must have a separable dependence on
        position :math:`\mathbf{r}` and surface tilt :math:`\beta`.
        The former sets the 'flow' dependence of the erosion model,
        and is given by
        the function :math:`\varphi(\mathbf{})`,
        which must be specified at some point.
        The latter is specified in `self.beta_type` and may be `sin` or `tan`;
        it is given a power exponent :math:`\eta` which must
        take a rational value.

        Attributes:
            xi_varphi_beta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\xi^{\perp}
                = \varphi(\mathbf{r}) \, \left| \sin\beta \right|^\eta`
        """
        logging.info('core.xi.define_xi_model_eqn')
        if self.beta_type == 'sin':
            xi_model = varphi_r(rvec)*abs(sin(beta))**eta
        else:
            xi_model = varphi_r(rvec)*abs(tan(beta))**eta
        self.xi_varphi_beta_raw_eqn = Eq(xi, xi_model)
        # self.xi_varphi_beta_eqn = Eq(xi, xi_model)
        if self.do_raw:
            self.xi_varphi_beta_eqn = self.xi_varphi_beta_raw_eqn
        else:
            self.xi_varphi_beta_eqn \
                = Eq(self.xi_varphi_beta_raw_eqn.lhs,
                     self.xi_varphi_beta_raw_eqn.rhs.subs({eta: self.eta_}))

    def define_xi_related_eqns(self) -> None:
        r"""
        Define equations related to surface erosion speed :math:`\xi`
        and its vertical behavior.
        The derivations below are for an erosion model with
        :math:`\left|\tan\beta\right|^\eta`, :math:`\eta=3/2`.

        Attributes:
            xiv_varphi_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\xi^{\downarrow}
                = - \dfrac{p_{x}^{\eta}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{\tfrac{1}{2}
                - \tfrac{\eta}{2}} \varphi{\left(\mathbf{r} \right)}}{p_{z}}`

            px_xiv_varphi_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\left(\xi^{\downarrow}\right)^{2}`
                :math:`\left(p_{x}^{2} p_{x}^{2 \eta} \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta}
                \varphi^{2}{\left(\mathbf{r} \right)} - 1 \
                + \dfrac{p_{x}^{2 \eta} \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta}
                \varphi^{2}{\left(\mathbf{r} \right)}}
                {\left(\xi^{\downarrow}\right)^{2}}\right)`
                :math:`\times\,\,\,\left(p_{x}^{2} p_{x}^{2 \eta}
                \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta}
                \varphi^{2}{\left(\mathbf{r} \right)} + 1 \
                + \dfrac{p_{x}^{2 \eta} \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta}
                \varphi^{2}{\left(\mathbf{r} \right)}}
                {\left(\xi^{\downarrow}\right)^{2}}\right) = 0`

            eta_dbldenom  (:class:`~sympy.core.numbers.Integer`) :
                a convenience variable, recording double the denominator of
                :math:`\eta`, which must itself be a rational number
        """
        logging.info('core.xi.define_xi_related_eqns')
        eta_dbldenom = 2*denom(self.eta_)
        self.xiv_varphi_pxpz_eqn = simplify(
            Eq(xiv, (self.xi_varphi_beta_eqn.rhs/cos(beta))
                .subs(e2d(self.tanbeta_pxpz_eqn))
                .subs(e2d(self.cosbeta_pxpz_eqn))
                .subs(e2d(self.sinbeta_pxpz_eqn))
                .subs({Abs(px): px}))
        )
        xiv_eqn = self.xiv_varphi_pxpz_eqn
        px_xiv_varphi_eqn = simplify(
            Eq((xiv_eqn.subs({Abs(px): px})).rhs**eta_dbldenom
                - xiv_eqn.lhs**eta_dbldenom, 0)
            .subs(e2d(self.pz_xiv_eqn))
        )
        # HACK!!
        # Get rid of xiv**2 multiplier...
        #    should be a cleaner way of doing this
        self.px_xiv_varphi_eqn = factor(Eq(px_xiv_varphi_eqn.lhs/xiv**2, 0))
        self.eta__dbldenom = eta_dbldenom


#

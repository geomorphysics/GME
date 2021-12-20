"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`sympy`
  -  `GMPLib`_
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix: https://docs.sympy.org/latest/modules/matrices\
/immutablematrices.html

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
from sympy import Eq, sqrt, simplify, solve, Matrix, \
                  sin, cos, trigsimp

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import \
    p, r, rx, rz, px, pz, pcovec, alpha, beta

warnings.filterwarnings("ignore")

__all__ = ['RpMixin']


class RpMixin:
    r"""
    """

    def define_p_eqns(self) -> None:
        r"""
        Define normal slowness :math:`p` and derive related equations

        Attributes:
            p_covec_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\mathbf{\widetilde{p}} := [p_x, p_z]`
            px_p_beta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p_x = p \sin\beta`
            pz_p_beta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p_z = p \cos\beta`
            p_norm_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p = \sqrt{p_x^2+p_z^2}`
            tanbeta_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan\beta = -\dfrac{p_x}{p_z}`
            sinbeta_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\sin\beta = \dfrac{p_x}{\sqrt{p_x^2+p_z^2}}`
            cosbeta_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\cos\beta = \dfrac{-p_z}{\sqrt{p_x^2+p_z^2}}`
            pz_px_tanbeta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p_z = -\dfrac{p_x}{\tan\beta}`
            px_pz_tanbeta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p_x = -{p_z}{\tan\beta}`
            p_pz_cosbeta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p = -\dfrac{p_z}{\cos\beta}`
        """
        logging.info('define_p_eqns')
        self.p_covec_eqn = Eq(pcovec, Matrix([px, pz]).T)
        self.px_p_beta_eqn = Eq(px, p*sin(beta))
        self.pz_p_beta_eqn = Eq(pz, -p*cos(beta))
        self.p_norm_pxpz_eqn \
            = Eq(trigsimp(sqrt(self.px_p_beta_eqn.rhs**2
                               + self.pz_p_beta_eqn.rhs**2)),
                 (sqrt(self.px_p_beta_eqn.lhs**2 + self.pz_p_beta_eqn.lhs**2)))
        self.tanbeta_pxpz_eqn \
            = Eq(simplify(-self.px_p_beta_eqn.rhs/self.pz_p_beta_eqn.rhs),
                 -self.px_p_beta_eqn.lhs/self.pz_p_beta_eqn.lhs)
        self.sinbeta_pxpz_eqn \
            = Eq(sin(beta),
                 solve(self.px_p_beta_eqn, sin(beta))[0]
                 .subs(e2d(self.p_norm_pxpz_eqn)))
        self.cosbeta_pxpz_eqn \
            = Eq(cos(beta),
                 solve(self.pz_p_beta_eqn, cos(beta))[0]
                 .subs(e2d(self.p_norm_pxpz_eqn)))
        self.pz_px_tanbeta_eqn = Eq(pz, solve(self.tanbeta_pxpz_eqn, pz)[0])
        self.px_pz_tanbeta_eqn = Eq(px, solve(self.tanbeta_pxpz_eqn, px)[0])
        self.p_pz_cosbeta_eqn = Eq(p, solve(self.pz_p_beta_eqn, p)[0])

    def define_r_eqns(self) -> None:
        r"""
        Define equations for ray position :math:`\vec{r}`

        Attributes:
            rx_r_alpha_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`r^x = r\cos\alpha`
            rz_r_alpha_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`r^z = r\sin\alpha`
        """
        logging.info('define_r_eqns')
        self.rx_r_alpha_eqn = Eq(rx, r*cos(alpha))
        self.rz_r_alpha_eqn = Eq(rz, r*sin(alpha))


#

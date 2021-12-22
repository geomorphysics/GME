r"""
---------------------------------------------------------------------

Derive equations for the geomorphic surface fundamental function
:math:`\mathcal{F}` and the corresponding Hamiltonian
:math:`\mathcal{H}`.

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

# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable

# Library
import warnings
import logging
# from typing import Dict, Type, Optional  # , Tuple, Any, List

# SymPy
from sympy import Eq, simplify, solve, Abs, sign

# GME
from gme.core.symbols import \
    Fstar, H, px, varphi, varphi_r, varphi_rx, rx, rvec

warnings.filterwarnings("ignore")

__all__ = ['FundamentalMixin']


class FundamentalMixin:
    r"""
    Fundamental function and Hamiltonian equations supplement
    to equation definition class.
    """
    # Prerequisites
    p_norm_pxpz_eqn: Eq
    p_varphi_pxpz_eqn: Eq
    varphi_rx_eqn: Eq

    def define_Fstar_eqns(self) -> None:
        r"""
        Define the fundamental function.

        Attributes:

            Okubo_Fstar_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\dfrac{\sqrt{p_{x}^{2} + p_{z}^{2}}}{F^{*}}
                = \dfrac{p_{x}^{- \eta}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{\eta}{2}}}
                {\varphi{\left(\mathbf{r} \right)}}`

            Fstar_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`F^{*} = p_{x}^{\eta}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{1}{2}-\frac{\eta}{2}}
                \varphi{\left(\mathbf{r} \right)}`
        """
        logging.info('core.fundamental.define_Fstar_eqns')
        # Note force px >= 0
        self.Okubo_Fstar_eqn: Eq \
            = simplify(
                Eq(self.p_norm_pxpz_eqn.rhs/Fstar, self.p_varphi_pxpz_eqn.rhs)
                .subs({Abs(px): px, sign(px): 1})
            )
        self.Fstar_eqn: Eq \
            = Eq(Fstar,
                 (solve(self.Okubo_Fstar_eqn, Fstar)[0])
                 .subs({varphi_rx(rx): varphi})
                 ).subs({Abs(px): px, sign(px): 1})

    def define_H_eqns(self) -> None:
        r"""
        Define the Hamiltonian.

        Attributes:

            H_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`H
                = \dfrac{p_{x}^{2 \eta}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta}
                \varphi^{2}{\left(\mathbf{r} \right)}}{2}`

            H_varphi_rx_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`H
                = \dfrac{\varphi_0^{2} p_{x}^{2 \eta} x_{1}^{- 4 \mu}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta}
                \left(\varepsilon x_{1}^{2 \mu} +
                \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2}}{2}`
        """
        logging.info('core.fundamental.define_H_eqns')
        self.H_eqn: Eq \
            = Eq(H, simplify(self.Fstar_eqn.rhs**2/2))

        self.H_varphi_rx_eqn: Eq \
            = simplify(self.H_eqn.subs(varphi_r(rvec), self.varphi_rx_eqn.rhs))


#

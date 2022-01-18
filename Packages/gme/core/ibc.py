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

# Library
import warnings
import logging
# from typing import Dict, Type, Optional  # , Tuple, Any, List

# SymPy
from sympy import Eq, sqrt,  simplify, expand, diff, \
                  tan, sin, cos, tanh, Abs

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import \
    x, rx, rz, rvec, px, pz, pz_0, beta, \
    varphi, rdotx, rdotz, varphi_r, xiv_0, Lc, \
    h, h_0, kappa_h, h_fn

warnings.filterwarnings("ignore")

__all__ = ['IbcMixin']


class IbcMixin:
    r"""
    Initial condition/boundary condition equations supplement
    to equation definition class.
    """
    # Prerequisites
    pz0_xiv0_eqn: Eq
    pzpx_unity_eqn: Eq
    rdot_p_unity_eqn: Eq
    rdotx_pxpz_eqn: Eq
    rdotz_pxpz_eqn: Eq
    ibc_type: str
    p_varphi_beta_eqn: Eq
    varphi_rx_eqn: Eq

    def prep_ibc_eqns(self) -> None:
        r"""
        Define boundary (ray initial) condition equations

        Attributes:

            pz0_xiv0_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p_{z_0} = - \dfrac{1}{\xi^{\downarrow{0}}}`

            pzpx_unity_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\varphi^{2} p_{x}^{2} p_{x}^{2 \eta} \left(p_{x}^{2}
                + p_{z}^{2}\right)^{- \eta} + \varphi^{2} p_{x}^{2 \eta}
                p_{z}^{2} \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} = 1`
        """
        logging.info('gme.core.ibc.prep_ibc_eqns')
        self.pz0_xiv0_eqn = Eq(pz_0, (-1/xiv_0))
        self.pzpx_unity_eqn = expand(simplify(
            self.rdot_p_unity_eqn
                .subs({rdotx: self.rdotx_pxpz_eqn.rhs,
                       rdotz: self.rdotz_pxpz_eqn.rhs})
                .subs({varphi_r(rvec): varphi})
        )).subs({Abs(pz): -pz})

    def define_ibc_eqns(self) -> None:
        r"""
        Define initial profile equations

        Attributes:
            boundary_eqns (`dict` of :class:`~sympy.core.relational.Equality`):

                'planar':
                    'h': :math:`h = \dfrac{h_{0} x}{x_{1}}`

                    'gradh': :math:`\dfrac{d}{d x} h{\left(x \right)}
                    = \dfrac{h_{0}}{x_{1}}`

                'convex-up':
                    'h': :math:`h = \dfrac{h_{0} \tanh{\left(\dfrac{\kappa_
                    \mathrm{h} x}{x_{1}} \right)}}{\tanh{\left(\dfrac{\kappa_
                    \mathrm{h}}{x_{1}} \right)}}`

                    'gradh': :math:`\dfrac{d}{d x} h{\left(x \right)}
                    = \dfrac{\kappa_\mathrm{h} h_{0} \left(1 - \tanh^{2}{\left(
                    \dfrac{\kappa_\mathrm{h} x}{x_{1}} \right)}\right)}{x_{1}
                    \tanh{\left(\dfrac{\kappa_\mathrm{h}}{x_{1}} \right)}}`

                'concave-up':
                    'h': :math:`h = h``_{0} \left(1 + \dfrac{\tanh{\left(
                    \dfrac{\kappa_\mathrm{h} x}{x_{1}} - \kappa_\mathrm{h}
                    \right)}}{\tanh{\left(\dfrac{\kappa_\mathrm{h}}{x_{1}}
                    \right)}}\right)```

                    'gradh': :math:`\dfrac{d}{d x} h{\left(x \right)}
                    = \dfrac{\kappa_\mathrm{h} h_{0} \left(1 -
                    \tanh^{2}{\left(\dfrac{\kappa_\mathrm{h} x}{x_{1}}
                    - \kappa_\mathrm{h} \right)}\right)}{x_{1}
                    \tanh{\left(\dfrac{\kappa_\mathrm{h}}{x_{1}} \right)}}`
        """
        logging.info('gme.core.ibc.define_ibc_eqns')
        self.boundary_eqns = {
            'planar': {'h': Eq(h, (h_0*x/Lc))},
            'convex-up': {'h':
                          simplify(Eq(h,
                                      h_0*tanh(kappa_h*x/Lc)
                                      / tanh(kappa_h/Lc)))},
            'concave-up': {'h':
                           simplify(Eq(h,
                                       h_0+h_0*tanh(-kappa_h*(Lc-x)/Lc)
                                       / tanh(kappa_h/Lc)))}
        }
        # Math concave-up not geo concave-up, i.e., with minimum
        # Math convex-up not geo convex-up, i.e., with maximum
        for ibc_type in ('planar', 'convex-up', 'concave-up'):
            self.boundary_eqns[ibc_type].update({
                'gradh': Eq(diff(h_fn, x),
                            diff(self.boundary_eqns[ibc_type]['h'].rhs, x))
                })

    def set_ibc_eqns(self) -> None:
        r"""
        Define initial condition equations

        Attributes:

            rz_initial_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`{r}^z = \dfrac{h_{0} \tanh{\left(\frac{
                \kappa_\mathrm{h} {r}^x}{x_{1}} \right)}}{\tanh{\left(
                \frac{\kappa_\mathrm{h}}{x_{1}} \right)}}`

            tanbeta_initial_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\beta \right)} = \dfrac{\kappa_\mathrm{h}
                h_{0} \left(1 - \tanh^{2}{\left(\frac{\kappa_
                \mathrm{h} x}{x_{1}} \right)}\right)}{x_{1}
                \tanh{\left(\frac{\kappa_\mathrm{h}}{x_{1}} \right)}}`

            p_initial_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p = \dfrac{x_{1}^{2 \mu} \left|{\sin{\left(\beta
                \right)}}\right|^{- \eta}}{\varphi_0 \left(\varepsilon
                x_{1}^{2 \mu} + \left(- x + x_{1}\right)^{2 \mu}\right)}`

            px_initial_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p_{x} = \dfrac{\kappa_\mathrm{h} h_{0} x_{1}^{2 \mu}
                \left(\dfrac{1}{\kappa_\mathrm{h} h_{0} \left|{\tanh^{2}{
                \left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)} - 1}\right|}
                \right)^{\eta} \left(\kappa_\mathrm{h}^{2} h_{0}^{2}
                \left(\tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}}
                \right)} - 1\right)^{2} + x_{1}^{2} \tanh^{2}{\left(
                \frac{\kappa_\mathrm{h}}{x_{1}} \right)}\right)^{\frac{\eta}{2}
                - \frac{1}{2}} \left|{\tanh^{2}{\left(\frac{\kappa_
                \mathrm{h} x}{x_{1}} \right)} - 1}\right|}{\varphi_0
                \left(\varepsilon x_{1}^{2 \mu}
                + \left(- x + x_{1}\right)^{2 \mu}\right)}`

            pz_initial_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`p_{z} = - \dfrac{x_{1}^{2 \mu + 1}
                \left(\dfrac{1}{\kappa_\mathrm{h} h_{0} \left|{\tanh^{2}{
                \left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)} - 1}
                \right|}\right)^{\eta} \left(\kappa_\mathrm{h}^{2} h_{0}^{2}
                \left(\tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}}
                \right)} - 1\right)^{2} + x_{1}^{2} \tanh^{2}{\left(\frac{
                \kappa_\mathrm{h}}{x_{1}} \right)}\right)^{\frac{\eta}{2}
                - \frac{1}{2}} \tanh{\left(\frac{\kappa_\mathrm{h}}{x_{1}}
                \right)}}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} +
                \left(- x + x_{1}\right)^{2 \mu}\right)}`
        """
        logging.info('gme.core.ibc.set_ibc_eqns')
        cosbeta_eqn = Eq(cos(beta), 1/sqrt(1+tan(beta)**2))
        sinbeta_eqn = Eq(sin(beta), sqrt(1-1/(1+tan(beta)**2)))
        # sintwobeta_eqn = Eq(sin(2*beta), cos(beta)**2-sin(beta)**2)

        ibc_type = self.ibc_type
        self.rz_initial_eqn \
            = self.boundary_eqns[ibc_type]['h'].subs({h: rz, x: rx})
        self.tanbeta_initial_eqn \
            = Eq(tan(beta), self.boundary_eqns[ibc_type]['gradh'].rhs)
        self.p_initial_eqn = simplify(
            self.p_varphi_beta_eqn
            .subs(e2d(self.varphi_rx_eqn))
            # .subs({varphi_r(rvec):self.varphi_rx_eqn.rhs})
            .subs({self.tanbeta_initial_eqn.lhs: self.tanbeta_initial_eqn.rhs})
            .subs({rx: x})
        )
        self.px_initial_eqn = Eq(px, simplify(
            (+self.p_initial_eqn.rhs*sin(beta))
            .subs(e2d(sinbeta_eqn))
            .subs(e2d(cosbeta_eqn))
            .subs({tan(beta): self.tanbeta_initial_eqn.rhs, rx: x})))
        self.pz_initial_eqn = Eq(pz, simplify(
            (-self.p_initial_eqn.rhs*cos(beta))
            .subs(e2d(sinbeta_eqn))
            .subs(e2d(cosbeta_eqn))
            .subs({tan(beta): self.tanbeta_initial_eqn.rhs, rx: x})))

#

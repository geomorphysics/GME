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
.. _Equality: https://docs.sympy.org/latest/modules/core.html\
    #sympy.core.relational.Equality

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
from sympy import Eq, Rational, sqrt, simplify, solve, integrate, \
                  sin, cos, tan, exp, Abs

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import \
    p, rx, px, pz, beta, varphi_r, varphi_0, rvec, x, varepsilon, mu, Lc, \
    chi, x_h, xiv, x_sigma

warnings.filterwarnings("ignore")

__all__ = ['VarphiMixin']


class VarphiMixin:
    r"""
    Flow model :math:`\varphi` equations supplement
    to equation definition class.
    """
    # Prerequisites
    varphi_type: str
    mu_: float
    p_xi_eqn: Eq
    xi_varphi_beta_eqn: Eq
    tanbeta_pxpz_eqn: Eq
    sinbeta_pxpz_eqn: Eq
    p_norm_pxpz_eqn: Eq
    pz_px_tanbeta_eqn: Eq
    xi_p_eqn: Eq
    p_pz_cosbeta_eqn: Eq

    def define_varphi_model_eqns(self) -> None:
        r"""
        Define flow component of erosion model function

        Args:
            do_new: use new form of flow model

        Attributes:
            varphi_model_ramp_eqn (`Equality`_):
                :math:`\varphi{\left(\mathbf{r} \right)}
                = \varphi_0 \left(\varepsilon
                + \left(\dfrac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)`

            varphi_model_sramp_eqn (`Equality`_):
                    :math:`` TBD

            varphi_rx_eqn (`Equality`_):
                specific choice of :math:`\varphi` model from the above

                - pure "channel" model `varphi_model_ramp_eqn`
                    if `self.varphi_type=='ramp'`

                - "hillslope-channel" model `varphi_model_sramp_eqn`
                    if `self.varphi_type=='ramp-flat'`

        """
        logging.info('core.varphi.define_varphi_model_eqns')
        # The implicit assumption here is that upstream area A ~ x^2,
        #   which will not be true for a "hillslope" component,
        #   and for which we should have a transition to A ~ x
        self.varphi_model_ramp_eqn \
            = Eq(varphi_r(rvec),
                    varphi_0*(x+varepsilon)**(mu*2)).subs({x: Lc-rx})
        # else:
        #     self.varphi_model_ramp_eqn \
        #         = Eq(varphi_r(rvec),
        #              varphi_0*((x/Lc)**(mu*2)+varepsilon)).subs({x: Lc-rx})

        # self.varphi_model_rampmu_chi0_eqn \
        # =Eq(varphi_r, varphi_0*((x/Lc)**(mu*2) + varepsilon)).subs({x:Lc-rx})
        self.varphi_model_sramp_eqn = Eq(varphi_r(rvec), simplify(
            varphi_0*((chi/(Lc))*integrate(1/(1+exp(-x/x_sigma)), x) + 1)
            .subs({x: -rx+Lc})))
        smooth_step_fn = 1/(1+exp(((Lc-x_h)-x)/x_sigma))
        # smooth_break_fn = (1+(chi/(Lc))**mu*integrate(smooth_step_fn,x))
        # TODO: fix deprecated chi usage
        smooth_break_fn \
            = simplify(
                ((chi/Lc)*(integrate(smooth_step_fn, x))-chi*(1-x_h/Lc)+1)
                ** (mu*2)
            )
        self.varphi_model_srampmu_eqn \
            = Eq(varphi_r(rvec), simplify(
                    varphi_0*smooth_break_fn.subs({x: Lc-x}).subs({x: rx}))
                 )
        if self.varphi_type == 'ramp':
            varphi_model_eqn = self.varphi_model_ramp_eqn
        elif self.varphi_type == 'ramp-flat':
            if self.mu_ == Rational(1, 2):
                varphi_model_eqn = self.varphi_model_sramp_eqn
            else:
                varphi_model_eqn = self.varphi_model_srampmu_eqn
        else:
            raise ValueError('Unknown flow model')
        # self.varphi_rx_eqn =varphi_model_eqn.subs({varphi_r(rvec):varphi_rx})
        self.varphi_rx_eqn = varphi_model_eqn

    def define_varphi_related_eqns(self) -> None:
        r"""
        Define further equations related to normal slowness :math:`p`

        Attributes:
            p_varphi_beta_eqn (`Equality`_):
                :math:`p = \dfrac{1}{\varphi(\mathbf{r})|\sin\beta|^\eta}`

            p_varphi_pxpz_eqn (`Equality`_):
                :math:`\sqrt{p_{x}^{2} + p_{z}^{2}}
                = \dfrac{\left( {\sqrt{p_{x}^{2} + p_{z}^{2}}}  \right)^{\eta}}
                {\varphi{\left(\mathbf{r} \right)}{p_{x}}^\eta}`

            p_rx_pxpz_eqn (`Equality`_):
                :math:`\sqrt{p_{x}^{2} + p_{z}^{2}}
                = \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{\eta}{2}}}
                {\varphi_0 \left(\varepsilon x_{1}^{2 \mu}
                + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`

            p_rx_tanbeta_eqn (`Equality`_):
                :math:`\sqrt{p_{x}^{2} + \dfrac{p_{x}^{2}}{\tan^{2}
                {\left(\beta \right)}}}
                = \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu} \left(p_{x}^{2}
                + \dfrac{p_{x}^{2}}{\tan^{2}{\left(\beta
                \right)}}\right)^{\frac{\eta}{2}}}
                {\varphi_0 \left(\varepsilon x_{1}^{2 \mu}
                + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`

            px_beta_eqn (`Equality`_):
                :math:`p_{x}
                = \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu} \left(p_{x}^{2}
                + \dfrac{p_{x}^{2}}
                {\tan^{2}{\left(\beta \right)}}\right)^{\frac{\eta}{2}}
                \sin{\left(\beta \right)}}{\varphi_0
                \left(\varepsilon x_{1}^{2 \mu}
                + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`

            pz_beta_eqn (`Equality`_):
                :math:`p_{z}
                = - \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu} \left(p_{x}^{2}
                + \dfrac{p_{x}^{2}}{\tan^{2}
                {\left(\beta \right)}}\right)^{\frac{\eta}{2}}
                \cos{\left(\beta \right)}}
                {\varphi_0 \left(\varepsilon x_{1}^{2 \mu}
                + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`

            xiv_pxpz_eqn (`Equality`_):
                :math:`\xi^{\downarrow} = \dfrac{p_z}{p_x^{2} + p_z^{2}}`

            px_varphi_beta_eqn (`Equality`_):
                :math:`p_{x} = \dfrac{\sin{\left(\beta \right)}
                \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}
                {\varphi{\left(\mathbf{r} \right)}}`

            pz_varphi_beta_eqn (`Equality`_):
                :math:`p_{z} = - \dfrac{\cos{\left(\beta \right)}
                \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}
                {\varphi{\left(\mathbf{r} \right)}}`

            px_varphi_rx_beta_eqn (`Equality`_):
                :math:`p_{x} = \dfrac{\sin{\left(\beta \right)}
                \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}
                {\varphi_0 \left(\varepsilon
                + \left(\dfrac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)}`

            pz_varphi_rx_beta_eqn (`Equality`_):
                :math:`p_{z} = - \dfrac{\cos{\left(\beta \right)}
                \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}
                {\varphi_0 \left(\varepsilon
                + \left(\dfrac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)}`
        """
        logging.info('core.varphi.define_varphi_related_eqns')
        self.p_varphi_beta_eqn = self.p_xi_eqn.subs(
            e2d(self.xi_varphi_beta_eqn))
        # Note force px >= 0
        self.p_varphi_pxpz_eqn = (
            self.p_varphi_beta_eqn
            .subs(e2d(self.tanbeta_pxpz_eqn))
            .subs(e2d(self.sinbeta_pxpz_eqn))
            .subs(e2d(self.p_norm_pxpz_eqn))
            .subs({Abs(px): px})
        )
        # Don't do this simplification step because it messes up
        #    later calc of rdotz_on_rdotx_eqn etc
        # if self.eta_==1 and self.beta_type=='sin':
        #     self.p_varphi_pxpz_eqn \
        #   = simplify(Eq(self.p_varphi_pxpz_eqn.lhs/sqrt(px**2+pz**2),
        #           self.p_varphi_pxpz_eqn.rhs/sqrt(px**2+pz**2)))

        self.p_rx_pxpz_eqn = simplify(
                self.p_varphi_pxpz_eqn.subs(
                    {varphi_r(rvec): self.varphi_rx_eqn.rhs})
        )
        self.p_rx_tanbeta_eqn = self.p_rx_pxpz_eqn.subs(
            {pz: self.pz_px_tanbeta_eqn.rhs})
        self.px_beta_eqn = Eq(px, self.p_rx_tanbeta_eqn.rhs * sin(beta))
        self.pz_beta_eqn = Eq(pz, -self.p_rx_tanbeta_eqn.rhs * cos(beta))
        self.xiv_pxpz_eqn \
            = simplify(
                Eq(xiv, -cos(beta)/p)
                .subs({cos(beta): 1/sqrt(1+tan(beta)**2)})
                .subs({self.tanbeta_pxpz_eqn.lhs: self.tanbeta_pxpz_eqn.rhs})
                .subs({self.p_norm_pxpz_eqn.lhs: self.p_norm_pxpz_eqn.rhs})
                )

        tmp = self.xi_varphi_beta_eqn.subs(e2d(self.xi_p_eqn))\
                                     .subs(e2d(self.p_pz_cosbeta_eqn))
        self.pz_varphi_beta_eqn = Eq(pz, solve(tmp, pz)[0])
        tmp = self.pz_varphi_beta_eqn.subs(e2d(self.pz_px_tanbeta_eqn))
        self.px_varphi_beta_eqn = Eq(px, solve(tmp, px)[0])
        self.pz_varphi_rx_beta_eqn = self.pz_varphi_beta_eqn.subs(
            e2d(self.varphi_rx_eqn))
        self.px_varphi_rx_beta_eqn = self.px_varphi_beta_eqn.subs(
            e2d(self.varphi_rx_eqn))


#

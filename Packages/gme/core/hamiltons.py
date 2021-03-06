"""
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
.. _Equality:
    https://docs.sympy.org/latest/modules/core.html
    #sympy.core.relational.Equality

---------------------------------------------------------------------
"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable

# Library
import warnings
import logging
from typing import Callable, Optional

# SymPy
from sympy import Eq, simplify, Matrix, sin, cos, factor, diff, Abs

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import (
    rx,
    px,
    pz,
    alpha,
    rdot,
    rdotx,
    rdotz,
    pdotx,
    pdotz,
    rvec,
    rdotvec,
    varphi_r,
    pdotcovec,
)

warnings.filterwarnings("ignore")

__all__ = ["HamiltonsMixin"]


class HamiltonsMixin:
    """Hamilton's equations supplement to equation definition class."""

    # Prerequisites
    H_eqn: Eq
    px_pz_tanbeta_eqn: Eq
    H_varphi_rx_eqn: Eq
    varphi_rx_eqn: Eq
    tanbeta_pxpz_eqn: Eq
    vdotx_lambdified: Optional[Callable]
    vdotz_lambdified: Optional[Callable]

    # Definitions
    rdotx_rdot_alpha_eqn: Eq
    rdotz_rdot_alpha_eqn: Eq
    rdotx_pxpz_eqn: Eq
    rdotz_pxpz_eqn: Eq
    rdotz_on_rdotx_eqn: Eq
    rdotz_on_rdotx_tanbeta_eqn: Eq
    rdot_vec_eqn: Eq
    rdot_p_unity_eqn: Eq
    pdotx_pxpz_eqn: Eq
    pdotz_pxpz_eqn: Eq
    pdot_covec_eqn: Eq
    hamiltons_eqns: Matrix
    geodesic_eqns: Matrix  # dummy

    def define_rdot_eqns(self) -> None:
        r"""
        Define equations for :math:`\dot{r}`, the rate of change of position.

        Attributes:
            rdotx_rdot_alpha_eqn (`Equality`_):
                :math:`v^{x} = v \cos{\left(\alpha \right)}`

            rdotz_rdot_alpha_eqn (`Equality`_):
                :math:`v^{z} = v \sin{\left(\alpha \right)}`

            rdotx_pxpz_eqn (`Equality`_):
                :math:`v^{x} = p_{x}^{2 \eta - 1} \left(p_{x}^{2}
                + p_{z}^{2}\right)^{- \eta} \left(\eta p_{z}^{2}
                + p_{x}^{2}\right)
                \varphi^{2}{\left(\mathbf{r} \right)}`

            rdotz_pxpz_eqn (`Equality`_):
                :math:`v^{z} = - p_{x}^{2 \eta} p_{z} \left(\eta - 1\right)
                \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} \varphi^{2}
                {\left(\mathbf{r} \right)}`

            rdotz_on_rdotx_eqn (`Equality`_):
                :math:`\dfrac{v^{z}}{v^{x}} = - \dfrac{p_{x} p_{z}
                \left(\eta - 1\right)}{\eta p_{z}^{2} + p_{x}^{2}}`

            rdotz_on_rdotx_tanbeta_eqn (`Equality`_):
                :math:`\dfrac{v^{z}}{v^{x}} =   \dfrac{\left(\eta - 1\right)
                \tan{\left(\beta \right)}}{\eta + \tan^{2}
                {\left(\beta \right)}}`

            rdot_vec_eqn (`Equality`_):
                :math:`\mathbf{v} = \left[\begin{matrix}p_{x}^{2 \eta - 1}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta}
                \left(\eta p_{z}^{2}
                + p_{x}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}
                \\- p_{x}^{2 \eta} p_{z} \left(\eta - 1\right) \left(p_{x}^{2}
                + p_{z}^{2}\right)^{- \eta} \varphi^{2}{\left(\mathbf{r}
                \right)}\end{matrix}\right]`

            rdot_p_unity_eqn (`Equality`_):
                :math:`p_{x} v^{x} + p_{z} v^{z} = 1`
        """
        logging.info("gme.core.hamiltons.define_rdot_eqns")
        self.rdotx_rdot_alpha_eqn = Eq(rdotx, rdot * cos(alpha))
        self.rdotz_rdot_alpha_eqn = Eq(rdotz, rdot * sin(alpha))
        self.rdotx_pxpz_eqn = factor(Eq(rdotx, diff(self.H_eqn.rhs, px)))
        # simplify(diff(self.H_eqn.rhs,px)).subs({Abs(px):px,sign(px):1}) ) )
        self.rdotz_pxpz_eqn = factor(Eq(rdotz, diff(self.H_eqn.rhs, pz)))
        # self.rdotz_pxpz_eqn \
        #   = simplify( simplify( Eq( rdotz, simplify(diff(self.H_eqn.rhs,pz))\
        #                                 .subs({Abs(px):px,sign(px):1}) ) )
        #                                     .subs({px:pxp}) ) \
        #                                         .subs({pxp:px})
        self.rdotz_on_rdotx_eqn = factor(
            Eq(
                rdotz / rdotx,
                simplify((self.rdotz_pxpz_eqn.rhs / self.rdotx_pxpz_eqn.rhs)),
            ).subs({Abs(px): px})
        )
        self.rdotz_on_rdotx_tanbeta_eqn = factor(
            self.rdotz_on_rdotx_eqn.subs({px: self.px_pz_tanbeta_eqn.rhs})
        )
        self.rdot_vec_eqn = Eq(
            rdotvec, Matrix([self.rdotx_pxpz_eqn.rhs, self.rdotz_pxpz_eqn.rhs])
        )
        self.rdot_p_unity_eqn = Eq(rdotx * px + rdotz * pz, 1)

        self.vdotx_lambdified = None
        self.vdotz_lambdified = None

    def define_pdot_eqns(self) -> None:
        r"""
        Define eqns for :math:`\dot{p}`, the rate of change of normal slowness.

        Attributes:
            pdotx_pxpz_eqn (`Equality`_):
                :math:`\dot{p}_x = 2 \mu \varphi_0^{2}
                p_{x}^{2 \eta} x_{1}^{- 4 \mu} \left(p_{x}^{2}
                + p_{z}^{2}\right)^{1 - \eta}
                \left(x_{1} - {r}^x\right)^{2 \mu - 1}
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1}
                - {r}^x\right)^{2 \mu}\right)`

            pdotz_pxpz_eqn (`Equality`_):
                :math:`\dot{p}_z = 0`

            pdot_covec_eqn (`Equality`_):
                :math:`\mathbf{\dot{\widetilde{p}}}
                = \left[\begin{matrix}2 \mu \varphi_0^{2}
                p_{x}^{2 \eta} x_{1}^{- 4 \mu}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta}
                \left(x_{1} - {r}^x\right)^{2 \mu - 1}
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1}
                - {r}^x\right)^{2 \mu}\right) & 0\end{matrix}\right]`
        """
        logging.info("gme.core.hamiltons.define_pdot_eqns")
        self.pdotx_pxpz_eqn = simplify(
            Eq(pdotx, (-diff(self.H_varphi_rx_eqn.rhs, rx)))
        ).subs(
            {
                Abs(pz): -pz,
                Abs(px): px,
                Abs(px * pz): -px * pz,
                Abs(px / pz): -px / pz,
            }
        )
        self.pdotz_pxpz_eqn = simplify(
            Eq(
                pdotz,
                (
                    0
                    * diff(self.varphi_rx_eqn.rhs, rx)
                    * (-self.tanbeta_pxpz_eqn.rhs)
                    * self.H_eqn.rhs
                    / varphi_r(rvec)
                ),
            )
        )
        self.pdot_covec_eqn = Eq(
            pdotcovec,
            Matrix([[self.pdotx_pxpz_eqn.rhs], [self.pdotz_pxpz_eqn.rhs]]).T,
        )

    def define_Hamiltons_eqns(self) -> None:
        r"""
        Define Hamilton's equations.

        Attributes:
            hamiltons_eqns (`Matrix`_):
                :math:`\left[\begin{matrix}\
                \dot{r}^x = \varphi_0^{2} p_{x}^{2 \eta - 1} x_{1}^{- 4 \mu}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta}
                \left(\eta p_{z}^{2} + p_{x}^{2}\right)
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1}
                - {r}^x\right)^{2 \mu}\right)^{2}\\
                \dot{r}^z = - \varphi_0^{2} p_{x}^{2 \eta} p_{z}
                x_{1}^{- 4 \mu} \left(\eta - 1\right) \left(p_{x}^{2}
                + p_{z}^{2}\right)^{- \eta}
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1}
                - {r}^x\right)^{2 \mu}\right)^{2}\\
                \dot{p}_x = 2 \mu \varphi_0^{2} p_{x}^{2 \eta} x_{1}^{- 4 \mu}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta} \left(x_{1}
                - {r}^x\right)^{2 \mu - 1}
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1}
                - {r}^x\right)^{2 \mu}\right)\\
                \dot{p}_z = 0
                \end{matrix}\right]`
        """
        logging.info("gme.core.hamiltons.define_Hamiltons_eqns")
        self.hamiltons_eqns = Matrix(
            (
                self.rdotx_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn)),
                self.rdotz_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn)),
                self.pdotx_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn)),
                self.pdotz_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn))
                # .subs({pdotz:pdotz_tfn})
            )
        )


#

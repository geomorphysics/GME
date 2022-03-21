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

---------------------------------------------------------------------
"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable

# Library
import warnings
import logging

# from typing import Dict, Type, Optional  # , Tuple, Eq, List

# SymPy
from sympy import (
    Eq,
    Rational,
    simplify,
    solve,
    numer,
    denom,
    separatevars,
    poly,
)

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import (
    px,
    eta,
    varphi_0,
    varphi_r,
    rvec,
    xiv,
    xiv_0,
    xih_0,
)

warnings.filterwarnings("ignore")

__all__ = ["PxpolyMixin"]


class PxpolyMixin:
    r"""Polynomial :math:`p_x` eqns supplement to equation definition class."""

    # Prerequisites
    sinCi_xih0_eqn: Eq
    px_xiv_varphi_eqn: Eq
    varphi_rxhat_eqn: Eq
    px_pxhat_eqn: Eq
    xiv0_xih0_Ci_eqn: Eq
    varphi_rx_eqn: Eq

    # Definitions
    poly_pxhat_xiv_eqn: Eq
    poly_pxhat_xiv0_eqn: Eq
    poly_px_xiv_varphi_eqn: Eq
    poly_px_xiv_eqn: Eq
    poly_px_xiv0_eqn: Eq  # HACK

    def define_px_poly_eqn(
        self, eta_choice: Rational = None, do_ndim: bool = False
    ) -> None:
        r"""
        Define :math:`p_x` polynomial.

        TODO: remove ref to xiv_0

        Define polynomial form of function combining normal-slowness
        covector components :math:`(p_x,p_z)`
        (where the latter is given in terms of the vertical erosion rate
        :math:`\xi^{\downarrow} = -\dfrac{1}{p_z}`)
        and the erosion model flow component :math:`\varphi(\mathbf{r})`

        Args:
            eta_choice (:class:`~sympy.core.numbers.Rational`):
                value of :math:`\eta` to use instead value given at
                instantiation; otherwise the latter value is used

        Attributes:

            poly_px_xiv_varphi_eqn (:class:`~sympy.polys.polytools.Poly`):
                :math:`\operatorname{Poly}{\left(
                \left(\xi^{\downarrow}\right)^{4}
                \varphi^{4}{\left(\mathbf{r} \right)} p_{x}^{6}
                -  \left(\xi^{\downarrow}\right)^{4} p_{x}^{2}
                -  \left(\xi^{\downarrow}\right)^{2}, p_{x},
                domain=\mathbb{Z}\left[\varphi{\left(\mathbf{r} \right)},
                \xi^{\downarrow}\right] \right)}`

            poly_px_xiv_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\varphi_0^{4}
                \left(\xi^{\downarrow{0}}\right)^{4} p_{x}^{6}
                \left(\varepsilon
                + \left(\frac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)^{4}
                - \left(\xi^{\downarrow{0}}\right)^{4} p_{x}^{2}
                - \left(\xi^{\downarrow{0}}\right)^{2} = 0`
        """
        logging.info(f"gme.core.pxpoly.define_px_poly_eqn (ndim={do_ndim})")
        if do_ndim:
            # Non-dimensionalized version
            varphi0_solns = solve(
                self.sinCi_xih0_eqn.subs({eta: eta_choice}), varphi_0
            )
            varphi0_eqn = Eq(varphi_0, varphi0_solns[0])
            if eta_choice is not None and eta_choice <= 1:
                tmp_eqn = separatevars(
                    simplify(
                        self.px_xiv_varphi_eqn.subs({eta: eta_choice})
                        .subs({varphi_r(rvec): self.varphi_rxhat_eqn.rhs})
                        .subs(e2d(self.px_pxhat_eqn))
                        .subs(e2d(varphi0_eqn))
                    )
                )
                self.poly_pxhat_xiv_eqn = simplify(
                    Eq(
                        (
                            numer(tmp_eqn.lhs)
                            - denom(tmp_eqn.lhs) * (tmp_eqn.rhs)
                        )
                        / xiv ** 2,
                        0,
                    )
                )
                self.poly_pxhat_xiv0_eqn = self.poly_pxhat_xiv_eqn.subs(
                    {xiv: xiv_0}
                ).subs(e2d(self.xiv0_xih0_Ci_eqn))
            else:
                tmp_eqn = separatevars(
                    simplify(
                        self.px_xiv_varphi_eqn.subs({eta: eta_choice})
                        .subs({varphi_r(rvec): self.varphi_rxhat_eqn.rhs})
                        .subs(e2d(self.px_pxhat_eqn))
                        .subs(e2d(varphi0_eqn))
                    )
                )
                self.poly_pxhat_xiv_eqn = simplify(
                    Eq(
                        (
                            numer(tmp_eqn.lhs)
                            - denom(tmp_eqn.lhs) * (tmp_eqn.rhs)
                        )
                        / xiv ** 2,
                        0,
                    )
                )
                self.poly_pxhat_xiv0_eqn = simplify(
                    Eq(
                        self.poly_pxhat_xiv_eqn.lhs.subs({xiv: xiv_0}).subs(
                            e2d(self.xiv0_xih0_Ci_eqn)
                        )
                        / xih_0 ** 2,
                        0,
                    )
                )
        else:
            # Dimensioned version
            tmp_eqn = simplify(self.px_xiv_varphi_eqn.subs({eta: eta_choice}))
            if eta_choice is not None and eta_choice <= 1:
                self.poly_px_xiv_varphi_eqn = poly(tmp_eqn.lhs, px)
            else:
                self.poly_px_xiv_varphi_eqn = poly(
                    numer(tmp_eqn.lhs) - denom(tmp_eqn.lhs) * (tmp_eqn.rhs), px
                )
            self.poly_px_xiv_eqn = Eq(
                self.poly_px_xiv_varphi_eqn.subs(e2d(self.varphi_rx_eqn)), 0
            )


#

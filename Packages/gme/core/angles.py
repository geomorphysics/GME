r"""
Equation definitions for angular properties such as :math:`\alpha` and :math:`\beta`.

Requires Python packages/modules:
  -  :mod:`SymPy <sympy>`
  -  `GMPLib`_
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html
"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable

# Library
import warnings
import logging

# from typing import Dict, Type, Optional  # , Tuple, Any, List

# SymPy
import sympy as sy
from sympy import (
    Eq,
    sqrt,
    simplify,
    factor,
    solve,
    tan,
    numer,
    Abs,
    Piecewise,
    N,
    pi,
    atan,
    lambdify,
)

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import (
    pz,
    alpha,
    beta,
    rdotx,
    rdotz,
    ta,
    alpha_ext,
    eta,
    beta_crit,
    psi,
)

warnings.filterwarnings("ignore")

__all__ = ["AnglesMixin"]


class AnglesMixin:
    r"""Angle equations supplement to equation definition class."""

    # Prerequisites
    eta_: float
    beta_type: str
    rdotz_on_rdotx_eqn: Eq
    rdotz_rdot_alpha_eqn: Eq
    rdotx_rdot_alpha_eqn: Eq
    rdotz_on_rdotx_tanbeta_eqn: Eq
    pz_xiv_eqn: Eq

    # Definitions
    tanalpha_rdot_eqn: Eq
    tanalpha_pxpz_eqn: Eq
    tanalpha_beta_eqn: Eq
    tanbeta_alpha_eqns: Eq
    tanalpha_ext_eqns: Eq
    tanalpha_ext_eqn: Eq
    tanbeta_crit_eqns: Eq
    tanbeta_crit_eqn: Eq
    tanbeta_rdotxz_pz_eqn: Eq
    tanbeta_rdotxz_xiv_eqn: Eq
    tanalpha_ext: Eq
    tanbeta_crit: Eq
    psi_alpha_beta_eqn: Eq
    psi_alpha_eta_eqns: Eq
    psi_eta_beta_lambdas: Eq

    def define_tanalpha_eqns(self) -> None:
        r"""
        Define equations for ray angle :math:`\alpha`.

        Attributes:
            tanalpha_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\alpha \right)} = \dfrac{v^{z}}{v^{x}}`

            tanalpha_beta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\alpha \right)}
                = - \dfrac{p_{x} p_{z} \left(\eta - 1\right)}
                {\eta p_{z}^{2} + p_{x}^{2}}`

            tanalpha_rdot_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\alpha \right)}
                = \dfrac{\left(\eta - 1\right) \tan{\left(\beta \right)}}
                {\eta + \tan^{2}{\left(\beta \right)}}`
        """
        logging.info("gme.core.angles.define_tanalpha_eqns")
        self.tanalpha_rdot_eqn = Eq(
            simplify(
                self.rdotz_rdot_alpha_eqn.rhs / self.rdotx_rdot_alpha_eqn.rhs
            ),
            rdotz / rdotx,
        )

        self.tanalpha_pxpz_eqn = self.tanalpha_rdot_eqn.subs(
            {self.rdotz_on_rdotx_eqn.lhs: self.rdotz_on_rdotx_eqn.rhs}
        )

        self.tanalpha_beta_eqn = self.tanalpha_rdot_eqn.subs(
            {self.rdotz_on_rdotx_eqn.lhs: self.rdotz_on_rdotx_tanbeta_eqn.rhs}
        )

    def define_tanbeta_eqns(self) -> None:
        r"""
        Define equations for surface tilt angle :math:`\beta`.

        Attributes:
            tanbeta_alpha_eqns (list) :
                :math:`\left[ \tan{\left(\beta \right)} \
                = \dfrac{\eta - \sqrt{\eta^{2}
                - 4 \eta \tan^{2}{\left(\alpha \right)} - 2 \eta + 1} - 1}
                {2 \tan{\left(\alpha \right)}},
                \tan{\left(\beta \right)}
                = \dfrac{\eta + \sqrt{\eta^{2}
                - 4 \eta \tan^{2}{\left(\alpha \right)} - 2 \eta + 1} - 1}
                {2 \tan{\left(\alpha \right)}}\right]`

            tanbeta_alpha_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\beta \right)}
                = \dfrac{\eta - \sqrt{\eta^{2}
                - 4 \eta \tan^{2}{\left(\alpha \right)} - 2 \eta + 1} - 1}
                {2 \tan{\left(\alpha \right)}}`

            tanalpha_ext_eqns (list) :
                :math:`\left[ \tan{\left(\alpha_c \right)}
                = - \frac{\sqrt{\eta - 2 + \frac{1}{\eta}}}{2},
                \tan{\left(\alpha_c \right)}
                = \frac{\sqrt{\eta - 2 + \frac{1}{\eta}}}{2}\right]`

            tanalpha_ext_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\alpha_c \right)}
                = \dfrac{\eta - 1}{2 \sqrt{\eta}}`

            tanbeta_crit_eqns (list) :
                :math:`\left[ \tan{\left(\beta_c \right)}
                = - \dfrac{\eta - 1}{\sqrt{\eta - 2 + \frac{1}{\eta}}},
                \tan{\left(\beta_c \right)}
                = \dfrac{\eta - 1}{\sqrt{\eta - 2 + \frac{1}{\eta}}}\right]`

            tanbeta_crit_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\beta_c \right)} = \sqrt{\eta}`

            tanbeta_rdotxz_pz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\beta \right)}
                = \dfrac{v^{z} - \frac{1}{p_{z}}}{v^{x}}`

            tanbeta_rdotxz_xiv_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\beta \right)}
                = \dfrac{\xi^{\downarrow} + v^{z}}{v^{x}}`
        """
        logging.info("gme.core.angles.define_tanbeta_eqns")

        # eta_sub = {eta: self.eta_}

        if self.eta_ == 1 and self.beta_type == "sin":
            logging.info(
                r"Cannot compute all $\beta$ equations "
                + r"for $\sin\beta$ model and $\eta=1$"
            )
            return
        solns = solve(self.tanalpha_beta_eqn.subs({tan(alpha): ta}), tan(beta))
        self.tanbeta_alpha_eqns = [
            Eq(tan(beta), soln.subs({ta: tan(alpha)})) for soln in solns
        ]
        # Bit of a hack - extracts the square root term in tan(beta)
        #    as a fn of tan(alpha), which then gives the critical alpha
        root_terms = [
            [
                arg_
                for arg__ in arg_.args
                if isinstance(arg__, sy.core.power.Pow)
                or isinstance(arg_, sy.core.power.Pow)
            ]
            for arg_ in numer(self.tanbeta_alpha_eqns[0].rhs).args
            if isinstance(arg_, (sy.core.mul.Mul, sy.core.power.Pow))
        ]
        self.tanalpha_ext_eqns = [
            Eq(tan(alpha_ext), soln)
            for soln in solve(Eq(root_terms[0][0], 0), tan(alpha))
        ]

        tac_lt1 = simplify(
            (
                factor(simplify(self.tanalpha_ext_eqns[0].rhs * sqrt(eta)))
                / sqrt(eta)
            ).subs({Abs(eta - 1): 1 - eta})
        )
        tac_gt1 = simplify(
            (
                factor(simplify(self.tanalpha_ext_eqns[1].rhs * sqrt(eta)))
                / sqrt(eta)
            ).subs({Abs(eta - 1): eta - 1})
        )
        self.tanalpha_ext_eqn = Eq(
            tan(alpha_ext), Piecewise((tac_lt1, eta < 1), (tac_gt1, True))
        )

        self.tanbeta_crit_eqns = [
            factor(
                tanbeta_alpha_eqn_.subs(
                    {beta: beta_crit, alpha: alpha_ext}
                ).subs(e2d(tanalpha_ext_eqn_))
            )
            for tanalpha_ext_eqn_, tanbeta_alpha_eqn_ in zip(
                self.tanalpha_ext_eqns, self.tanbeta_alpha_eqns
            )
        ]
        # This is a hack, because SymPy simplify can't handle it
        self.tanbeta_crit_eqn = Eq(
            tan(beta_crit), sqrt(simplify((self.tanbeta_crit_eqns[0].rhs) ** 2))
        )

        self.tanbeta_rdotxz_pz_eqn = Eq(tan(beta), (rdotz - 1 / pz) / rdotx)

        self.tanbeta_rdotxz_xiv_eqn = self.tanbeta_rdotxz_pz_eqn.subs(
            {pz: self.pz_xiv_eqn.rhs}
        )

        self.tanalpha_ext = float(
            N(self.tanalpha_ext_eqn.rhs.subs({eta: self.eta_}))
        )

        self.tanbeta_crit = float(
            N(self.tanbeta_crit_eqn.rhs.subs({eta: self.eta_}))
        )

    def define_psi_eqns(self) -> None:
        r"""
        Define equation for anisotropy angle :math:`\psi`.

        Attributes:
            psi_alpha_beta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\beta \right)}
                = \dfrac{\xi^{\downarrow} + v^{z}}{v^{x}}`
        """
        logging.info("gme.core.angles.define_psi_eqns")
        self.psi_alpha_beta_eqn = Eq(psi, alpha - beta + pi / 2)
        self.psi_alpha_eta_eqns = [
            self.psi_alpha_beta_eqn.subs({beta: atan(tanbeta_alpha_eqn_.rhs)})
            for tanbeta_alpha_eqn_ in self.tanbeta_alpha_eqns
        ]
        self.psi_eta_beta_lambdas = [
            lambdify((eta, alpha), psi_alpha_eta_eqn_.rhs)
            for psi_alpha_eta_eqn_ in self.psi_alpha_eta_eqns
        ]


#

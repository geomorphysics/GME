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

# from typing import Dict

# SymPy
from sympy import (
    Eq,
    sqrt,
    simplify,
    solve,
    factor,
    expand_power_base,
    Abs,
    tan,
    sin,
    cos,
)

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import (
    eta,
    beta,
    xiv,
    varphi_r,
    rvec,
    xiv_0,
    varepsilon,
    varepsilonhat,
    xhat,
    rx,
    Lc,
    dzdx,
    pzhat_0,
    xivhat_0,
)

warnings.filterwarnings("ignore")

__all__ = ["ProfileMixin"]


class ProfileMixin:
    """Generate differential eqn(s) for time-invariant topographic profile."""

    # Definitions
    xiv_eqn: Eq
    xvi_abs_eqn: Eq
    dzdx_Ci_polylike_eqn: Eq
    dzdx_polylike_eqn: Eq
    dzdx_Ci_polylike_prelim_eqn: Eq

    def define_z_eqns(self) -> None:
        r"""
        Form a polynomial-type ODE in $\hat{z}(\hat{x})$.

        Attributes:
            dzdx_Ci_polylike_eqn
        """
        logging.info("gme.core.profile.define_z_eqns")
        tmp_eqn = Eq(
            (xiv / varphi_r(rvec)),
            solve(
                simplify(
                    self.pz_varphi_beta_eqn.subs(e2d(self.pz_xiv_eqn)).subs(
                        {Abs(sin(beta)): sin(beta)}
                    )
                ),
                xiv,
            )[0]
            / varphi_r(rvec),
        )
        self.xiv_eqn = tmp_eqn.subs({sin(beta): sqrt(1 - cos(beta) ** 2)}).subs(
            {cos(beta): sqrt(1 / (1 + tan(beta) ** 2))}
        )
        self.xvi_abs_eqn = factor(
            Eq(self.xiv_eqn.lhs ** 4, (self.xiv_eqn.rhs) ** 4)
        )

        self.dzdx_polylike_eqn = self.xvi_abs_eqn.subs({tan(beta): dzdx})

        self.dzdx_Ci_polylike_prelim_eqn = (
            expand_power_base(
                self.dzdx_polylike_eqn.subs(e2d(self.varphi_rx_eqn))
                .subs(e2d(self.varphi0_Lc_xiv0_Ci_eqn))
                .subs({xiv: xiv_0})
                .subs({varepsilon: varepsilonhat * Lc})
                .subs({rx: xhat * Lc})
                .subs(
                    {
                        (Lc * varepsilonhat - Lc * xhat + Lc): (
                            Lc * (varepsilonhat - xhat + 1)
                        )
                    }
                )
            )
        ).subs({Abs(dzdx): dzdx})

        self.dzdx_Ci_polylike_eqn = Eq(
            self.dzdx_Ci_polylike_prelim_eqn.rhs
            * ((dzdx ** 2 + 1) ** (2 * eta))
            - self.dzdx_Ci_polylike_prelim_eqn.lhs
            * ((dzdx ** 2 + 1) ** (2 * eta)),
            0,
        ).subs({pzhat_0: 1 / xivhat_0})


#

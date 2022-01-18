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
from typing import Dict

# SymPy
from sympy import Eq, Rational, sqrt, simplify, factor, solve, Matrix, \
                  sin, cos, tan, asin, Abs, deg, diff

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import \
    rx, rz, px, pz, beta, Lc, rxhat, rzhat, rvec, \
    varepsilon, varepsilonhat, varphi_0, varphi_r, varphi_rxhat_fn, \
    xiv, xih, xiv_0, xih_0, beta_0, \
    eta, mu, H, Ci, \
    th_0, tv_0, t, that, pxhat, pzhat, \
    rdotxhat_thatfn, rdotzhat_thatfn, pdotxhat_thatfn, pdotzhat_thatfn

warnings.filterwarnings("ignore")

__all__ = ['NdimMixin']


class NdimMixin:
    r"""
    Non-dimensionalization supplement to equation definition class.
    """
    # Prerequisites
    varphi_rx_eqn: Eq
    xi_varphi_beta_eqn: Eq
    pz_xiv_eqn: Eq
    H_varphi_rx_eqn: Eq

    def nondimensionalize(self) -> None:
        r"""
        Non-dimensionalize variables, Hamiltonian, and Hamilton's equations;
        define dimensionless channel incision number :math:`\mathsf{Ci}`.

        Attributes:
        """
        logging.info('gme.core.ndim.nondimensionalize')
        varsub: Dict = {}

        self.rx_rxhat_eqn = Eq(rx, Lc*rxhat)
        self.rz_rzhat_eqn = Eq(rz, Lc*rzhat)

        self.varepsilon_varepsilonhat_eqn \
            = Eq(varepsilon, Lc*varepsilonhat)
        self.varepsilonhat_varepsilon_eqn \
            = Eq(varepsilonhat,
                 solve(self.varepsilon_varepsilonhat_eqn, varepsilonhat)[0])

        self.varphi_rxhat_eqn \
            = Eq(varphi_rxhat_fn(rxhat),
                 factor(self.varphi_rx_eqn.rhs
                        .subs(e2d(self.rx_rxhat_eqn))
                        .subs(e2d(self.varepsilon_varepsilonhat_eqn))
                        .subs(varsub)))

        self.xi_rxhat_eqn\
            = simplify(self.xi_varphi_beta_eqn
                       .subs({varphi_r(rvec): varphi_rxhat_fn(rxhat)})
                       .subs(e2d(self.varphi_rxhat_eqn)))

        self.xih0_beta0_eqn \
            = simplify(Eq(xih_0,
                          (self.xi_rxhat_eqn.rhs / sin(beta_0))
                          .subs(e2d(self.varphi_rx_eqn))
                          .subs({Abs(sin(beta)): sin(beta_0)})
                          .subs({rxhat: 0}).subs(varsub)))
        self.xiv0_beta0_eqn \
            = simplify(Eq(xiv_0,
                          (self.xi_rxhat_eqn.rhs / cos(beta_0))
                          .subs(e2d(self.varphi_rx_eqn))
                          .subs({Abs(sin(beta)): sin(beta_0)})
                          .subs({rxhat: 0}).subs(varsub)))
        self.xih0_xiv0_beta0_eqn \
            = Eq((xih_0),
                 xiv_0*simplify((xih_0/xiv_0)
                                .subs(e2d(self.xiv0_beta0_eqn))
                                .subs(e2d(self.xih0_beta0_eqn))))
        self.xih_xiv_tanbeta_eqn = Eq(xih, xiv/tan(beta))
        self.xiv_xih_tanbeta_eqn = Eq(
            xiv, solve(self.xih_xiv_tanbeta_eqn, xiv)[0])
        # Eq(xiv, xih*tan(beta))

        self.th0_xih0_eqn = Eq(th_0, (Lc/xih_0))
        self.tv0_xiv0_eqn = Eq(tv_0, (Lc/xiv_0))

        self.th0_beta0_eqn \
            = factor(simplify(
                                self.th0_xih0_eqn
                                .subs(e2d(self.xih0_beta0_eqn))
                ))
        self.tv0_beta0_eqn = simplify(
            self.tv0_xiv0_eqn.subs(e2d(self.xiv0_beta0_eqn)))

        self.t_that_eqn = Eq(t, th_0*that)
        self.px_pxhat_eqn = Eq(px, pxhat/xih_0)
        self.pz_pzhat_eqn = Eq(pz, pzhat/xih_0)

        self.pzhat_xiv_eqn \
            = Eq(pzhat,
                 solve(self.pz_xiv_eqn.subs(e2d(self.pz_pzhat_eqn)), pzhat)[0])

        self.H_varphi_rxhat_eqn = factor(simplify(
            self.H_varphi_rx_eqn
                .subs(varsub)
                .subs(e2d(self.varepsilon_varepsilonhat_eqn))
                .subs(e2d(self.rx_rxhat_eqn))
                .subs(e2d(self.px_pxhat_eqn))
                .subs(e2d(self.pz_pzhat_eqn))
        ))

        self.H_split = (
            2*pxhat**(-2*eta)
            * (pxhat**2+pzhat**2)**(eta-1)
            * (1-rxhat+varepsilonhat)**(-4*mu)
        )
        self.H_Ci_eqn = Eq(H, simplify(((sin(Ci))**(2*(1-eta))/self.H_split)
                                       .subs(e2d(self.H_varphi_rxhat_eqn))))
        self.degCi_H0p5_eqn \
            = Eq(Ci,
                 deg(solve(self.H_Ci_eqn.subs({H: Rational(1, 2)}), Ci)[0]))
        self.sinCi_xih0_eqn \
            = Eq(sin(Ci), (((sqrt(simplify(
                    (H*self.H_split)
                    .subs(e2d(self.H_varphi_rxhat_eqn))
                )))**(1/(1-eta)))))
        self.Ci_xih0_eqn = Eq(Ci, asin(self.sinCi_xih0_eqn.rhs))
        self.sinCi_beta0_eqn \
            = Eq(sin(Ci),
                 factor(simplify(self.sinCi_xih0_eqn.rhs.subs(
                  e2d(self.xih0_beta0_eqn))))
                 .subs({sin(beta_0)*sin(beta_0)**(-eta):
                        (sin(beta_0)**(1-eta))})
                 .subs({(sin(beta_0)**(1-eta))**(-1/(eta-1)): sin(beta_0)}))
        self.Ci_beta0_eqn = Eq(Ci, asin(self.sinCi_beta0_eqn.rhs))
        self.beta0_Ci_eqn = Eq(beta_0, solve(self.Ci_beta0_eqn, beta_0)[1])

        self.rdotxhat_eqn \
            = Eq(rdotxhat_thatfn(that),
                 simplify(diff(self.H_Ci_eqn.rhs, pxhat)))
        self.rdotzhat_eqn \
            = Eq(rdotzhat_thatfn(that),
                 simplify(diff(self.H_Ci_eqn.rhs, pzhat)))

        self.pdotxhat_eqn \
            = Eq(pdotxhat_thatfn(that),
                 simplify(-diff(self.H_Ci_eqn.rhs, rxhat)))
        self.pdotzhat_eqn \
            = Eq(pdotzhat_thatfn(that),
                 simplify(-diff(self.H_Ci_eqn.rhs, rzhat)))

        self.xih0_Ci_eqn = factor(
            self.xih0_beta0_eqn.subs(e2d(self.beta0_Ci_eqn)))

        # .subs({varepsilonhat:0})
        self.xih0_Lc_varphi0_Ci_eqn = self.xih0_Ci_eqn
        self.xiv0_xih0_Ci_eqn \
            = self.xiv_xih_tanbeta_eqn \
            .subs({xiv: xiv_0, xih: xih_0, beta: beta_0}) \
            .subs(e2d(self.beta0_Ci_eqn))
        # .subs({varepsilonhat:0})
        self.xiv0_Lc_varphi0_Ci_eqn \
            = simplify(self.xiv0_xih0_Ci_eqn
                       .subs(e2d(self.xih0_Lc_varphi0_Ci_eqn)))
        self.varphi0_Lc_xiv0_Ci_eqn \
            = Eq(varphi_0, solve(self.xiv0_Lc_varphi0_Ci_eqn, varphi_0)[0]) \
            .subs({Abs(cos(Ci)): cos(Ci)})

        self.ratio_xiv0_xih0_eqn \
            = Eq(xiv_0/xih_0, (xiv_0/xih_0).subs(e2d(self.xiv0_xih0_Ci_eqn)))

    def define_nodimensionalized_Hamiltons_eqns(self) -> None:
        """
        Group Hamilton's equations into matrix form
        """
        logging.info('gme.core.ndim.define_nodimensionalized_Hamiltons_eqns')
        self.hamiltons_ndim_eqns = Matrix((
                                        self.rdotxhat_eqn.rhs,
                                        self.rdotzhat_eqn.rhs,
                                        self.pdotxhat_eqn.rhs,
                                        self.pdotzhat_eqn.rhs
                                    ))


#

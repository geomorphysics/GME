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
from functools import reduce

# Typing
from typing import Dict

# SymPy
from sympy import Eq, Rational, simplify, expand_trig, sqrt, solve, \
                  sin, cos, tan, diff, Matrix, numer, denom, lambdify, \
                  derive_by_array

# GMPLib
from gmplib.utils import e2d

# GME
from gme.core.symbols import \
    rx, rz, px, pz, beta, eta, rvec, varphi, varphi_r, \
    mu, alpha, ta, rdotx, rdotz, varepsilon

warnings.filterwarnings("ignore")

__all__ = ['GeodesicMixin']


class GeodesicMixin:
    r"""
    Geodesic equations supplement to equation definition class.
    """

    def prep_geodesic_eqns(self, parameters: Dict = None):
        r"""
        Define geodesic equations.

        Args:
            parameters:
                dictionary of model parameter values to be used for
                equation substitutions

        Attributes:
            gstar_ij_tanbeta_mat (`Matrix`_):
                :math:`\dots`

            g_ij_tanbeta_mat (`Matrix`_):
                :math:`\dots`

            tanbeta_poly_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\dots` where :math:`a := \tan\alpha`

            tanbeta_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\tan{\left(\beta \right)} = \dots` where
                :math:`a := \tan\alpha`

            gstar_ij_tanalpha_mat (`Matrix`_):
                a symmetric tensor with components (using shorthand
                :math:`a := \tan\alpha`)

                :math:`g^*[1,1] = \dots`

                :math:`g^*[1,2] = g^*[2,1] = \dots`

                :math:`g^*[2,2] = \dots`

            gstar_ij_mat (`Matrix`_):
                a symmetric tensor with components
                (using shorthand :math:`a := \tan\alpha`, and with a
                particular choice of model parameters)

                :math:`g^*[1,1] = \dots`

                :math:`g^*[1,2] = g^*[2,1] = \dots`

                :math:`g^*[2,2] =  \dots`

            g_ij_tanalpha_mat (`Matrix`_):
                a symmetric tensor with components (using shorthand
                :math:`a := \tan\alpha`)

                :math:`g[1,1] = \dots`

                :math:`g[1,2] = g[2,1] = \dots`

                :math:`g[2,2] = \dots`

            g_ij_mat (`Matrix`_):
                a symmetric tensor with components
                (using shorthand :math:`a := \tan\alpha`, and with a
                particular choice of model parameters)

                :math:`g[1,1] =\dots`

                :math:`g[1,2] = g[2,1] = \dots`

                :math:`g[2,2] = \dots`

            g_ij_mat_lambdified   (function) :
                lambdified version of `g_ij_mat`

            gstar_ij_mat_lambdified (function) :
                lambdified version of `gstar_ij_mat`
        """
        logging.info('core.geodesic.prep_geodesic_eqns')
        self.gstar_ij_tanbeta_mat = None
        self.g_ij_tanbeta_mat = None
        self.tanbeta_poly_eqn = None
        self.tanbeta_eqn = None
        self.gstar_ij_tanalpha_mat = None
        self.gstar_ij_mat = None
        self.g_ij_tanalpha_mat = None
        self.g_ij_mat = None
        self.g_ij_mat_lambdified = None
        self.gstar_ij_mat_lambdified = None

        mu_eta_sub = {mu: self.mu_, eta: self.eta_}

        # if parameters is None: return
        H_ = self.H_eqn.rhs.subs(mu_eta_sub)
        # Assume indexing here ranges in [1,2]
        def p_i_lambda(i): return [px, pz][i-1]
        # r_i_lambda = lambda i: [rx, rz][i-1]
        # rdot_i_lambda = lambda i: [rdotx, rdotz][i-1]

        def gstar_ij_lambda(i, j): return simplify(
            Rational(2, 2)*diff(diff(H_, p_i_lambda(i)), p_i_lambda(j))
        )
        gstar_ij_mat = Matrix([[gstar_ij_lambda(1, 1), gstar_ij_lambda(2, 1)],
                               [gstar_ij_lambda(1, 2), gstar_ij_lambda(2, 2)]])
        gstar_ij_pxpz_mat = gstar_ij_mat.subs({varphi_r(rvec): varphi})
        g_ij_pxpz_mat = gstar_ij_mat.inv().subs({varphi_r(rvec): varphi})

        cosbeta_eqn = Eq(cos(beta), 1/sqrt(1+tan(beta)**2))
        sinbeta_eqn = Eq(sin(beta), sqrt(1-1/(1+tan(beta)**2)))
        sintwobeta_eqn = Eq(sin(2*beta), cos(beta)**2-sin(beta)**2)

        self.gstar_ij_tanbeta_mat = expand_trig(simplify(
            gstar_ij_pxpz_mat.subs(e2d(self.px_pz_tanbeta_eqn))
        )).subs(e2d(cosbeta_eqn))
        self.g_ij_tanbeta_mat = expand_trig(simplify(
            g_ij_pxpz_mat.subs(e2d(self.px_pz_tanbeta_eqn))
        )).subs(e2d(cosbeta_eqn))

        tanalpha_beta_eqn = self.tanalpha_beta_eqn.subs(mu_eta_sub)
        tanbeta_poly_eqn \
            = Eq(numer(tanalpha_beta_eqn.rhs)
                 - tanalpha_beta_eqn.lhs*denom(tanalpha_beta_eqn.rhs), 0) \
            .subs({tan(alpha): ta})

        tanbeta_eqn = (Eq(tan(beta), solve(tanbeta_poly_eqn, tan(beta))[0]))
        self.tanbeta_poly_eqn = tanbeta_poly_eqn
        self.tanbeta_eqn = tanbeta_eqn

        # Replace all refs to beta with refs to alpha
        self.gstar_ij_tanalpha_mat = (
            self.gstar_ij_tanbeta_mat
                .subs(e2d(sintwobeta_eqn))
                .subs(e2d(sinbeta_eqn))
                .subs(e2d(cosbeta_eqn))
                .subs(e2d(tanbeta_eqn))
        ).subs(mu_eta_sub)
        self.gstar_ij_mat = (
            self.gstar_ij_tanalpha_mat
                .subs({ta: tan(alpha)})
                .subs(e2d(self.tanalpha_rdot_eqn))
                .subs(e2d(self.varphi_rx_eqn.subs({varphi_r(rvec): varphi})))
                .subs(parameters)
        ).subs(mu_eta_sub)
        self.g_ij_tanalpha_mat = (
            expand_trig(self.g_ij_tanbeta_mat)
            .subs(e2d(sintwobeta_eqn))
            .subs(e2d(sinbeta_eqn))
            .subs(e2d(cosbeta_eqn))
            .subs(e2d(tanbeta_eqn))
        ).subs(mu_eta_sub)
        self.g_ij_mat = (
            self.g_ij_tanalpha_mat
                .subs({ta: rdotz/rdotx})
                .subs(e2d(self.varphi_rx_eqn.subs({varphi_r(rvec): varphi})))
                .subs(parameters)
        ).subs(mu_eta_sub)
        self.g_ij_mat_lambdified \
            = lambdify((rx, rdotx, rdotz, varepsilon),
                       self.g_ij_mat,
                       'numpy')
        self.gstar_ij_mat_lambdified \
            = lambdify((rx, rdotx, rdotz, varepsilon),
                       self.gstar_ij_mat,
                       'numpy')

    def define_geodesic_eqns(self):
        r"""
        Define geodesic equations

        Attributes:
            dg_rk_ij_mat (`Matrix`_):
                Derivatives of the components of the metric tensor:
                these values are used to construct the Christoffel tensor.
                Too unwieldy to display here.

            christoffel_ij_k_rx_rdot_lambda   (function) :
                The Christoffel tensor coefficients, as a `lambda` function,
                for each component :math:`r^x`, :math:`{\dot{r}^x}`
                and :math:`{\dot{r}^z}`.

            christoffel_ij_k_lambda   (function) :
                The Christoffel tensor coefficients, as a `lambda` function,
                in a compact and indexable form.

            geodesic_eqns (list of :class:`~sympy.core.relational.Equality`) :
                Ray geodesic equations, but expressed indirectly as
                a pair of coupled 1st-order vector ODEs
                rather than a 2nd-order vector ODE for ray acceleration.
                The 1st-order ODE form is easier to solve numerically.

                :math:`\dot{r}^x = v^{x}`

                :math:`\dot{r}^z = v^{z}`

                :math:`\dot{v}^x = \dots`

                :math:`\dot{v}^z = \dots`

            vdotx_lambdified (function) :
                lambdified version of :math:`\dot{v}^x`

            vdotz_lambdified (function) :
                lambdified version of :math:`\dot{v}^z`
        """
        logging.info('core.geodesic.define_geodesic_eqns')
        self.dg_rk_ij_mat = None
        self.christoffel_ij_k_rx_rdot_lambda = None
        self.christoffel_ij_k_lambda = None
        self.geodesic_eqns = None
        self.vdotx_lambdified = None
        self.vdotz_lambdified = None
        # if self.eta_>=1 and self.beta_type=='sin':
        #     print(r'Cannot compute geodesic equations
        #               for $\sin\beta$ model and $\eta>=1$')
        #     return
        # eta_sub = {eta: self.eta_}

        # Manipulate metric tensors
        def gstar_ij_lambda(i_, j_): return self.gstar_ij_mat[i_, j_]
        # g_ij_lambda = lambda i_,j_: self.g_ij_mat[i_,j_]
        r_k_mat = Matrix([rx, rz])
        self.dg_rk_ij_mat = (derive_by_array(self.g_ij_mat, r_k_mat))

        def dg_ij_rk_lambda(i_, j_, k_):
            return self.dg_rk_ij_mat[k_, 0, i_, j_]

        # Generate Christoffel "symbols" tensor
        def christoffel_ij_k_raw(i_, j_, k_):
            return [
                Rational(1, 2)*gstar_ij_lambda(k_, m_)*(
                                        dg_ij_rk_lambda(m_, i_, j_)
                                        + dg_ij_rk_lambda(m_, j_, i_)
                                        - dg_ij_rk_lambda(i_, j_, m_)
                                                )
                for m_ in [0, 1]
            ]
        # Use of 'factor' here messes things up for eta<1
        self.christoffel_ij_k_rx_rdot_lambda = lambda i_, j_, k_: \
            (reduce(lambda a, b: a+b, christoffel_ij_k_raw(i_, j_, k_)))
        christoffel_ij_k_rx_rdot_list = [
            [
                [
                    lambdify((rx, rdotx, rdotz, varepsilon),
                             self.christoffel_ij_k_rx_rdot_lambda(i_, j_, k_))
                    for i_ in [0, 1]]
                for j_ in [0, 1]]
            for k_ in [0, 1]
        ]
        self.christoffel_ij_k_lambda \
            = lambda i_, j_, k_, varepsilon_: \
            christoffel_ij_k_rx_rdot_list[i_][j_][k_]

        # Obtain geodesic equations as a set of coupled 1st order ODEs
        self.geodesic_eqns = Matrix([
            rdotx,
            rdotz,
            # Use symmetry to abbreviate sum of diagonal terms
            (-self.christoffel_ij_k_rx_rdot_lambda(0, 0, 0)*rdotx*rdotx
             - 2*self.christoffel_ij_k_rx_rdot_lambda(0, 1, 0)*rdotx*rdotz
             # -christoffel_ij_k_rx_rdot_lambda(1,0,0)*rdotz*rdotx
             - self.christoffel_ij_k_rx_rdot_lambda(1, 1, 0)*rdotz*rdotz),
            # Use symmetry to abbreviate sum of diagonal terms
            (-self.christoffel_ij_k_rx_rdot_lambda(0, 0, 1)*rdotx*rdotx
             - 2*self.christoffel_ij_k_rx_rdot_lambda(0, 1, 1)*rdotx*rdotz
             # -christoffel_ij_k_rx_rdot_lambda(1,0,1)*rdotz*rdotx
             - self.christoffel_ij_k_rx_rdot_lambda(1, 1, 1)*rdotz*rdotz)
        ])
# self.geodesic_eqns = Matrix([
#     Eq(rdotx_true, rdotx),
#     Eq(rdotz_true, rdotz),
#     # Use symmetry to abbreviate sum of diagonal terms
#     Eq(vdotx, (-self.christoffel_ij_k_rx_rdot_lambda(0,0,0)*rdotx*rdotx
#                -2*self.christoffel_ij_k_rx_rdot_lambda(0,1,0)*rdotx*rdotz
#                #-christoffel_ij_k_rx_rdot_lambda(1,0,0)*rdotz*rdotx
#                -self.christoffel_ij_k_rx_rdot_lambda(1,1,0)*rdotz*rdotz) ),
#     # Use symmetry to abbreviate sum of diagonal terms
#     Eq(vdotz, (-self.christoffel_ij_k_rx_rdot_lambda(0,0,1)*rdotx*rdotx
#                -2*self.christoffel_ij_k_rx_rdot_lambda(0,1,1)*rdotx*rdotz
#                #-christoffel_ij_k_rx_rdot_lambda(1,0,1)*rdotz*rdotx
#                -self.christoffel_ij_k_rx_rdot_lambda(1,1,1)*rdotz*rdotz) )
# ])
# Use of 'factor' here messes things up for eta<1
        self.vdotx_lambdified \
            = lambdify((rx, rdotx, rdotz, varepsilon),
                       (self.geodesic_eqns[2]), 'numpy')
        self.vdotz_lambdified \
            = lambdify((rx, rdotx, rdotz, varepsilon),
                       (self.geodesic_eqns[3]), 'numpy')

#

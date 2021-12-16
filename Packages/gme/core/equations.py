"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

This module provides a derivation of GME theory using :mod:`SymPy <sympy>`
for a 2D slice of a 3D landscape along a channel transect.

Starting from a model equation for the surface-normal erosion rate in terms of
the tilt angle of the topographic surface and the distance downstream
(used to infer the flow component of the model erosion process), we derive
the fundamental function (of a co-Finsler metric space), the corresponding
Hamiltonian,
and the equivalent metric tensor.
The rest of the equations are derived from these core results.

---------------------------------------------------------------------


Requires Python packages/modules:
  -  :mod:`scipy`, :mod:`sympy`
  -  :mod:`gmplib.utils`
  -  :mod:`gme.core.symbols`

---------------------------------------------------------------------

"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable
import warnings
from functools import reduce

# Typing
from typing import Dict, Type, Optional  # , Tuple, Any, List

# SymPy
import sympy as sy
from sympy import Eq, Rational, N, \
                    pi, sqrt, numer, denom, \
                    simplify, trigsimp, factor, expand, lambdify, expand_trig, \
                    solve, diff, Matrix, \
                    exp, tan, sin, cos, asin, deg, tanh,  \
                    Abs, sign, re, im, \
                    integrate, derive_by_array, Piecewise, \
                    poly, separatevars

# GMPLib
from gmplib.utils import e2d, omitdict

# GME
from gme.core.symbols import \
    p, r, x, rx, rz, rvec, px, pz, pcovec, pz_0, \
    rdotx, rdotz, rdot, rdotvec, pdotx, pdotz, pdotcovec,  \
    pxhat, pzhat, rxhat, rzhat, \
    rdotx_tfn, rdotxhat_thatfn, rdotzhat_thatfn, rdotz_tfn, \
    pdotx_tfn, pdotxhat_thatfn, pdotzhat_thatfn, pdotz_tfn, \
    varphi, varphi_rxhat_fn, \
    varphi_r, varphi_0, varphi_rx, varphi_rhat, \
    xi, xiv, xiv_0, xih_0, xih, xivhat, chi, Lc, mu, eta, \
    alpha, alpha_ext, ta, beta, beta_0, beta_crit, varepsilon, varepsilonhat, \
    x_sigma, x_h, Fstar,\
    t, th_0, tv_0, that, \
    H, Ci, psi, gstar, det_gstar, \
    g, h, h_0, kappa_h, h_fn

warnings.filterwarnings("ignore")

__all__ = ['Equations', 'EquationSubset']


class Equations:
    r"""
    Class to solve the set of GME equations (using :mod:`SymPy <sympy>`)
    and to provide them in a form (sometimes lambdified) that can be used for
    numerical evaluation.

    Much of the derivation sequence here keeps :math:`\eta` and :math:`\mu`
    unspecified, up until the Hamiltonian is defined, but eventually values
    for these parameters need to be substituted in order to make further
    progress. In this documentation, we set :math:`\eta=3/2`, for now.

    TODO: provide solutions for both :math:`\eta=1/2` and
          :math:`\eta=3/2` where appropriate.

    """

    def __init__(self,
                 parameters: Optional[Dict] = None,
                 eta_: Rational = Rational(3, 2),
                 mu_: Rational = Rational(3, 4),
                 beta_type: str = 'sin',
                 varphi_type: str = 'ramp',
                 ibc_type: str = 'convex-up',
                 do_raw: bool = True,
                 do_idtx: bool = False,
                 do_geodesic: bool = False,
                 do_nothing: bool = False,
                 do_new_varphi_model: bool = True) -> None:
        r"""
        Initialize class instance.
        Define/derive all the GME equations (unless `'do_nothing'` is true)
        using :mod:`SymPy <sympy>`.

        Args:
            parameters (dict): dictionary of model parameter values to be
                               used for equation substitutions
                               (used when defining geodesic equations)
            eta\_ (:class:`sympy.Rational <sympy.core.numbers.Rational>`):
                exponent in slope component of erosion model (equivalent of
                gradient exponent :math:`n` in SPIM)
            mu\_ (:class:`sympy.Rational <sympy.core.numbers.Rational>`):
                exponent in flow component of erosion model
                (equivalent of area exponent :math:`m` in SPIM)
            beta_type (str): choice of slope component of erosion model
                             (`'sin'` or `'tan'`)
            varphi_type (str): choice of flow component of erosion model
                               (`'ramp'` or `'ramp-flat'`)
            ibc_type (str): choice of initial boundary shape
                            (`'convex-up'` or `'concave-up'`,
                            i.e., concave vs convex in mathematical parlance)
            do_raw (bool): suppress substitution of :math:`eta` value
                            when defining `xi_varphi_beta_eqn`?
            do_idtx (bool): generate indicatrix and figuratrix equations?
            do_geodesic (bool): generate geodesic equations?
            do_nothing (bool):
                just create the class instance and set its data,
                 but don't run any of the equation definition methods
            do_new_varphi_model (bool): use new form of varphi model?

        Attributes:
            GME equations
            (:class:`sympy.Eq <sympy.core.relational.Equality>` etc):
                See below
        """

        self.eta_ = eta_
        self.mu_ = mu_
        self.do_raw = do_raw
        self.ibc_type = ibc_type
        self.beta_type = beta_type
        self.varphi_type = varphi_type
        if do_nothing:
            return

        self.define_p_eqns()
        self.define_r_eqns()
        self.define_xi_eqns()
        self.define_xi_model_eqn()
        self.define_xi_related_eqns()
        self.define_varphi_model_eqn(do_new=do_new_varphi_model)
        self.define_varphi_related_eqns()
        self.define_Fstar_eqns()
        self.define_H_eqns()
        self.define_rdot_eqns()
        self.define_pdot_eqns()
        self.define_Hamiltons_eqns()
        self.nondimensionalize()
        self.define_nodimensionalized_Hamiltons_eqns()
        self.define_tanalpha_eqns()
        self.define_tanbeta_eqns()
        self.define_psi_eqns()
        self.define_g_eqns()
        if do_idtx:
            self.define_idtx_fgtx_eqns()
        if do_geodesic:
            self.prep_geodesic_eqns(parameters if not do_raw else None)
            self.define_geodesic_eqns()  # parameters if not do_raw else None)
        self.define_px_poly_eqn(eta_choice=self.eta_, do_ndim=False)
        self.define_px_poly_eqn(eta_choice=self.eta_, do_ndim=True)
        self.prep_ibc_eqns()
        self.define_ibc_eqns()
        self.set_ibc_eqns()

    def define_p_eqns(self) -> None:
        r"""
        Define normal slowness :math:`p` and derive related equations

        Attributes:
            p_covec_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\mathbf{\widetilde{p}} := [p_x, p_z]`
            px_p_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_x = p \sin\beta`
            pz_p_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_z = p \cos\beta`
            p_norm_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p = \sqrt{p_x^2+p_z^2}`
            tanbeta_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan\beta = -\dfrac{p_x}{p_z}`
            sinbeta_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\sin\beta = \dfrac{p_x}{\sqrt{p_x^2+p_z^2}}`
            cosbeta_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\cos\beta = \dfrac{-p_z}{\sqrt{p_x^2+p_z^2}}`
            pz_px_tanbeta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_z = -\dfrac{p_x}{\tan\beta}`
            px_pz_tanbeta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_x = -{p_z}{\tan\beta}`
            p_pz_cosbeta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p = -\dfrac{p_z}{\cos\beta}`
        """
        self.p_covec_eqn = Eq(pcovec, Matrix([px, pz]).T)
        self.px_p_beta_eqn = Eq(px, p*sin(beta))
        self.pz_p_beta_eqn = Eq(pz, -p*cos(beta))
        self.p_norm_pxpz_eqn \
            = Eq(trigsimp(sqrt(self.px_p_beta_eqn.rhs**2 + self.pz_p_beta_eqn.rhs**2)),
                 (sqrt(self.px_p_beta_eqn.lhs**2 + self.pz_p_beta_eqn.lhs**2)))
        self.tanbeta_pxpz_eqn \
            = Eq(simplify(-self.px_p_beta_eqn.rhs/self.pz_p_beta_eqn.rhs),
                 -self.px_p_beta_eqn.lhs/self.pz_p_beta_eqn.lhs)
        self.sinbeta_pxpz_eqn = Eq(sin(beta),
                                   solve(self.px_p_beta_eqn, sin(beta))[0].subs(e2d(self.p_norm_pxpz_eqn)))
        self.cosbeta_pxpz_eqn = Eq(cos(beta),
                                   solve(self.pz_p_beta_eqn, cos(beta))[0].subs(e2d(self.p_norm_pxpz_eqn)))
        self.pz_px_tanbeta_eqn = Eq(pz, solve(self.tanbeta_pxpz_eqn, pz)[0])
        self.px_pz_tanbeta_eqn = Eq(px, solve(self.tanbeta_pxpz_eqn, px)[0])
        self.p_pz_cosbeta_eqn = Eq(p, solve(self.pz_p_beta_eqn, p)[0])

    def define_r_eqns(self) -> None:
        r"""
        Define equations for ray position :math:`\vec{r}`

        Attributes:
            rx_r_alpha_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`r^x = r\cos\alpha`
            rz_r_alpha_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`r^z = r\sin\alpha`
        """
        self.rx_r_alpha_eqn = Eq(rx, r*cos(alpha))
        self.rz_r_alpha_eqn = Eq(rz, r*sin(alpha))

    def define_xi_eqns(self) -> None:
        r"""
        Define equations for surface erosion speed :math:`\xi` and its vertical behavior

        Attributes:
            xi_p_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\xi^{\perp} := \dfrac{1}{p}`
            xiv_pz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\xi^{\downarrow} := -\dfrac{1}{p_z}`
            p_xi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`{p} = \dfrac{1}{\xi^{\perp}}`
            pz_xiv_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`{p_z} = -\dfrac{1}{\xi^{\downarrow}}`
        """
        self.xi_p_eqn = Eq(xi, 1/p)
        self.xiv_pz_eqn = (Eq(xiv, -1/pz))
        self.p_xi_eqn = Eq(p, solve(self.xi_p_eqn, p)[0])
        self.pz_xiv_eqn = Eq(pz, solve(self.xiv_pz_eqn, pz)[0])

    def define_xi_model_eqn(self) -> None:
        r"""
        Define the form of the surface erosion model,
        giving the speed of surface motion in its normal direction :math:`\xi^{\perp}``.
        For now, the model must have a separable dependence on
        position :math:`\mathbf{r}` and surface tilt :math:`\beta`.
        The former sets the 'flow' dependence of the erosion model, and is given by
        the function :math:`\varphi(\mathbf{})`, which must be specified at some point.
        The latter is specified in `self.beta_type` and may be `sin` or `tan`;
        it is given a power exponent :math:`\eta` which must take a rational value.

        Attributes:
            xi_varphi_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\xi^{\perp} = \varphi(\mathbf{r}) \, \left| \sin\beta \right|^\eta`
        """
        if self.beta_type == 'sin':
            xi_model = varphi_r(rvec)*abs(sin(beta))**eta
        else:
            xi_model = varphi_r(rvec)*abs(tan(beta))**eta
        self.xi_varphi_beta_raw_eqn = Eq(xi, xi_model)
        # self.xi_varphi_beta_eqn = Eq(xi, xi_model)
        if self.do_raw:
            self.xi_varphi_beta_eqn = self.xi_varphi_beta_raw_eqn
        else:
            self.xi_varphi_beta_eqn \
                = Eq(self.xi_varphi_beta_raw_eqn.lhs,
                     self.xi_varphi_beta_raw_eqn.rhs.subs({eta: self.eta_}))

    def define_xi_related_eqns(self) -> None:
        r"""
        Define equations related to surface erosion speed :math:`\xi`
        and its vertical behavior.
        The derivations below are for an erosion model with
        :math:`\left|\tan\beta\right|^\eta`, :math:`\eta=3/2`.

        Attributes:
            xiv_varphi_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\xi^{\downarrow} = - \dfrac{p_{x}^{\eta} \left(p_{x}^{2} + p_{z}^{2}\right)^{\tfrac{1}{2} - \tfrac{\eta}{2}} \varphi{\left(\mathbf{r} \right)}}{p_{z}}`
            px_xiv_varphi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\left(\xi^{\downarrow}\right)^{2}`
                :math:`\left(p_{x}^{2} p_{x}^{2 \eta} \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta} \varphi^{2}{\left(\mathbf{r} \right)} - 1 \
                + \dfrac{p_{x}^{2 \eta} \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta} \varphi^{2}{\left(\mathbf{r} \right)}}{\left(\xi^{\downarrow}\right)^{2}}\right)`
                :math:`\times\,\,\,\left(p_{x}^{2} p_{x}^{2 \eta} \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta} \varphi^{2}{\left(\mathbf{r} \right)} + 1 \
                + \dfrac{p_{x}^{2 \eta} \left(p_{x}^{2} \
                + \frac{1}{\left(\xi^{\downarrow}\right)^{2}}\right)^{- \eta} \varphi^{2}{\left(\mathbf{r} \right)}}{\left(\xi^{\downarrow}\right)^{2}}\right) = 0`
            eta_dbldenom   (:class:`sympy.Int <sympy.core.numbers.Integer>`) :
                a convenience variable, recording double the denominator of :math:`\eta`, which must itself be a rational number
        """
        eta_dbldenom = 2*denom(self.eta_)
        self.xiv_varphi_pxpz_eqn = simplify(
            Eq(xiv, (self.xi_varphi_beta_eqn.rhs/cos(beta))
                .subs(e2d(self.tanbeta_pxpz_eqn))
               .subs(e2d(self.cosbeta_pxpz_eqn))
               .subs(e2d(self.sinbeta_pxpz_eqn))
               .subs({Abs(px): px}))
        )
        xiv_eqn = self.xiv_varphi_pxpz_eqn
        px_xiv_varphi_eqn = simplify(
            Eq((xiv_eqn.subs({Abs(px): px})).rhs**eta_dbldenom
                - xiv_eqn.lhs**eta_dbldenom, 0)
            .subs(e2d(self.pz_xiv_eqn))
        )
        # HACK!!  Get rid of xiv**2 multiplier... should be a cleaner way of doing this
        self.px_xiv_varphi_eqn = factor(Eq(px_xiv_varphi_eqn.lhs/xiv**2, 0))
        self.eta__dbldenom = eta_dbldenom

    def define_varphi_model_eqn(self, do_new: bool = True) -> None:
        r"""
        Define flow component of erosion model function


        Attributes:
            varphi_model_ramp_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\varphi{\left(\mathbf{r} \right)} = \varphi_0 \left(\varepsilon + \left(\dfrac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)`
            varphi_model_rampflat_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                    :math:`` TBD
            varphi_rx_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                specific choice of :math:`\varphi` model from the above

                - pure "channel" model `varphi_model_ramp_eqn` if `self.varphi_type=='ramp'`

                - "hillslope-channel" model `varphi_model_rampflat_eqn` if `self.varphi_type=='ramp-flat'`

        """
        # The implicit assumption here is that upstream area A ~ x^2,
        #   which will not be true for a "hillslope" component,
        #   and for which we should have a transition to A ~ x
        if do_new:
            self.varphi_model_ramp_eqn \
                = Eq(varphi_r(rvec), varphi_0*(x+varepsilon)**(mu*2)).subs({x: Lc-rx})
        else:
            self.varphi_model_ramp_eqn \
                = Eq(varphi_r(rvec), varphi_0*((x/Lc)**(mu*2)+varepsilon)).subs({x: Lc-rx})

        # self.varphi_model_rampmu_chi0_eqn \
        #  = Eq(varphi_r, varphi_0*((x/Lc)**(mu*2) + varepsilon)).subs({x:Lc-rx})
        self.varphi_model_rampflat_eqn = Eq(varphi_r(rvec), simplify(
            varphi_0*((chi/(Lc))*integrate(1/(1+exp(-x/x_sigma)), x) + 1)
            .subs({x: -rx+Lc})))
        smooth_step_fn = 1/(1+exp(((Lc-x_h)-x)/x_sigma))
        # smooth_break_fn = (1+(chi/(Lc))**mu*integrate(smooth_step_fn,x))
        # TODO: fix deprecated chi usage
        smooth_break_fn \
            = simplify(
                ((chi/(Lc))*(integrate(smooth_step_fn, x))-chi*(1-x_h/Lc)+1)**(mu*2)
            )
        self.varphi_model_rampflatmu_eqn \
            = Eq(varphi_r(rvec), simplify(
                    varphi_0*smooth_break_fn.subs({x: Lc-x}).subs({x: rx}))
                 )
        if self.varphi_type == 'ramp':
            varphi_model_eqn = self.varphi_model_ramp_eqn
        elif self.varphi_type == 'ramp-flat':
            if self.mu_ == Rational(1, 2):
                varphi_model_eqn = self.varphi_model_rampflat_eqn
            else:
                varphi_model_eqn = self.varphi_model_rampflatmu_eqn
        else:
            raise ValueError('Unknown flow model')
        # self.varphi_rx_eqn = varphi_model_eqn.subs({varphi_r(rvec):varphi_rx})
        self.varphi_rx_eqn = varphi_model_eqn

    def define_varphi_related_eqns(self) -> None:
        r"""
        Define further equations related to normal slowness :math:`p`

        Attributes:
            p_varphi_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p = \dfrac{1}{\varphi(\mathbf{r})|\sin\beta|^\eta}`
            p_varphi_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\sqrt{p_{x}^{2} + p_{z}^{2}} = \dfrac{\left(  {\sqrt{p_{x}^{2} + p_{z}^{2}}}  \right)^{\eta}}{\varphi{\left(\mathbf{r} \right)}{p_{x}}^\eta}`
            p_rx_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\sqrt{p_{x}^{2} + p_{z}^{2}}
                = \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu} \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{\eta}{2}}}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`
            p_rx_tanbeta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\sqrt{p_{x}^{2} + \dfrac{p_{x}^{2}}{\tan^{2}{\left(\beta \right)}}}
                = \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu} \left(p_{x}^{2}
                + \dfrac{p_{x}^{2}}{\tan^{2}{\left(\beta \right)}}\right)^{\frac{\eta}{2}}}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`
            px_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{x} = \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu} \left(p_{x}^{2}
                + \dfrac{p_{x}^{2}}{\tan^{2}{\left(\beta \right)}}\right)^{\frac{\eta}{2}} \sin{\left(\beta \right)}}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`
            pz_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{z} = - \dfrac{p_{x}^{- \eta} x_{1}^{2 \mu} \left(p_{x}^{2}
                + \dfrac{p_{x}^{2}}{\tan^{2}{\left(\beta \right)}}\right)^{\frac{\eta}{2}} \cos{\left(\beta \right)}}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)}`
            xiv_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\xi^{\downarrow} = \dfrac{p_z}{p_x^{2} + p_z^{2}}`
            px_varphi_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{x} = \dfrac{\sin{\left(\beta \right)} \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}{\varphi{\left(\mathbf{r} \right)}}`
            pz_varphi_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{z} = - \dfrac{\cos{\left(\beta \right)} \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}{\varphi{\left(\mathbf{r} \right)}}`
            px_varphi_rx_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{x} = \dfrac{\sin{\left(\beta \right)} \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}{\varphi_0 \left(\varepsilon + \left(\dfrac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)}`
            pz_varphi_rx_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{z} = - \dfrac{\cos{\left(\beta \right)} \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}{\varphi_0 \left(\varepsilon + \left(\dfrac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)}`
        """
        self.p_varphi_beta_eqn = self.p_xi_eqn.subs(
            e2d(self.xi_varphi_beta_eqn))
        # Note force px >= 0
        self.p_varphi_pxpz_eqn = (self.p_varphi_beta_eqn
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
        #                                  self.p_varphi_pxpz_eqn.rhs/sqrt(px**2+pz**2)))

        self.p_rx_pxpz_eqn = simplify(
                self.p_varphi_pxpz_eqn.subs(
                    {varphi_r(rvec): self.varphi_rx_eqn.rhs})
        )
        self.p_rx_tanbeta_eqn = self.p_rx_pxpz_eqn.subs(
            {pz: self.pz_px_tanbeta_eqn.rhs})
        self.px_beta_eqn = Eq(px, self.p_rx_tanbeta_eqn.rhs * sin(beta))
        self.pz_beta_eqn = Eq(pz, -self.p_rx_tanbeta_eqn.rhs * cos(beta))
        self.xiv_pxpz_eqn = simplify(Eq(xiv, -cos(beta)/p)
                                     .subs({cos(beta): 1/sqrt(1+tan(beta)**2)})
                                     .subs({self.tanbeta_pxpz_eqn.lhs: self.tanbeta_pxpz_eqn.rhs})
                                     .subs({self.p_norm_pxpz_eqn.lhs: self.p_norm_pxpz_eqn.rhs}))

        tmp = self.xi_varphi_beta_eqn.subs(e2d(self.xi_p_eqn))\
                                     .subs(e2d(self.p_pz_cosbeta_eqn))
        self.pz_varphi_beta_eqn = Eq(pz, solve(tmp, pz)[0])
        tmp = self.pz_varphi_beta_eqn.subs(e2d(self.pz_px_tanbeta_eqn))
        self.px_varphi_beta_eqn = Eq(px, solve(tmp, px)[0])
        self.pz_varphi_rx_beta_eqn = self.pz_varphi_beta_eqn.subs(
            e2d(self.varphi_rx_eqn))
        self.px_varphi_rx_beta_eqn = self.px_varphi_beta_eqn.subs(
            e2d(self.varphi_rx_eqn))

    def define_Fstar_eqns(self) -> None:
        r"""
        Define the fundamental function

        Attributes:
            Okubo_Fstar_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\dfrac{\sqrt{p_{x}^{2} + p_{z}^{2}}}{F^{*}} = \dfrac{p_{x}^{- \eta} \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{\eta}{2}}}{\varphi{\left(\mathbf{r} \right)}}`
            Fstar_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`F^{*} = p_{x}^{\eta} \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{1}{2} - \frac{\eta}{2}} \varphi{\left(\mathbf{r} \right)}`
        """
        # Note force px >= 0
        self.Okubo_Fstar_eqn = simplify(
            Eq(self.p_norm_pxpz_eqn.rhs/Fstar, self.p_varphi_pxpz_eqn.rhs)
            .subs({Abs(px): px, sign(px): 1})
        )
        self.Fstar_eqn \
            = Eq(Fstar, (solve(self.Okubo_Fstar_eqn, Fstar)[0]).subs({varphi_rx(rx): varphi}))\
            .subs({Abs(px): px, sign(px): 1})

    def define_H_eqns(self) -> None:
        r"""
        Define the Hamiltonian


        Attributes:
            H_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`H = \dfrac{p_{x}^{2 \eta} \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta} \varphi^{2}{\left(\mathbf{r} \right)}}{2}`
            H_varphi_rx_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`H = \dfrac{\varphi_0^{2} p_{x}^{2 \eta} x_{1}^{- 4 \mu} \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta} \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2}}{2}`
        """
        self.H_eqn = (Eq(H, simplify(self.Fstar_eqn.rhs**2/2))
                      # .subs({Abs(px):px,sign(px):1})
                      )
        self.H_varphi_rx_eqn \
            = simplify(self.H_eqn.subs(varphi_r(rvec), self.varphi_rx_eqn.rhs))

    def define_rdot_eqns(self) -> None:
        r"""
        Define equations for :math:`\dot{r}`, the rate of change of position



        Attributes:
            rdotx_rdot_alpha_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`v^{x} = v \cos{\left(\alpha \right)}`
            rdotz_rdot_alpha_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`v^{z} = v \sin{\left(\alpha \right)}`
            rdotx_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`v^{x} = p_{x}^{2 \eta - 1} \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} \left(\eta p_{z}^{2} + p_{x}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}`
            rdotz_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`v^{z} = - p_{x}^{2 \eta} p_{z} \left(\eta - 1\right) \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} \varphi^{2}{\left(\mathbf{r} \right)}`
            rdotz_on_rdotx_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\dfrac{v^{z}}{v^{x}} = - \dfrac{p_{x} p_{z} \left(\eta - 1\right)}{\eta p_{z}^{2} + p_{x}^{2}}`
            rdotz_on_rdotx_tanbeta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\dfrac{v^{z}}{v^{x}} =   \dfrac{\left(\eta - 1\right) \tan{\left(\beta \right)}}{\eta + \tan^{2}{\left(\beta \right)}}`
            rdot_vec_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\mathbf{v} = \left[\begin{matrix}p_{x}^{2 \eta - 1} \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} \left(\eta p_{z}^{2} + p_{x}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}\\- p_{x}^{2 \eta} p_{z} \left(\eta - 1\right) \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} \varphi^{2}{\left(\mathbf{r} \right)}\end{matrix}\right]`
            rdot_p_unity_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{x} v^{x} + p_{z} v^{z} = 1`
        """
        self.rdotx_rdot_alpha_eqn = Eq(rdotx, rdot*cos(alpha))
        self.rdotz_rdot_alpha_eqn = Eq(rdotz, rdot*sin(alpha))
        self.rdotx_pxpz_eqn = factor(Eq(rdotx, diff(self.H_eqn.rhs, px)))
        # simplify(diff(self.H_eqn.rhs,px)).subs({Abs(px):px,sign(px):1}) ) )
        self.rdotz_pxpz_eqn = factor(Eq(rdotz, diff(self.H_eqn.rhs, pz)))
        # self.rdotz_pxpz_eqn \
        #     = simplify( simplify( Eq( rdotz, simplify(diff(self.H_eqn.rhs,pz))\
        #                                 .subs({Abs(px):px,sign(px):1}) ) )
        #                                     .subs({px:pxp}) ) \
        #                                         .subs({pxp:px})
        self.rdotz_on_rdotx_eqn \
            = factor(Eq(rdotz/rdotx,
                        simplify((self.rdotz_pxpz_eqn.rhs/self.rdotx_pxpz_eqn.rhs)))
                     .subs({Abs(px): px}))
        self.rdotz_on_rdotx_tanbeta_eqn \
            = factor(self.rdotz_on_rdotx_eqn.subs({px: self.px_pz_tanbeta_eqn.rhs}))
        self.rdot_vec_eqn \
            = Eq(rdotvec, Matrix([self.rdotx_pxpz_eqn.rhs, self.rdotz_pxpz_eqn.rhs]))
        self.rdot_p_unity_eqn = Eq(rdotx*px+rdotz*pz, 1)

    def define_pdot_eqns(self) -> None:
        r"""
        Define equations for :math:`\dot{p}`, the rate of change of normal slowness


        Attributes:
            pdotx_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\dot{p}_x = 2 \mu \varphi_0^{2} p_{x}^{2 \eta} x_{1}^{- 4 \mu} \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta}
                \left(x_{1} - {r}^x\right)^{2 \mu - 1} \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)`
            pdotz_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\dot{p}_z = 0`
            pdot_covec_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\mathbf{\dot{\widetilde{p}}} = \left[\begin{matrix}2 \mu \varphi_0^{2} p_{x}^{2 \eta} x_{1}^{- 4 \mu}
                \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta} \left(x_{1} - {r}^x\right)^{2 \mu - 1}
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right) & 0\end{matrix}\right]`
        """
        self.pdotx_pxpz_eqn \
            = simplify(Eq(pdotx, (-diff(self.H_varphi_rx_eqn.rhs, rx))))\
            .subs({Abs(pz): -pz, Abs(px): px, Abs(px*pz): -px*pz, Abs(px/pz): -px/pz})
        self.pdotz_pxpz_eqn = simplify(
            Eq(pdotz, (0*diff(self.varphi_rx_eqn.rhs, rx)
                       * (-self.tanbeta_pxpz_eqn.rhs)
                       * self.H_eqn.rhs/varphi_r(rvec)))
        )
        self.pdot_covec_eqn \
            = Eq(pdotcovec, Matrix([[self.pdotx_pxpz_eqn.rhs],
                                    [self.pdotz_pxpz_eqn.rhs]]).T)

    def define_Hamiltons_eqns(self) -> None:
        r"""
        Define Hamilton's equations


        Attributes:
            hamiltons_eqns (:class:`sympy.Matrix <sympy.matrices.immutable.ImmutableDenseMatrix>` of :class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\left[\begin{matrix}\
                \dot{r}^x = \varphi_0^{2} p_{x}^{2 \eta - 1} x_{1}^{- 4 \mu} \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} \left(\eta p_{z}^{2} + p_{x}^{2}\right)
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2}\\
                \dot{r}^z = - \varphi_0^{2} p_{x}^{2 \eta} p_{z} x_{1}^{- 4 \mu} \left(\eta - 1\right) \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta}
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2}\\
                \dot{p}_x = 2 \mu \varphi_0^{2} p_{x}^{2 \eta} x_{1}^{- 4 \mu} \left(p_{x}^{2} + p_{z}^{2}\right)^{1 - \eta} \left(x_{1} - {r}^x\right)^{2 \mu - 1}
                \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)\\
                \dot{p}_z = 0
                \end{matrix}\right]`
        """
        self.hamiltons_eqns \
            = Matrix((
                self.rdotx_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn)),
                self.rdotz_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn)),
                self.pdotx_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn)),
                self.pdotz_pxpz_eqn.rhs.subs(e2d(self.varphi_rx_eqn))
                #.subs({pdotz:pdotz_tfn})
            ))

    def nondimensionalize(self) -> None:
        r"""
        Non-dimensionalize variables, Hamiltonian, and Hamilton's equations;
        define dimensionless channel incision number :math:`\mathsf{Ci}`.

        Attributes:
        """
        varsub: Dict = {}

        self.rx_rxhat_eqn = Eq(rx, Lc*rxhat)
        self.rz_rzhat_eqn = Eq(rz, Lc*rzhat)

        self.varepsilon_varepsilonhat_eqn = Eq(varepsilon, Lc*varepsilonhat)
        self.varepsilonhat_varepsilon_eqn = Eq(varepsilonhat,
                                               solve(self.varepsilon_varepsilonhat_eqn, varepsilonhat)[0])

        self.varphi_rxhat_eqn = Eq(varphi_rxhat_fn(rxhat),
                                   factor(self.varphi_rx_eqn.rhs
                                          .subs(e2d(self.rx_rxhat_eqn))
                                          .subs(e2d(self.varepsilon_varepsilonhat_eqn))
                                          .subs(varsub)))

        self.xi_rxhat_eqn = simplify(self.xi_varphi_beta_eqn
                                     .subs({varphi_r(rvec): varphi_rxhat_fn(rxhat)})
                                     .subs(e2d(self.varphi_rxhat_eqn)))

        self.xih0_beta0_eqn = simplify(Eq(xih_0,
                                          (self.xi_rxhat_eqn.rhs / sin(beta_0))
                                          .subs(e2d(self.varphi_rx_eqn))
                                          .subs({Abs(sin(beta)): sin(beta_0)})
                                          .subs({rxhat: 0}).subs(varsub)))
        self.xiv0_beta0_eqn = simplify(Eq(xiv_0,
                                          (self.xi_rxhat_eqn.rhs / cos(beta_0))
                                          .subs(e2d(self.varphi_rx_eqn))
                                          .subs({Abs(sin(beta)): sin(beta_0)})
                                          .subs({rxhat: 0}).subs(varsub)))
        self.xih0_xiv0_beta0_eqn = Eq((xih_0),
                                      xiv_0*simplify((xih_0/xiv_0).subs(e2d(self.xiv0_beta0_eqn))
                                                     .subs(e2d(self.xih0_beta0_eqn))))
        self.xih_xiv_tanbeta_eqn = Eq(xih, xiv/tan(beta))
        self.xiv_xih_tanbeta_eqn = Eq(
            xiv, solve(self.xih_xiv_tanbeta_eqn, xiv)[0])
        #Eq(xiv, xih*tan(beta))

        self.th0_xih0_eqn = Eq(th_0, (Lc/xih_0))
        self.tv0_xiv0_eqn = Eq(tv_0, (Lc/xiv_0))

        self.th0_beta0_eqn = factor(simplify(
                                self.th0_xih0_eqn.subs(e2d(self.xih0_beta0_eqn))))
        self.tv0_beta0_eqn = simplify(
            self.tv0_xiv0_eqn.subs(e2d(self.xiv0_beta0_eqn)))

        self.t_that_eqn = Eq(t, th_0*that)
        self.px_pxhat_eqn = Eq(px, pxhat/xih_0)
        self.pz_pzhat_eqn = Eq(pz, pzhat/xih_0)

        self.pzhat_xiv_eqn \
            = Eq(pzhat, solve(self.pz_xiv_eqn.subs(e2d(self.pz_pzhat_eqn)), pzhat)[0])

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
            = Eq(Ci, deg(solve(self.H_Ci_eqn.subs({H: Rational(1, 2)}), Ci)[0]))
        self.sinCi_xih0_eqn = Eq(sin(Ci), (((sqrt(simplify((H*self.H_split)
                                                           .subs(e2d(self.H_varphi_rxhat_eqn)))))**(1/(1-eta)))))
        self.Ci_xih0_eqn = Eq(Ci, asin(self.sinCi_xih0_eqn.rhs))
        self.sinCi_beta0_eqn = Eq(sin(Ci),
                                  factor(simplify(self.sinCi_xih0_eqn.rhs.subs(
                                      e2d(self.xih0_beta0_eqn))))
                                  .subs({sin(beta_0)*sin(beta_0)**(-eta): (sin(beta_0)**(1-eta))})
                                  .subs({(sin(beta_0)**(1-eta))**(-1/(eta-1)): sin(beta_0)}))
        self.Ci_beta0_eqn = Eq(Ci, asin(self.sinCi_beta0_eqn.rhs))
        self.beta0_Ci_eqn = Eq(beta_0, solve(self.Ci_beta0_eqn, beta_0)[1])

        self.rdotxhat_eqn \
            = Eq(rdotxhat_thatfn(that), simplify(diff(self.H_Ci_eqn.rhs, pxhat)))
        self.rdotzhat_eqn \
            = Eq(rdotzhat_thatfn(that), simplify(diff(self.H_Ci_eqn.rhs, pzhat)))

        self.pdotxhat_eqn \
            = Eq(pdotxhat_thatfn(that), simplify(-diff(self.H_Ci_eqn.rhs, rxhat)))
        self.pdotzhat_eqn \
            = Eq(pdotzhat_thatfn(that), simplify(-diff(self.H_Ci_eqn.rhs, rzhat)))

        self.xih0_Ci_eqn = factor(
            self.xih0_beta0_eqn.subs(e2d(self.beta0_Ci_eqn)))

        # .subs({varepsilonhat:0})
        self.xih0_Lc_varphi0_Ci_eqn = self.xih0_Ci_eqn
        self.xiv0_xih0_Ci_eqn \
            = self.xiv_xih_tanbeta_eqn.subs({xiv: xiv_0, xih: xih_0, beta: beta_0}) \
            .subs(e2d(self.beta0_Ci_eqn))
        #.subs({varepsilonhat:0})
        self.xiv0_Lc_varphi0_Ci_eqn = simplify(self.xiv0_xih0_Ci_eqn
                                               .subs(e2d(self.xih0_Lc_varphi0_Ci_eqn)))
        self.varphi0_Lc_xiv0_Ci_eqn \
            = Eq(varphi_0, solve(self.xiv0_Lc_varphi0_Ci_eqn, varphi_0)[0]) \
            .subs({Abs(cos(Ci)): cos(Ci)})

        self.ratio_xiv0_xih0_eqn \
            = Eq(xiv_0/xih_0, (xiv_0/xih_0).subs(e2d(self.xiv0_xih0_Ci_eqn)))

    def define_nodimensionalized_Hamiltons_eqns(self) -> None:
        """
        TBD
        """
        self.hamiltons_ndim_eqns = Matrix((
                                        self.rdotxhat_eqn.rhs,
                                        self.rdotzhat_eqn.rhs,
                                        self.pdotxhat_eqn.rhs,
                                        self.pdotzhat_eqn.rhs
                                    ))

    def define_tanalpha_eqns(self) -> None:
        r"""
        Define equations for ray angle :math:`\tan(\alpha)`


        Attributes:
            tanalpha_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\alpha \right)} = \dfrac{v^{z}}{v^{x}}`
            tanalpha_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\alpha \right)} = - \dfrac{p_{x} p_{z} \left(\eta - 1\right)}{\eta p_{z}^{2} + p_{x}^{2}}`
            tanalpha_rdot_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\alpha \right)} = \dfrac{\left(\eta - 1\right) \tan{\left(\beta \right)}}{\eta + \tan^{2}{\left(\beta \right)}}`
        """
        self.tanalpha_rdot_eqn \
            = Eq(simplify(self.rdotz_rdot_alpha_eqn.rhs/self.rdotx_rdot_alpha_eqn.rhs),
                 rdotz/rdotx)
        self.tanalpha_pxpz_eqn = self.tanalpha_rdot_eqn \
            .subs({self.rdotz_on_rdotx_eqn.lhs: self.rdotz_on_rdotx_eqn.rhs})
        self.tanalpha_beta_eqn = self.tanalpha_rdot_eqn \
            .subs({self.rdotz_on_rdotx_eqn.lhs: self.rdotz_on_rdotx_tanbeta_eqn.rhs})

    def define_tanbeta_eqns(self) -> None:
        r"""
        Define equations for surface tilt angle :math:`\beta)`




        Attributes:
            tanbeta_alpha_eqns   (list of :class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\left[ \tan{\left(\beta \right)} \
                = \dfrac{\eta - \sqrt{\eta^{2} - 4 \eta \tan^{2}{\left(\alpha \right)} - 2 \eta + 1} - 1}{2 \tan{\left(\alpha \right)}},
                \tan{\left(\beta \right)} = \dfrac{\eta + \sqrt{\eta^{2} - 4 \eta \tan^{2}{\left(\alpha \right)} - 2 \eta + 1} - 1}{2 \tan{\left(\alpha \right)}}\right]`
            tanbeta_alpha_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta \right)}
                = \dfrac{\eta - \sqrt{\eta^{2} - 4 \eta \tan^{2}{\left(\alpha \right)} - 2 \eta + 1} - 1}{2 \tan{\left(\alpha \right)}}`
            tanalpha_ext_eqns   (list of :class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\left[ \tan{\left(\alpha_c \right)} = - \frac{\sqrt{\eta - 2 + \frac{1}{\eta}}}{2},
                \tan{\left(\alpha_c \right)} = \frac{\sqrt{\eta - 2 + \frac{1}{\eta}}}{2}\right]`
            tanalpha_ext_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\alpha_c \right)} = \dfrac{\eta - 1}{2 \sqrt{\eta}}`
            tanbeta_crit_eqns   (list of :class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\left[ \tan{\left(\beta_c \right)} = - \dfrac{\eta - 1}{\sqrt{\eta - 2 + \frac{1}{\eta}}},
                \tan{\left(\beta_c \right)} = \dfrac{\eta - 1}{\sqrt{\eta - 2 + \frac{1}{\eta}}}\right]`
            tanbeta_crit_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta_c \right)} = \sqrt{\eta}`
            tanbeta_rdotxz_pz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta \right)} = \dfrac{v^{z} - \frac{1}{p_{z}}}{v^{x}}`
            tanbeta_rdotxz_xiv_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta \right)} = \dfrac{\xi^{\downarrow} + v^{z}}{v^{x}}`
        """

        # eta_sub = {eta: self.eta_}

        if self.eta_ == 1 and self.beta_type == 'sin':
            print(r'Cannot compute all $\beta$ equations '
                  + r'for $\sin\beta$ model and $\eta=1$')
            return
        solns = solve(self.tanalpha_beta_eqn.subs({tan(alpha): ta}), tan(beta))
        self.tanbeta_alpha_eqns \
            = [Eq(tan(beta), soln.subs({ta: tan(alpha)})) for soln in solns]
        # Bit of a hack - extracts the square root term in tan(beta)
        #    as a fn of tan(alpha), which then gives the critical alpha
        root_terms = [
            [arg_ for arg__ in arg_.args if isinstance(arg__, sy.core.power.Pow)
             or isinstance(arg_, sy.core.power.Pow)]
            for arg_ in numer(self.tanbeta_alpha_eqns[0].rhs).args
            if isinstance(arg_, (sy.core.mul.Mul, sy.core.power.Pow))
        ]
        self.tanalpha_ext_eqns \
            = [Eq(tan(alpha_ext), soln)
                for soln in solve(Eq(root_terms[0][0], 0), tan(alpha))]

        tac_lt1 = simplify((factor(simplify(
                self.tanalpha_ext_eqns[0].rhs*sqrt(eta)
                            ))/sqrt(eta))
                           .subs({Abs(eta-1): 1-eta})
                           )
        tac_gt1 = simplify(
            (factor(
                simplify(self.tanalpha_ext_eqns[1].rhs*sqrt(eta)))/sqrt(eta))
            .subs({Abs(eta-1): eta-1})
        )
        self.tanalpha_ext_eqn \
            = Eq(tan(alpha_ext), Piecewise((tac_lt1, eta < 1), (tac_gt1, True)))

        self.tanbeta_crit_eqns \
            = [factor(tanbeta_alpha_eqn_
                      .subs({beta: beta_crit, alpha: alpha_ext})
                      .subs(e2d(tanalpha_ext_eqn_)))
                for tanalpha_ext_eqn_, tanbeta_alpha_eqn_
                in zip(self.tanalpha_ext_eqns, self.tanbeta_alpha_eqns)]
        # This is a hack, because SymPy simplify can't handle it
        self.tanbeta_crit_eqn \
            = Eq(tan(beta_crit), sqrt(simplify((self.tanbeta_crit_eqns[0].rhs)**2)))

        self.tanbeta_rdotxz_pz_eqn = Eq(tan(beta), (rdotz - 1/pz)/rdotx)
        self.tanbeta_rdotxz_xiv_eqn \
            = self.tanbeta_rdotxz_pz_eqn.subs({pz: self.pz_xiv_eqn.rhs})

        self.tanalpha_ext = float(
            N(self.tanalpha_ext_eqn.rhs.subs({eta: self.eta_})))
        self.tanbeta_crit = float(
            N(self.tanbeta_crit_eqn.rhs.subs({eta: self.eta_})))

    def define_psi_eqns(self) -> None:
        r"""
        Define equations for anisotropy angle :math:`\psi)`

        Attributes:
            psi_alpha_beta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta \right)} = \dfrac{\xi^{\downarrow} + v^{z}}{v^{x}}`
        """
        self.psi_alpha_beta_eqn = Eq(psi, alpha-beta+pi/2)

    def define_g_eqns(self) -> None:
        r"""
        Define equations for the metric tensor :math:`g` and its dual  :math:`g^*`

        Attributes:
            gstar_varphi_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`g^{*} = \left[\begin{matrix}\
                \dfrac{2 p_{x}^{3} \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}} \
                - \dfrac{3 p_{x}^{3} \left(p_{x}^{2} + \dfrac{3 p_{z}^{2}}{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                + \dfrac{2 p_{x} \left(p_{x}^{2} + \dfrac{3 p_{z}^{2}}{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}} \
                & \dfrac{3 p_{x}^{4} p_{z} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                - \dfrac{3 p_{x}^{2} p_{z} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}}\\ \
                \dfrac{3 p_{x}^{2} p_{z} \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}} \
                - \dfrac{3 p_{x}^{2} p_{z} \left(p_{x}^{2} + \dfrac{3 p_{z}^{2}}{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                & \dfrac{3 p_{x}^{3} p_{z}^{2} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                - \dfrac{p_{x}^{3} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}}\
                \end{matrix}\right]`
            det_gstar_varphi_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\det\left(g^*\right) \
                = \dfrac{p_{x}^{4} \left(- \dfrac{p_{x}^{2}}{2} \
                + \dfrac{3 p_{z}^{2}}{4}\right) \varphi^{4}{\left(\mathbf{r} \right)}}{p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}}`
            g_varphi_pxpz_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`g = \left[\begin{matrix}\
                \dfrac{2 \left(p_{x}^{2} - 2 p_{z}^{2}\right) \sqrt{p_{x}^{2} + p_{z}^{2}}}{p_{x} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}} \
                & - \dfrac{6 p_{z}^{3} \sqrt{p_{x}^{2} + p_{z}^{2}}}{p_{x}^{2} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}\\ \
                - \dfrac{6 p_{z}^{3} \sqrt{p_{x}^{2} + p_{z}^{2}}}{p_{x}^{2} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}} \
                & - \dfrac{4 p_{x}^{6} + 14 p_{x}^{4} p_{z}^{2} + 22 p_{x}^{2} p_{z}^{4} + 12 p_{z}^{6}}{p_{x}^{3} \sqrt{p_{x}^{2} + p_{z}^{2}} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}\
                \end{matrix}\right]`
            gstar_eigen_varphi_pxpz   (list of :class:`sympy.Expr <sympy.core.expr.Expr>`) :
                eigenvalues and eigenvectors of :math:`g^{*}` in one object
            gstar_eigenvalues   (:class:`sympy.Matrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                :math:`\left[\begin{matrix}\
                \dfrac{\varphi_0^{2} p_{x} x_{1}^{- 4 \mu} \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2} \left(- 3 \left(p_{x}^{2} + p_{z}^{2}\right) \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + \left(p_{x}^{2} + 6 p_{z}^{2}\right) \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)\right)}{4 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}} \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)}\\\
                \dfrac{\varphi_0^{2} p_{x} x_{1}^{- 4 \mu} \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2} \left(3 \left(p_{x}^{2} + p_{z}^{2}\right) \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + \left(p_{x}^{2} + 6 p_{z}^{2}\right) \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)\right)}{4 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}} \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)}\
                \end{matrix}\right]`
            gstar_eigenvectors   (list containing pair of :class:`sympy.Matrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                :math:`\left[\
                \begin{matrix}\dfrac{p_{x} p_{z}^{3} \left(p_{x}^{6} + 2 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + 2 p_{z}^{6} + \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}\right)}{p_{x}^{10} + 3 p_{x}^{8} p_{z}^{2} + 7 p_{x}^{6} p_{z}^{4} + 11 p_{x}^{4} p_{z}^{6} + p_{x}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 10 p_{x}^{2} p_{z}^{8} + p_{x}^{2} p_{z}^{2} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 4 p_{z}^{10} + 2 p_{z}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}}\\ \
                1 \
                \end{matrix}\
                \right] \\ \
                \left[\
                \begin{matrix}\dfrac{p_{x} p_{z}^{3} \left(p_{x}^{6} + 2 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + 2 p_{z}^{6} - \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}\right)}{p_{x}^{10} + 3 p_{x}^{8} p_{z}^{2} + 7 p_{x}^{6} p_{z}^{4} + 11 p_{x}^{4} p_{z}^{6} - p_{x}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 10 p_{x}^{2} p_{z}^{8} - p_{x}^{2} p_{z}^{2} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 4 p_{z}^{10} - 2 p_{z}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}}\\ \
                1 \
                \end{matrix}\right]`
        """
        self.gstar_varphi_pxpz_eqn = None
        self.det_gstar_varphi_pxpz_eqn = None
        self.g_varphi_pxpz_eqn = None
        self.gstar_eigen_varphi_pxpz = None
        self.gstar_eigenvalues = None
        self.gstar_eigenvectors = None

        eta_sub = {eta: self.eta_}
        self.gstar_varphi_pxpz_eqn \
            = Eq(gstar, factor(
                    Matrix([diff(self.rdot_vec_eqn.rhs, self.p_covec_eqn.rhs[0]).T,
                            diff(self.rdot_vec_eqn.rhs, self.p_covec_eqn.rhs[1]).T])
                )).subs(eta_sub)
        self.det_gstar_varphi_pxpz_eqn \
            = Eq(det_gstar, (simplify(self.gstar_varphi_pxpz_eqn.rhs.subs(eta_sub).det())))
        if self.eta_ == 1 and self.beta_type == 'sin':
            print(r'Cannot compute all metric tensor $g^{ij}$ equations '
                  + r'for $\sin\beta$ model and $\eta=1$')
            return
        self.g_varphi_pxpz_eqn \
            = Eq(g, simplify(self.gstar_varphi_pxpz_eqn.rhs.subs(eta_sub).inverse()))
        self.gstar_eigen_varphi_pxpz = self.gstar_varphi_pxpz_eqn.rhs.eigenvects()
        self.gstar_eigenvalues = simplify(
            Matrix([self.gstar_eigen_varphi_pxpz[0][0],
                    self.gstar_eigen_varphi_pxpz[1][0]])
            .subs({varphi_r(rvec): self.varphi_rx_eqn.rhs}))
        self.gstar_eigenvectors = (
            [simplify(Matrix(self.gstar_eigen_varphi_pxpz[0][2][0])
                      .subs({varphi_r(rvec): self.varphi_rx_eqn.rhs})),
             simplify(Matrix(self.gstar_eigen_varphi_pxpz[1][2][0])
                      .subs({varphi_r(rvec): self.varphi_rx_eqn.rhs}))])

    def define_idtx_fgtx_eqns(self):
        r"""
        Define indicatrix and figuratrix equations



        Attributes:
            pz_cosbeta_varphi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{z}^{4} = \dfrac{\cos^{4}{\left(\beta \right)}}{\varphi^{4} \left(1 - \cos^{2}{\left(\beta \right)}\right)^{3}}`
            cosbetasqrd_pz_varphi_solns  (list of :class:`sympy.Expr <sympy.core.expr.Expr>`) :
                :math:`\left[
                -\frac{ \left(  -6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4}
                + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2\right)^{\frac{2}{3}}
                + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} + 2 \sqrt[3]{2} \right)
                }{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}}
                ,\dots \right]`
            fgtx_cossqrdbeta_pz_varphi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\cos^{2}{\left(\beta \right)}
                = - \frac{- 6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4}
                + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}
                + 2 \sqrt[3]{2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}}`
            fgtx_tanbeta_pz_varphi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta \right)}
                = \sqrt{\frac{- 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2\right)^{\frac{2}{3}}
                + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} + 2 \sqrt[3]{2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}
                + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2\right)^{\frac{2}{3}}
                - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} - 2 \sqrt[3]{2}}}`
            fgtx_px_pz_varphi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{x}
                = - p_{z} \sqrt{\frac{- 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2\right)^{\frac{2}{3}}
                + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} + 2 \sqrt[3]{2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}
                + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2\right)^{\frac{2}{3}}
                - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} - 2 \sqrt[3]{2}}}`
            idtx_rdotx_pz_varphi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`{r}^x = \dfrac{\sqrt{6} \left(- 81 \cdot 2^{\frac{2}{3}} \varphi^{12} p_{z}^{12} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 378 \varphi^{12} p_{z}^{12} - 9 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{10} p_{z}^{10} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 42 \sqrt{3} \varphi^{10} p_{z}^{10} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} + 45 \sqrt[3]{2} \varphi^{8} p_{z}^{8} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 96 \cdot 2^{\frac{2}{3}} \varphi^{8} p_{z}^{8} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 306 \varphi^{8} p_{z}^{8} + \sqrt[3]{2} \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 2 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 6 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 20 \sqrt[3]{2} \varphi^{4} p_{z}^{4} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} - 26 \cdot 2^{\frac{2}{3}} \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 64 \varphi^{4} p_{z}^{4} + 2 \sqrt[3]{2} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 2 \cdot 2^{\frac{2}{3}} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 4\right)}{72 \varphi^{4} p_{z}^{5} \left(\frac{\sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 2 \sqrt[3]{2}}\right)^{\frac{3}{2}} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} \left(- 54 \cdot 2^{\frac{2}{3}} \varphi^{12} p_{z}^{12} - 6 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{10} p_{z}^{10} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} + 6 \varphi^{8} p_{z}^{8} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 33 \sqrt[3]{2} \varphi^{8} p_{z}^{8} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 78 \cdot 2^{\frac{2}{3}} \varphi^{8} p_{z}^{8} + \sqrt[3]{2} \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 2 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 12 \varphi^{4} p_{z}^{4} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} - 18 \sqrt[3]{2} \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 24 \cdot 2^{\frac{2}{3}} \varphi^{4} p_{z}^{4} + 2 \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{2} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 2 \cdot 2^{\frac{2}{3}}\right)}`
            idtx_rdotz_pz_varphi_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`{r}^z = \dfrac{\sqrt{6} \sqrt{\frac{- 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 2 \sqrt[3]{2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 2 \sqrt[3]{2}}} \left(27 \cdot 2^{\frac{2}{3}} \varphi^{8} p_{z}^{8} + 3 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 18 \cdot 2^{\frac{2}{3}} \varphi^{4} p_{z}^{4} + 2 \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{2} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 2 \cdot 2^{\frac{2}{3}}\right)}{72 \varphi^{4} p_{z}^{5} \left(\frac{\sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 2 \sqrt[3]{2}}\right)^{\frac{3}{2}} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} \left(- 6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 2 \sqrt[3]{2}\right)}`
        """
        # if self.eta_ == 2:
        #     pz_tanbeta_varphi_eqn = ( self.pz_p_beta_eqn
        #      .subs({p:self.p_varphi_beta_eqn.rhs})
        #      .subs({varphi_r(rvec):varphi})
        #      .subs({cos(beta):sqrt(1/(1+tan(beta)**2))})
        #      .subs({Abs(tan(beta)):tan(beta)})
        #     )
        #     tanbeta_pz_varphi_solns = solve( pz_tanbeta_varphi_eqn, tan(beta) )
        #     tanbeta_pz_varphi_eqn = Eq(tan(beta), ([soln for soln in tanbeta_pz_varphi_solns
        #                                         if Abs(im(N(soln.subs({varphi:1,pz:1}))))<1e-20][0]) )
        #     self.fgtx_tanbeta_pz_varphi_eqn = tanbeta_pz_varphi_eqn
        #     self.fgtx_cossqrdbeta_pz_varphi_eqn = Eq(cos(beta)**2, 1/(1+tanbeta_pz_varphi_eqn.rhs**2))
        # else:
        eta_sub = {eta: self.eta_}
        pz_cosbeta_varphi_tmp_eqn = (self.pz_p_beta_eqn
                                     .subs({p: self.p_varphi_beta_eqn.rhs})
                                     .subs({varphi_r(rvec): varphi})
                                     .subs(eta_sub)
                                     .subs({Abs(tan(beta)): Abs(sin(beta))/Abs(cos(beta))})
                                     .subs({Abs(cos(beta)): cos(beta), Abs(sin(beta)): sin(beta)})
                                     .subs({sin(beta): sqrt(1-cos(beta)**2)})
                                     )
        pz_cosbeta_varphi_eqn = Eq(pz_cosbeta_varphi_tmp_eqn.lhs**self.eta__dbldenom,
                                   pz_cosbeta_varphi_tmp_eqn.rhs**self.eta__dbldenom)

        # New
        # pz_cosbeta_varphi_eqn \
        # = (self.pz_varphi_beta_eqn.subs({Abs(sin(beta)**self.eta_):sin(beta)**self.eta_})
        #                                  .subs({varphi_r(rvec):varphi})
        #                                  .subs({sin(beta):sqrt(1-cos(beta)**2)}))
        # cosbetasqrd_pz_varphi_soln = (solve( pz_cosbeta_varphi_eqn, cos(beta)**2 ))[0]

        self.pz_cosbeta_varphi_eqn = pz_cosbeta_varphi_eqn
        self.cosbetasqrd_pz_varphi_solns = None
        self.cosbetasqrd_pz_varphi_soln = None
        self.fgtx_cossqrdbeta_pz_varphi_eqn = None
        self.fgtx_tanbeta_pz_varphi_eqn = None
        self.fgtx_px_pz_varphi_eqn = None
        self.idtx_rdotx_pz_varphi_eqn = None
        self.idtx_rdotz_pz_varphi_eqn = None
        self.cosbetasqrd_pz_varphi_solns \
            = solve(self.pz_cosbeta_varphi_eqn, cos(beta)**2)
        if (self.eta_ == Rational(1, 4) or self.eta_ == Rational(3, 2)) \
                and self.beta_type == 'tan':
            print('Cannot compute all indicatrix equations for '
                  + rf'$\tan\beta$ model and $\eta={self.eta_}$')
            return

        def find_cosbetasqrd_root(sub):
            return [soln for soln in self.cosbetasqrd_pz_varphi_solns
                    if Abs(im(N(soln.subs(sub)))) < 1e-20 and (re(N(soln.subs(sub)))) >= 0]

        self.cosbetasqrd_pz_varphi_soln = find_cosbetasqrd_root(
            {varphi: 1, pz: -0.01})
        if self.cosbetasqrd_pz_varphi_soln == []:
            self.cosbetasqrd_pz_varphi_soln = find_cosbetasqrd_root(
                {varphi: 10, pz: -0.5})
        self.fgtx_cossqrdbeta_pz_varphi_eqn \
            = Eq(cos(beta)**2, self.cosbetasqrd_pz_varphi_soln[0])
        self.fgtx_tanbeta_pz_varphi_eqn \
            = Eq(tan(beta), sqrt(1/(self.fgtx_cossqrdbeta_pz_varphi_eqn.rhs)-1))
        self.fgtx_px_pz_varphi_eqn \
            = factor(Eq(px, -pz*self.fgtx_tanbeta_pz_varphi_eqn.rhs))
        g_xx = self.gstar_varphi_pxpz_eqn.rhs[0, 0]
        g_zx = self.gstar_varphi_pxpz_eqn.rhs[1, 0]
        g_xz = self.gstar_varphi_pxpz_eqn.rhs[0, 1]
        g_zz = self.gstar_varphi_pxpz_eqn.rhs[1, 1]
        self.idtx_rdotx_pz_varphi_eqn = factor(
            Eq(rx, (g_xx*px+g_xz*pz)
               .subs({px: self.fgtx_px_pz_varphi_eqn.rhs, varphi_r(rvec): varphi}))
        )
        self.idtx_rdotz_pz_varphi_eqn = factor(factor(
            Eq(rz, (g_zx*px+g_zz*pz)
               .subs({px: self.fgtx_px_pz_varphi_eqn.rhs, varphi_r(rvec): varphi}))
        ))

    def prep_geodesic_eqns(self, parameters: Dict = None):
        r"""
        Define geodesic equations

        Args:
            parameters (dict): dictionary of model parameter values to be used for equation substitutions

        Attributes:
            gstar_ij_tanbeta_mat   (:class:`sympy.ImmutableDenseMatrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                :math:`\dots`
            g_ij_tanbeta_mat   (:class:`sympy.ImmutableDenseMatrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                :math:`\dots`
            tanbeta_poly_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\dots` where :math:`a := \tan\alpha`
            tanbeta_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta \right)} = \dots` where :math:`a := \tan\alpha`
            gstar_ij_tanalpha_mat   (:class:`sympy.ImmutableDenseMatrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                a symmetric tensor with components (using shorthand :math:`a := \tan\alpha`)

                :math:`g^*[1,1] = \dots`

                :math:`g^*[1,2] = g^*[2,1] = \dots`

                :math:`g^*[2,2] = \dots`
            gstar_ij_mat   (:class:`sympy.ImmutableDenseMatrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                a symmetric tensor with components
                (using shorthand :math:`a := \tan\alpha`, and with a particular choice of model parameters)

                :math:`g^*[1,1] = \dots`

                :math:`g^*[1,2] = g^*[2,1] = \dots`

                :math:`g^*[2,2] =  \dots`
            g_ij_tanalpha_mat   (:class:`sympy.ImmutableDenseMatrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                a symmetric tensor with components (using shorthand :math:`a := \tan\alpha`)

                :math:`g[1,1] = \dots`

                :math:`g[1,2] = g[2,1] = \dots`

                :math:`g[2,2] = \dots`
            g_ij_mat   (:class:`sympy.ImmutableDenseMatrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                a symmetric tensor with components
                (using shorthand :math:`a := \tan\alpha`, and with a particular choice of model parameters)

                :math:`g[1,1] =\dots`

                :math:`g[1,2] = g[2,1] = \dots`

                :math:`g[2,2] = \dots`
            g_ij_mat_lambdified   (function) :
                lambdified version of `g_ij_mat`
            gstar_ij_mat_lambdified   (function) :
                lambdified version of `gstar_ij_mat`
        """
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
            = lambdify((rx, rdotx, rdotz, varepsilon), self.g_ij_mat, 'numpy')
        self.gstar_ij_mat_lambdified \
            = lambdify((rx, rdotx, rdotz, varepsilon), self.gstar_ij_mat, 'numpy')

    def define_geodesic_eqns(self):  # , parameters=None
        r"""
        Define geodesic equations

        Args:
            parameters (dict): dictionary of model parameter values to be used for equation substitutions

        Attributes:
            dg_rk_ij_mat   (:class:`sympy.ImmutableDenseMatrix <sympy.matrices.immutable.ImmutableDenseMatrix>`) :
                Derivatives of the components of the metric tensor:
                these values are used to construct the Christoffel tensor.
                Too unwieldy to display here.
            christoffel_ij_k_rx_rdot_lambda   (function) :
                The Christoffel tensor coefficients, as a `lambda` function,
                for each component :math:`r^x`, :math:`{\dot{r}^x}` and :math:`{\dot{r}^z}`.
            christoffel_ij_k_lambda   (function) :
                The Christoffel tensor coefficients, as a `lambda` function, in a compact and indexable form.
            geodesic_eqns (list of :class:`sympy.Eq <sympy.core.relational.Equality>`) :
                Ray geodesic equations, but expressed indirectly as a pair of coupled 1st-order vector ODEs
                rather than a 2nd-order vector ODE for ray acceleration.
                The 1st-order ODE form is easier to solve numerically.

                :math:`\dot{r}^x = v^{x}`

                :math:`\dot{r}^z = v^{z}`

                :math:`\dot{v}^x = \dots`

                :math:`\dot{v}^z = \dots`
            vdotx_lambdified   (function) :
                lambdified version of :math:`\dot{v}^x`
            vdotz_lambdified   (function) :
                lambdified version of :math:`\dot{v}^z`
        """
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

        def dg_ij_rk_lambda(
            i_, j_, k_): return self.dg_rk_ij_mat[k_, 0, i_, j_]

        # Generate Christoffel "symbols" tensor
        def christoffel_ij_k_raw(i_, j_, k_): return [
                                        Rational(1, 2)*gstar_ij_lambda(k_, m_)*(
                                          dg_ij_rk_lambda(m_, i_, j_)
                                            + dg_ij_rk_lambda(m_, j_, i_)
                                            - dg_ij_rk_lambda(i_, j_, m_))
                                    for m_ in [0, 1]]
        # Use of 'factor' here messes things up for eta<1
        self.christoffel_ij_k_rx_rdot_lambda = lambda i_, j_, k_: \
            (reduce(lambda a, b: a+b, christoffel_ij_k_raw(i_, j_, k_)))
        christoffel_ij_k_rx_rdot_list = [[[
            lambdify((rx, rdotx, rdotz, varepsilon),
                     self.christoffel_ij_k_rx_rdot_lambda(i_, j_, k_))
            for i_ in [0, 1]]
                                              for j_ in [0, 1]]
                                             for k_ in [0, 1]]
        self.christoffel_ij_k_lambda \
            = lambda i_, j_, k_, varepsilon_: christoffel_ij_k_rx_rdot_list[i_][j_][k_]

        # Obtain geodesic equations as a set of coupled 1st order ODEs
        self.geodesic_eqns = Matrix([
            rdotx,
            rdotz,
            # Use symmetry to abbreviate sum of diagonal terms
            (-self.christoffel_ij_k_rx_rdot_lambda(0, 0, 0)*rdotx*rdotx
             - 2*self.christoffel_ij_k_rx_rdot_lambda(0, 1, 0)*rdotx*rdotz
             #-christoffel_ij_k_rx_rdot_lambda(1,0,0)*rdotz*rdotx
             - self.christoffel_ij_k_rx_rdot_lambda(1, 1, 0)*rdotz*rdotz),
            # Use symmetry to abbreviate sum of diagonal terms
            (-self.christoffel_ij_k_rx_rdot_lambda(0, 0, 1)*rdotx*rdotx
             - 2*self.christoffel_ij_k_rx_rdot_lambda(0, 1, 1)*rdotx*rdotz
             #-christoffel_ij_k_rx_rdot_lambda(1,0,1)*rdotz*rdotx
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
            = lambdify((rx, rdotx, rdotz, varepsilon), (self.geodesic_eqns[2]), 'numpy')
        self.vdotz_lambdified \
            = lambdify((rx, rdotx, rdotz, varepsilon), (self.geodesic_eqns[3]), 'numpy')

    def define_px_poly_eqn(self, eta_choice: Rational = None, do_ndim: bool = False):
        r"""
        TODO: remove ref to xiv_0

        Define polynomial form of function combining normal-slowness covector components :math:`(p_x,p_z)`
        (where the latter is given in terms of the vertical erosion rate :math:`\xi^{\downarrow} = -\dfrac{1}{p_z}`)
        and the erosion model flow component :math:`\varphi(\mathbf{r})`


        Args:
            eta_choice (:class:`sympy.Rational <sympy.core.numbers.Rational>`):
                value of :math:`\eta` to use instead value given at instantiation; otherwise the latter value is used

        Attributes:
            poly_px_xiv_varphi_eqn   (:class:`sympy.Poly <sympy.polys.polytools.Poly>`) :
                :math:`\operatorname{Poly}{\left( \left(\xi^{\downarrow}\right)^{4} \varphi^{4}{\left(\mathbf{r} \right)} p_{x}^{6}
                -  \left(\xi^{\downarrow}\right)^{4} p_{x}^{2} -  \left(\xi^{\downarrow}\right)^{2}, p_{x},
                domain=\mathbb{Z}\left[\varphi{\left(\mathbf{r} \right)}, \xi^{\downarrow}\right] \right)}`
            poly_px_xiv_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\varphi_0^{4} \left(\xi^{\downarrow{0}}\right)^{4} p_{x}^{6} \left(\varepsilon
                + \left(\frac{x_{1} - {r}^x}{x_{1}}\right)^{2 \mu}\right)^{4}
                - \left(\xi^{\downarrow{0}}\right)^{4} p_{x}^{2}
                - \left(\xi^{\downarrow{0}}\right)^{2} = 0`
        """
        if do_ndim:
            # Non-dimensionalized version
            varphi0_solns = solve(self.sinCi_xih0_eqn.subs(
                {eta: eta_choice}), varphi_0)
            varphi0_eqn = Eq(varphi_0, varphi0_solns[0])
            if eta_choice is not None and eta_choice <= 1:
                tmp_eqn = separatevars(simplify(
                    self.px_xiv_varphi_eqn
                    .subs({eta: eta_choice})
                    .subs({varphi_r(rvec): self.varphi_rxhat_eqn.rhs})
                    .subs(e2d(self.px_pxhat_eqn))
                    .subs(e2d(varphi0_eqn)))
                )
                self.poly_pxhat_xiv_eqn = simplify(
                    Eq((numer(tmp_eqn.lhs)-denom(tmp_eqn.lhs)*(tmp_eqn.rhs))/xiv**2, 0))
                self.poly_pxhat_xiv0_eqn \
                    = self.poly_pxhat_xiv_eqn\
                    .subs({xiv: xiv_0})\
                    .subs(e2d(self.xiv0_xih0_Ci_eqn))
            else:
                tmp_eqn = separatevars(simplify(
                    self.px_xiv_varphi_eqn
                    .subs({eta: eta_choice})
                    .subs({varphi_r(rvec): self.varphi_rxhat_eqn.rhs})
                    .subs(e2d(self.px_pxhat_eqn))
                    .subs(e2d(varphi0_eqn)))
                )
                self.poly_pxhat_xiv_eqn = simplify(
                    Eq((numer(tmp_eqn.lhs)-denom(tmp_eqn.lhs)*(tmp_eqn.rhs))/xiv**2, 0)
                )
                self.poly_pxhat_xiv0_eqn = simplify(
                    Eq(self.poly_pxhat_xiv_eqn.lhs
                        .subs({xiv: xiv_0})
                        .subs(e2d(self.xiv0_xih0_Ci_eqn))/xih_0**2, 0)
                )
        else:
            # Dimensioned version
            tmp_eqn = simplify(self.px_xiv_varphi_eqn.subs({eta: eta_choice}))
            if eta_choice is not None and eta_choice <= 1:
                self.poly_px_xiv_varphi_eqn = poly(tmp_eqn.lhs, px)
            else:
                self.poly_px_xiv_varphi_eqn \
                    = poly(numer(tmp_eqn.lhs)-denom(tmp_eqn.lhs)*(tmp_eqn.rhs), px)
            self.poly_px_xiv_eqn \
                = Eq(self.poly_px_xiv_varphi_eqn.subs(e2d(self.varphi_rx_eqn)), 0)

    def prep_ibc_eqns(self):
        r"""
        Define boundary (ray initial) condition equations

        Attributes:
            pz0_xiv0_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{z_0} = - \dfrac{1}{\xi^{\downarrow{0}}}`
            pzpx_unity_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\varphi^{2} p_{x}^{2} p_{x}^{2 \eta} \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} + \varphi^{2} p_{x}^{2 \eta} p_{z}^{2} \left(p_{x}^{2} + p_{z}^{2}\right)^{- \eta} = 1`
        """
        self.pz0_xiv0_eqn = Eq(pz_0, (-1/xiv_0))
        self.pzpx_unity_eqn = expand(simplify(
            self.rdot_p_unity_eqn
                .subs({rdotx: self.rdotx_pxpz_eqn.rhs, rdotz: self.rdotz_pxpz_eqn.rhs})
                .subs({varphi_r(rvec): varphi})
        )).subs({Abs(pz): -pz})

    def define_ibc_eqns(self):
        r"""
        Define initial profile equations

        Attributes:
            boundary_eqns   (`dict` of :class:`sympy.Eq <sympy.core.relational.Equality>`) :

                'planar':
                    'h': :math:`h = \dfrac{h_{0} x}{x_{1}}`

                    'gradh': :math:`\dfrac{d}{d x} h{\left(x \right)} = \dfrac{h_{0}}{x_{1}}`

                'convex-up':
                    'h': :math:`h = \dfrac{h_{0} \tanh{\left(\dfrac{\kappa_\mathrm{h} x}{x_{1}} \right)}}{\tanh{\left(\dfrac{\kappa_\mathrm{h}}{x_{1}} \right)}}`

                    'gradh': :math:`\dfrac{d}{d x} h{\left(x \right)} = \dfrac{\kappa_\mathrm{h} h_{0} \left(1 - \tanh^{2}{\left(\dfrac{\kappa_\mathrm{h} x}{x_{1}} \right)}\right)}{x_{1} \tanh{\left(\dfrac{\kappa_\mathrm{h}}{x_{1}} \right)}}`

                'concave-up':
                    'h': :math:`h = h_{0} \left(1 + \dfrac{\tanh{\left(\dfrac{\kappa_\mathrm{h} x}{x_{1}} - \kappa_\mathrm{h} \right)}}{\tanh{\left(\dfrac{\kappa_\mathrm{h}}{x_{1}} \right)}}\right)`

                    'gradh': :math:`\dfrac{d}{d x} h{\left(x \right)} = \dfrac{\kappa_\mathrm{h} h_{0} \left(1 - \tanh^{2}{\left(\dfrac{\kappa_\mathrm{h} x}{x_{1}} - \kappa_\mathrm{h} \right)}\right)}{x_{1} \tanh{\left(\dfrac{\kappa_\mathrm{h}}{x_{1}} \right)}}`
        """
        self.boundary_eqns = {
            'planar': {'h': Eq(h, (h_0*x/Lc))},
            'convex-up': {'h': simplify(Eq(h,
                                           h_0*tanh(kappa_h*x/Lc)/tanh(kappa_h/Lc)))},
            'concave-up': {'h': simplify(Eq(h,
                                            h_0+h_0*tanh(-kappa_h*(Lc-x)/Lc)/tanh(kappa_h/Lc)))}
        }
        # Math concave-up not geo concave-up, i.e., with minimum
        # Math convex-up not geo convex-up, i.e., with maximum
        for ibc_type in ['planar', 'convex-up', 'concave-up']:
            self.boundary_eqns[ibc_type].update({
                'gradh': Eq(diff(h_fn, x), diff(self.boundary_eqns[ibc_type]['h'].rhs, x))
                })

    def set_ibc_eqns(self):
        r"""
        Define initial condition equations

        Attributes:
            rz_initial_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`{r}^z = \dfrac{h_{0} \tanh{\left(\frac{\kappa_\mathrm{h} {r}^x}{x_{1}} \right)}}{\tanh{\left(\frac{\kappa_\mathrm{h}}{x_{1}} \right)}}`
            tanbeta_initial_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`\tan{\left(\beta \right)} = \dfrac{\kappa_\mathrm{h} h_{0} \left(1 - \tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)}\right)}{x_{1} \tanh{\left(\frac{\kappa_\mathrm{h}}{x_{1}} \right)}}`
            p_initial_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p = \dfrac{x_{1}^{2 \mu} \left|{\sin{\left(\beta \right)}}\right|^{- \eta}}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} + \left(- x + x_{1}\right)^{2 \mu}\right)}`
            px_initial_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{x} = \dfrac{\kappa_\mathrm{h} h_{0} x_{1}^{2 \mu} \left(\dfrac{1}{\kappa_\mathrm{h} h_{0} \left|{\tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)} - 1}\right|}\right)^{\eta} \left(\kappa_\mathrm{h}^{2} h_{0}^{2} \left(\tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)} - 1\right)^{2} + x_{1}^{2} \tanh^{2}{\left(\frac{\kappa_\mathrm{h}}{x_{1}} \right)}\right)^{\frac{\eta}{2} - \frac{1}{2}} \left|{\tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)} - 1}\right|}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} + \left(- x + x_{1}\right)^{2 \mu}\right)}`
            pz_initial_eqn   (:class:`sympy.Eq <sympy.core.relational.Equality>`) :
                :math:`p_{z} = - \dfrac{x_{1}^{2 \mu + 1} \left(\dfrac{1}{\kappa_\mathrm{h} h_{0} \left|{\tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)} - 1}\right|}\right)^{\eta} \left(\kappa_\mathrm{h}^{2} h_{0}^{2} \left(\tanh^{2}{\left(\frac{\kappa_\mathrm{h} x}{x_{1}} \right)} - 1\right)^{2} + x_{1}^{2} \tanh^{2}{\left(\frac{\kappa_\mathrm{h}}{x_{1}} \right)}\right)^{\frac{\eta}{2} - \frac{1}{2}} \tanh{\left(\frac{\kappa_\mathrm{h}}{x_{1}} \right)}}{\varphi_0 \left(\varepsilon x_{1}^{2 \mu} + \left(- x + x_{1}\right)^{2 \mu}\right)}`
        """
        cosbeta_eqn = Eq(cos(beta), 1/sqrt(1+tan(beta)**2))
        sinbeta_eqn = Eq(sin(beta), sqrt(1-1/(1+tan(beta)**2)))
        # sintwobeta_eqn = Eq(sin(2*beta), cos(beta)**2-sin(beta)**2)

        ibc_type = self.ibc_type
        self.rz_initial_eqn = self.boundary_eqns[ibc_type]['h'].subs({
                                                                     h: rz, x: rx})
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


class EquationSubset:
    """
    TBD
    """

    def __init__(self,
                 gmeq: Type[Equations],
                 parameters: Dict,
                 do_ndim: bool = False,
                 do_revert: bool = True) -> None:
        """
        TBD
        """
        sub = parameters.copy()
        if do_revert and do_ndim:
            undimsub = {pxhat: px, pzhat: pz, rxhat: rx, xivhat: xiv,
                        varphi_rhat: varphi_rx, varphi_rxhat_fn: varphi_rx,
                        rdotxhat_thatfn: rdotx_tfn, rdotzhat_thatfn: rdotz_tfn,
                        pdotxhat_thatfn: pdotx_tfn, pdotzhat_thatfn: pdotz_tfn}
        else:
            undimsub = {}
        sub.update({mu: gmeq.mu_, eta: gmeq.eta_})
        # xisub = {}
        # varphi0_xiv0_Lc_eqn = (gmeq.varphi0_Lc_xiv0_Ci_eqn
        #                .subs(omitdict(sub,[varphi_0,xiv_0,Lc]))
        #                .n() )
        # self.varphi_rx_eqn   = (gmeq.varphi_rxhat_eqn.subs(e2d(varphi0_xiv0_Lc_eqn))
        #                      .subs(omitdict(sub,[varphi_0,xiv_0,Lc])).n().subs(undimsub)
        #                      #gmeq.varphi_rxhat_eqn
        #                         if do_ndim else gmeq.varphi_rx_eqn.subs(sub).n()
        #                            .subs(undimsub) )
        self.pz_xiv_eqn = ((gmeq.pzhat_xiv_eqn  # .subs(xisub)
                            if do_ndim else gmeq.pz_xiv_eqn).n()).subs(undimsub)
        #.subs({pz:pz_0, xiv:xiv_0}) #.subs({xiv:xiv/xih_0})
        #Eq(simplify((gmeq.poly_pxhat_xiv_eqn.lhs.subs(xisub))/xih_0**2),0)
        self.poly_px_xiv0_eqn = (gmeq.poly_pxhat_xiv0_eqn
                                 if do_ndim else gmeq.poly_px_xiv_eqn) \
            .subs(sub).n().subs(undimsub).subs({xih_0: 1})
        self.xiv0_xih0_Ci_eqn = gmeq.xiv0_xih0_Ci_eqn\
                                    .subs(omitdict(sub, [xiv_0, xih_0])).n()
        self.hamiltons_eqns = (gmeq.hamiltons_ndim_eqns if do_ndim
                               else gmeq.hamiltons_eqns).subs(sub).n().subs(undimsub)


#

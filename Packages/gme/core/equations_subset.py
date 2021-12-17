"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`sympy`
  -  :mod:`gmplib`
  -  :mod:`gme`

---------------------------------------------------------------------

"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable
import warnings

# Typing
from typing import Dict, Type

# SymPy
# from sympy import Eq

# GMPLib
from gmplib.utils import omitdict

# GME
from gme.core.symbols import pxhat, pzhat, rxhat, xivhat, px, pz, rx, xiv, \
                             varphi_rhat, varphi_rxhat_fn, varphi_rx, \
                             rdotxhat_thatfn, rdotzhat_thatfn, \
                             pdotxhat_thatfn, pdotzhat_thatfn, \
                             rdotx_tfn, rdotz_tfn, pdotx_tfn, pdotz_tfn, \
                             mu, eta, xih_0, xiv_0
from gme.core.equations import Equations

warnings.filterwarnings("ignore")

__all__ = ['EquationSubset']


class EquationSubset:
    """
    TBD
    """

    def __init__(
        self,
        gmeq: Type[Equations],
        parameters: Dict,
        do_ndim: bool = False,
        do_revert: bool = True
    ) -> None:
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
        # self.varphi_rx_eqn
        #   = (gmeq.varphi_rxhat_eqn.subs(e2d(varphi0_xiv0_Lc_eqn))
        #
        #   .subs(omitdict(sub,[varphi_0,xiv_0,Lc])).n().subs(undimsub)
        #                      #gmeq.varphi_rxhat_eqn
        #
        #  if do_ndim else gmeq.varphi_rx_eqn.subs(sub).n()
        #                            .subs(undimsub) )
        self.pz_xiv_eqn \
            = ((gmeq.pzhat_xiv_eqn  # .subs(xisub)
                if do_ndim else gmeq.pz_xiv_eqn).n()).subs(undimsub)
        # .subs({pz:pz_0, xiv:xiv_0}) #.subs({xiv:xiv/xih_0})
        # Eq(simplify((gmeq.poly_pxhat_xiv_eqn.lhs.subs(xisub))/xih_0**2),0)
        self.poly_px_xiv0_eqn = (gmeq.poly_pxhat_xiv0_eqn
                                 if do_ndim else gmeq.poly_px_xiv_eqn) \
            .subs(sub).n().subs(undimsub).subs({xih_0: 1})
        self.xiv0_xih0_Ci_eqn = gmeq.xiv0_xih0_Ci_eqn\
                                    .subs(omitdict(sub, [xiv_0, xih_0])).n()
        self.hamiltons_eqns \
            = (gmeq.hamiltons_ndim_eqns if do_ndim
               else gmeq.hamiltons_eqns).subs(sub).n().subs(undimsub)


#

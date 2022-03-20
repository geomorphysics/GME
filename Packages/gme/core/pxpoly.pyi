from sympy import Eq, Rational

class PxpolyMixin:
    sinCi_xih0_eqn: Eq
    px_xiv_varphi_eqn: Eq
    varphi_rxhat_eqn: Eq
    px_pxhat_eqn: Eq
    xiv0_xih0_Ci_eqn: Eq
    varphi_rx_eqn: Eq
    poly_pxhat_xiv_eqn: Eq
    poly_pxhat_xiv0_eqn: Eq
    poly_px_xiv_varphi_eqn: Eq
    poly_px_xiv_eqn: Eq
    poly_px_xiv0_eqn: Eq
    def define_px_poly_eqn(self, eta_choice: Rational = ..., do_ndim: bool = ...) -> None: ...

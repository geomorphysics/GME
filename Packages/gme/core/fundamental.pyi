from sympy import Eq

class FundamentalMixin:
    p_norm_pxpz_eqn: Eq
    p_varphi_pxpz_eqn: Eq
    varphi_rx_eqn: Eq
    Okubo_Fstar_eqn: Eq
    Fstar_eqn: Eq
    H_eqn: Eq
    H_varphi_rx_eqn: Eq
    def define_Fstar_eqns(self) -> None: ...
    def define_H_eqns(self) -> None: ...

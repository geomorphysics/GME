from sympy import Eq
from typing import Any

class FundamentalMixin:
    p_norm_pxpz_eqn: Eq
    p_varphi_pxpz_eqn: Eq
    varphi_rx_eqn: Eq
    Okubo_Fstar_eqn: Any
    Fstar_eqn: Any
    def define_Fstar_eqns(self) -> None: ...
    H_eqn: Any
    H_varphi_rx_eqn: Any
    def define_H_eqns(self) -> None: ...

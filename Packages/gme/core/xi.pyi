from sympy import Eq
from typing import Any

class XiMixin:
    beta_type: str
    do_raw: bool
    eta_: float
    tanbeta_pxpz_eqn: Eq
    cosbeta_pxpz_eqn: Eq
    sinbeta_pxpz_eqn: Eq
    xi_p_eqn: Any
    xiv_pz_eqn: Any
    p_xi_eqn: Any
    pz_xiv_eqn: Any
    def define_xi_eqns(self) -> None: ...
    xi_varphi_beta_raw_eqn: Any
    xi_varphi_beta_eqn: Any
    def define_xi_model_eqn(self) -> None: ...
    xiv_varphi_pxpz_eqn: Any
    px_xiv_varphi_eqn: Any
    eta__dbldenom: Any
    def define_xi_related_eqns(self) -> None: ...

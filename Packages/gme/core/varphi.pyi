from sympy import Eq
from typing import Any

class VarphiMixin:
    varphi_type: str
    mu_: float
    p_xi_eqn: Eq
    xi_varphi_beta_eqn: Eq
    tanbeta_pxpz_eqn: Eq
    sinbeta_pxpz_eqn: Eq
    p_norm_pxpz_eqn: Eq
    pz_px_tanbeta_eqn: Eq
    xi_p_eqn: Eq
    p_pz_cosbeta_eqn: Eq
    varphi_model_ramp_eqn: Any
    varphi_model_sramp_eqn: Any
    varphi_model_srampmu_eqn: Any
    varphi_rx_eqn: Any
    def define_varphi_model_eqns(self) -> None: ...
    p_varphi_beta_eqn: Any
    p_varphi_pxpz_eqn: Any
    p_rx_pxpz_eqn: Any
    p_rx_tanbeta_eqn: Any
    px_beta_eqn: Any
    pz_beta_eqn: Any
    xiv_pxpz_eqn: Any
    pz_varphi_beta_eqn: Any
    px_varphi_beta_eqn: Any
    pz_varphi_rx_beta_eqn: Any
    px_varphi_rx_beta_eqn: Any
    def define_varphi_related_eqns(self) -> None: ...

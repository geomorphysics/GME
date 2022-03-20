from sympy import Eq

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
    varphi_model_ramp_eqn: Eq
    varphi_model_sramp_eqn: Eq
    varphi_model_srampmu_eqn: Eq
    varphi_rx_eqn: Eq
    p_varphi_beta_eqn: Eq
    p_varphi_pxpz_eqn: Eq
    p_rx_pxpz_eqn: Eq
    p_rx_tanbeta_eqn: Eq
    px_beta_eqn: Eq
    pz_beta_eqn: Eq
    xiv_pxpz_eqn: Eq
    pz_varphi_beta_eqn: Eq
    px_varphi_beta_eqn: Eq
    pz_varphi_rx_beta_eqn: Eq
    px_varphi_rx_beta_eqn: Eq
    def define_varphi_model_eqns(self) -> None: ...
    def define_varphi_related_eqns(self) -> None: ...

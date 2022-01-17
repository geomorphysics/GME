from typing import Any

class RpMixin:
    p_covec_eqn: Any
    px_p_beta_eqn: Any
    pz_p_beta_eqn: Any
    p_norm_pxpz_eqn: Any
    tanbeta_pxpz_eqn: Any
    sinbeta_pxpz_eqn: Any
    cosbeta_pxpz_eqn: Any
    pz_px_tanbeta_eqn: Any
    px_pz_tanbeta_eqn: Any
    p_pz_cosbeta_eqn: Any
    def define_p_eqns(self) -> None: ...
    rx_r_alpha_eqn: Any
    rz_r_alpha_eqn: Any
    def define_r_eqns(self) -> None: ...

from sympy import Eq

class RpMixin:
    p_covec_eqn: Eq
    px_p_beta_eqn: Eq
    pz_p_beta_eqn: Eq
    p_norm_pxpz_eqn: Eq
    tanbeta_pxpz_eqn: Eq
    sinbeta_pxpz_eqn: Eq
    cosbeta_pxpz_eqn: Eq
    pz_px_tanbeta_eqn: Eq
    px_pz_tanbeta_eqn: Eq
    p_pz_cosbeta_eqn: Eq
    rx_r_alpha_eqn: Eq
    rz_r_alpha_eqn: Eq
    def define_p_eqns(self) -> None: ...
    def define_r_eqns(self) -> None: ...

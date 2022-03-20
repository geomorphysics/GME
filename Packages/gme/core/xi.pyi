from sympy import Eq

class XiMixin:
    beta_type: str
    do_raw: bool
    eta_: float
    tanbeta_pxpz_eqn: Eq
    cosbeta_pxpz_eqn: Eq
    sinbeta_pxpz_eqn: Eq
    xi_p_eqn: Eq
    xiv_pz_eqn: Eq
    p_xi_eqn: Eq
    pz_xiv_eqn: Eq
    xi_varphi_beta_raw_eqn: Eq
    xi_varphi_beta_eqn: Eq
    xiv_varphi_pxpz_eqn: Eq
    px_xiv_varphi_eqn: Eq
    eta__dbldenom: Eq
    def define_xi_eqns(self) -> None: ...
    def define_xi_model_eqn(self) -> None: ...
    def define_xi_related_eqns(self) -> None: ...

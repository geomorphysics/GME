from sympy import Eq

class IdtxMixin:
    eta_: float
    beta_type: str
    pz_p_beta_eqn: Eq
    p_varphi_beta_eqn: Eq
    gstar_varphi_pxpz_eqn: Eq
    pz_cosbeta_varphi_eqn: Eq
    cosbeta_pz_varphi_solns: Eq
    cosbeta_pz_varphi_soln: Eq
    fgtx_cosbeta_pz_varphi_eqn: Eq
    fgtx_tanbeta_pz_varphi_eqn: Eq
    fgtx_px_pz_varphi_eqn: Eq
    idtx_rdotx_pz_varphi_eqn: Eq
    idtx_rdotz_pz_varphi_eqn: Eq
    def define_idtx_fgtx_eqns(self) -> None: ...

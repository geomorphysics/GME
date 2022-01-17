from sympy import Eq
from typing import Any

class IdtxMixin:
    eta_: float
    beta_type: str
    pz_p_beta_eqn: Eq
    p_varphi_beta_eqn: Eq
    gstar_varphi_pxpz_eqn: Eq
    pz_cosbeta_varphi_eqn: Any
    cosbeta_pz_varphi_solns: Any
    cosbeta_pz_varphi_soln: Any
    fgtx_cosbeta_pz_varphi_eqn: Any
    fgtx_tanbeta_pz_varphi_eqn: Any
    fgtx_px_pz_varphi_eqn: Any
    idtx_rdotx_pz_varphi_eqn: Any
    idtx_rdotz_pz_varphi_eqn: Any
    def define_idtx_fgtx_eqns(self) -> None: ...

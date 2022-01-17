from sympy import Eq
from typing import Any

class AnglesMixin:
    eta_: float
    beta_type: str
    rdotz_on_rdotx_eqn: Eq
    rdotz_rdot_alpha_eqn: Eq
    rdotx_rdot_alpha_eqn: Eq
    rdotz_on_rdotx_tanbeta_eqn: Eq
    pz_xiv_eqn: Eq
    tanalpha_rdot_eqn: Any
    tanalpha_pxpz_eqn: Any
    tanalpha_beta_eqn: Any
    def define_tanalpha_eqns(self) -> None: ...
    tanbeta_alpha_eqns: Any
    tanalpha_ext_eqns: Any
    tanalpha_ext_eqn: Any
    tanbeta_crit_eqns: Any
    tanbeta_crit_eqn: Any
    tanbeta_rdotxz_pz_eqn: Any
    tanbeta_rdotxz_xiv_eqn: Any
    tanalpha_ext: Any
    tanbeta_crit: Any
    def define_tanbeta_eqns(self) -> None: ...
    psi_alpha_beta_eqn: Any
    psi_alpha_eta_eqns: Any
    psi_eta_beta_lambdas: Any
    def define_psi_eqns(self) -> None: ...

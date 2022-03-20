from sympy import Eq

class AnglesMixin:
    eta_: float
    beta_type: str
    rdotz_on_rdotx_eqn: Eq
    rdotz_rdot_alpha_eqn: Eq
    rdotx_rdot_alpha_eqn: Eq
    rdotz_on_rdotx_tanbeta_eqn: Eq
    pz_xiv_eqn: Eq
    tanalpha_rdot_eqn: Eq
    tanalpha_pxpz_eqn: Eq
    tanalpha_beta_eqn: Eq
    tanbeta_alpha_eqns: Eq
    tanalpha_ext_eqns: Eq
    tanalpha_ext_eqn: Eq
    tanbeta_crit_eqns: Eq
    tanbeta_crit_eqn: Eq
    tanbeta_rdotxz_pz_eqn: Eq
    tanbeta_rdotxz_xiv_eqn: Eq
    tanalpha_ext: Eq
    tanbeta_crit: Eq
    psi_alpha_beta_eqn: Eq
    psi_alpha_eta_eqns: Eq
    psi_eta_beta_lambdas: Eq
    def define_tanalpha_eqns(self) -> None: ...
    def define_tanbeta_eqns(self) -> None: ...
    def define_psi_eqns(self) -> None: ...

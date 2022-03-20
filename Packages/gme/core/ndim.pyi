from sympy import Eq

class NdimMixin:
    varphi_rx_eqn: Eq
    xi_varphi_beta_eqn: Eq
    pz_xiv_eqn: Eq
    H_varphi_rx_eqn: Eq
    rx_rxhat_eqn: Eq
    rz_rzhat_eqn: Eq
    varepsilon_varepsilonhat_eqn: Eq
    varepsilonhat_varepsilon_eqn: Eq
    varphi_rxhat_eqn: Eq
    xi_rxhat_eqn: Eq
    xih0_beta0_eqn: Eq
    xiv0_beta0_eqn: Eq
    xih0_xiv0_beta0_eqn: Eq
    xih_xiv_tanbeta_eqn: Eq
    xiv_xih_tanbeta_eqn: Eq
    th0_xih0_eqn: Eq
    tv0_xiv0_eqn: Eq
    th0_beta0_eqn: Eq
    tv0_beta0_eqn: Eq
    t_that_eqn: Eq
    px_pxhat_eqn: Eq
    pz_pzhat_eqn: Eq
    pzhat_xiv_eqn: Eq
    H_varphi_rxhat_eqn: Eq
    H_split: Eq
    H_Ci_eqn: Eq
    degCi_H0p5_eqn: Eq
    sinCi_xih0_eqn: Eq
    Ci_xih0_eqn: Eq
    sinCi_beta0_eqn: Eq
    Ci_beta0_eqn: Eq
    beta0_Ci_eqn: Eq
    rdotxhat_eqn: Eq
    rdotzhat_eqn: Eq
    pdotxhat_eqn: Eq
    pdotzhat_eqn: Eq
    xih0_Ci_eqn: Eq
    xih0_Lc_varphi0_Ci_eqn: Eq
    xiv0_xih0_Ci_eqn: Eq
    xiv0_Lc_varphi0_Ci_eqn: Eq
    varphi0_Lc_xiv0_Ci_eqn: Eq
    ratio_xiv0_xih0_eqn: Eq
    hamiltons_ndim_eqns: Eq
    def nondimensionalize(self) -> None: ...
    def define_nodimensionalized_Hamiltons_eqns(self) -> None: ...

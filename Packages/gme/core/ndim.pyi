from sympy import Eq
from typing import Any

class NdimMixin:
    varphi_rx_eqn: Eq
    xi_varphi_beta_eqn: Eq
    pz_xiv_eqn: Eq
    H_varphi_rx_eqn: Eq
    rx_rxhat_eqn: Any
    rz_rzhat_eqn: Any
    varepsilon_varepsilonhat_eqn: Any
    varepsilonhat_varepsilon_eqn: Any
    varphi_rxhat_eqn: Any
    xi_rxhat_eqn: Any
    xih0_beta0_eqn: Any
    xiv0_beta0_eqn: Any
    xih0_xiv0_beta0_eqn: Any
    xih_xiv_tanbeta_eqn: Any
    xiv_xih_tanbeta_eqn: Any
    th0_xih0_eqn: Any
    tv0_xiv0_eqn: Any
    th0_beta0_eqn: Any
    tv0_beta0_eqn: Any
    t_that_eqn: Any
    px_pxhat_eqn: Any
    pz_pzhat_eqn: Any
    pzhat_xiv_eqn: Any
    H_varphi_rxhat_eqn: Any
    H_split: Any
    H_Ci_eqn: Any
    degCi_H0p5_eqn: Any
    sinCi_xih0_eqn: Any
    Ci_xih0_eqn: Any
    sinCi_beta0_eqn: Any
    Ci_beta0_eqn: Any
    beta0_Ci_eqn: Any
    rdotxhat_eqn: Any
    rdotzhat_eqn: Any
    pdotxhat_eqn: Any
    pdotzhat_eqn: Any
    xih0_Ci_eqn: Any
    xih0_Lc_varphi0_Ci_eqn: Any
    xiv0_xih0_Ci_eqn: Any
    xiv0_Lc_varphi0_Ci_eqn: Any
    varphi0_Lc_xiv0_Ci_eqn: Any
    ratio_xiv0_xih0_eqn: Any
    def nondimensionalize(self) -> None: ...
    hamiltons_ndim_eqns: Any
    def define_nodimensionalized_Hamiltons_eqns(self) -> None: ...

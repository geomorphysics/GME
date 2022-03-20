from sympy import Eq, Matrix
from typing import Callable, Optional

class HamiltonsMixin:
    H_eqn: Eq
    px_pz_tanbeta_eqn: Eq
    H_varphi_rx_eqn: Eq
    varphi_rx_eqn: Eq
    tanbeta_pxpz_eqn: Eq
    vdotx_lambdified: Optional[Callable]
    vdotz_lambdified: Optional[Callable]
    rdotx_rdot_alpha_eqn: Eq
    rdotz_rdot_alpha_eqn: Eq
    rdotx_pxpz_eqn: Eq
    rdotz_pxpz_eqn: Eq
    rdotz_on_rdotx_eqn: Eq
    rdotz_on_rdotx_tanbeta_eqn: Eq
    rdot_vec_eqn: Eq
    rdot_p_unity_eqn: Eq
    pdotx_pxpz_eqn: Eq
    pdotz_pxpz_eqn: Eq
    pdot_covec_eqn: Eq
    hamiltons_eqns: Matrix
    geodesic_eqns: Matrix
    def define_rdot_eqns(self) -> None: ...
    def define_pdot_eqns(self) -> None: ...
    def define_Hamiltons_eqns(self) -> None: ...

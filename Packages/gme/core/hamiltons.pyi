from sympy import Eq
from typing import Any, Callable, Optional

class HamiltonsMixin:
    H_eqn: Eq
    px_pz_tanbeta_eqn: Eq
    H_varphi_rx_eqn: Eq
    varphi_rx_eqn: Eq
    tanbeta_pxpz_eqn: Eq
    vdotx_lambdified: Optional[Callable]
    vdotz_lambdified: Optional[Callable]
    rdotx_rdot_alpha_eqn: Any
    rdotz_rdot_alpha_eqn: Any
    rdotx_pxpz_eqn: Any
    rdotz_pxpz_eqn: Any
    rdotz_on_rdotx_eqn: Any
    rdotz_on_rdotx_tanbeta_eqn: Any
    rdot_vec_eqn: Any
    rdot_p_unity_eqn: Any
    def define_rdot_eqns(self) -> None: ...
    pdotx_pxpz_eqn: Any
    pdotz_pxpz_eqn: Any
    pdot_covec_eqn: Any
    def define_pdot_eqns(self) -> None: ...
    hamiltons_eqns: Any
    def define_Hamiltons_eqns(self) -> None: ...

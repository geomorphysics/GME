from sympy import Eq
from typing import Any

class IbcMixin:
    pz0_xiv0_eqn: Eq
    pzpx_unity_eqn: Eq
    rdot_p_unity_eqn: Eq
    rdotx_pxpz_eqn: Eq
    rdotz_pxpz_eqn: Eq
    ibc_type: str
    p_varphi_beta_eqn: Eq
    varphi_rx_eqn: Eq
    def prep_ibc_eqns(self) -> None: ...
    boundary_eqns: Any
    def define_ibc_eqns(self) -> None: ...
    rz_initial_eqn: Any
    tanbeta_initial_eqn: Any
    p_initial_eqn: Any
    px_initial_eqn: Any
    pz_initial_eqn: Any
    def set_ibc_eqns(self) -> None: ...

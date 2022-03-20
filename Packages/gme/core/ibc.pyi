from sympy import Eq

class IbcMixin:
    pz0_xiv0_eqn: Eq
    pzpx_unity_eqn: Eq
    rdot_p_unity_eqn: Eq
    rdotx_pxpz_eqn: Eq
    rdotz_pxpz_eqn: Eq
    ibc_type: str
    p_varphi_beta_eqn: Eq
    varphi_rx_eqn: Eq
    boundary_eqns: Eq
    rz_initial_eqn: Eq
    tanbeta_initial_eqn: Eq
    p_initial_eqn: Eq
    px_initial_eqn: Eq
    pz_initial_eqn: Eq
    def prep_ibc_eqns(self) -> None: ...
    def define_ibc_eqns(self) -> None: ...
    def set_ibc_eqns(self) -> None: ...

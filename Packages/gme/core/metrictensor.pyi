from sympy import Eq

class MetricTensorMixin:
    eta_: float
    beta_type: str
    rdot_vec_eqn: Eq
    p_covec_eqn: Eq
    varphi_rx_eqn: Eq
    gstar_varphi_pxpz_eqn: Eq
    det_gstar_varphi_pxpz_eqn: Eq
    g_varphi_pxpz_eqn: Eq
    gstar_eigen_varphi_pxpz: Eq
    gstar_eigenvalues: Eq
    gstar_eigenvectors: Eq
    gstarhat_eqn: Eq
    def define_g_eqns(self) -> None: ...

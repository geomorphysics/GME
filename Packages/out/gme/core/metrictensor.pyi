from sympy import Eq
from typing import Any

class MetricTensorMixin:
    eta_: float
    beta_type: str
    rdot_vec_eqn: Eq
    p_covec_eqn: Eq
    varphi_rx_eqn: Eq
    gstar_varphi_pxpz_eqn: Any
    det_gstar_varphi_pxpz_eqn: Any
    g_varphi_pxpz_eqn: Any
    gstar_eigen_varphi_pxpz: Any
    gstar_eigenvalues: Any
    gstar_eigenvectors: Any
    def define_g_eqns(self) -> None: ...

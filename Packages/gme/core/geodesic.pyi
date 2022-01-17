from sympy import Eq
from typing import Any, Dict

class GeodesicMixin:
    eta_: float
    mu_: float
    H_eqn: Eq
    px_pz_tanbeta_eqn: Eq
    tanalpha_beta_eqn: Eq
    tanalpha_rdot_eqn: Eq
    varphi_rx_eqn: Eq
    gstar_ij_tanbeta_mat: Any
    g_ij_tanbeta_mat: Any
    tanbeta_poly_eqn: Any
    tanbeta_eqn: Any
    gstar_ij_tanalpha_mat: Any
    gstar_ij_mat: Any
    g_ij_tanalpha_mat: Any
    g_ij_mat: Any
    g_ij_mat_lambdified: Any
    gstar_ij_mat_lambdified: Any
    def prep_geodesic_eqns(self, parameters: Dict = ...): ...
    dg_rk_ij_mat: Any
    christoffel_ij_k_rx_rdot_lambda: Any
    christoffel_ij_k_lambda: Any
    geodesic_eqns: Any
    vdotx_lambdified: Any
    vdotz_lambdified: Any
    def define_geodesic_eqns(self): ...

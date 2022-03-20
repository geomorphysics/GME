from sympy import Eq, Matrix
from typing import Callable, Dict, Optional

class GeodesicMixin:
    eta_: float
    mu_: float
    H_eqn: Eq
    px_pz_tanbeta_eqn: Eq
    tanalpha_beta_eqn: Eq
    tanalpha_rdot_eqn: Eq
    varphi_rx_eqn: Eq
    gstar_ij_tanbeta_mat: Matrix
    g_ij_tanbeta_mat: Matrix
    tanbeta_poly_eqn: Eq
    tanbeta_eqn: Eq
    gstar_ij_tanalpha_mat: Matrix
    gstar_ij_mat: Matrix
    g_ij_tanalpha_mat: Matrix
    g_ij_mat: Matrix
    g_ij_mat_lambdified: Optional[Callable]
    gstar_ij_mat_lambdified: Optional[Callable]
    dg_rk_ij_mat: Matrix
    christoffel_ij_k_rx_rdot_lambda: Optional[Callable]
    christoffel_ij_k_lambda: Optional[Callable]
    geodesic_eqns: Matrix
    vdotx_lambdified: Optional[Callable]
    vdotz_lambdified: Optional[Callable]
    def prep_geodesic_eqns(self, parameters: Dict = ...): ...
    def define_geodesic_eqns(self): ...

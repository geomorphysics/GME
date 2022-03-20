from sympy import Eq

class ProfileMixin:
    xiv_eqn: Eq
    xvi_abs_eqn: Eq
    dzdx_Ci_polylike_eqn: Eq
    dzdx_polylike_eqn: Eq
    dzdx_Ci_polylike_prelim_eqn: Eq
    def define_z_eqns(self) -> None: ...

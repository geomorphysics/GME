from gme.core.equations import Equations
from sympy import Eq
from typing import Dict, Type

class EquationSubset:
    pz_xiv_eqn: Eq
    poly_px_xiv0_eqn: Eq
    xiv0_xih0_Ci_eqn: Eq
    hamiltons_eqns: Eq
    def __init__(
        self,
        gmeq: Type[Equations],
        parameters: Dict,
        do_ndim: bool = ...,
        do_revert: bool = ...,
    ) -> None: ...

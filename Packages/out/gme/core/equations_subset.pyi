from gme.core.equations import Equations
from typing import Any, Dict, Type

class EquationSubset:
    pz_xiv_eqn: Any
    poly_px_xiv0_eqn: Any
    xiv0_xih0_Ci_eqn: Any
    hamiltons_eqns: Any
    def __init__(self, gmeq: Type[Equations], parameters: Dict, do_ndim: bool = ..., do_revert: bool = ...) -> None: ...

from gme.core.angles import AnglesMixin
from gme.core.fundamental import FundamentalMixin
from gme.core.hamiltons import HamiltonsMixin
from gme.core.metrictensor import MetricTensorMixin
from gme.core.ndim import NdimMixin
from gme.core.pxpoly import PxpolyMixin
from gme.core.rp import RpMixin
from gme.core.varphi import VarphiMixin
from gme.core.xi import XiMixin
from sympy import Rational
from typing import Any, Dict, Optional

class EquationsBase:
    eta_: Any
    mu_: Any
    beta_type: Any
    varphi_type: Any
    do_raw: Any
    def __init__(self, eta_: Rational = ..., mu_: Rational = ..., beta_type: str = ..., varphi_type: str = ..., do_raw: bool = ...) -> None: ...

class EquationsMixedIn(EquationsBase, RpMixin, XiMixin, VarphiMixin, FundamentalMixin, HamiltonsMixin, NdimMixin, AnglesMixin, MetricTensorMixin, PxpolyMixin):
    def __init__(self, **kwargs) -> None: ...

class Equations(EquationsMixedIn):
    def __init__(self, parameters: Optional[Dict] = ..., **kwargs) -> None: ...
